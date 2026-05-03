#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "opencv-python-headless>=4.10",
#   "Pillow>=11.0",
#   "numpy>=2.0",
#   "pytesseract>=0.3.13",
#   "python-magic>=0.4.27",
#   "PyYAML>=6.0",
#   "watchdog>=5.0",
#   "httpx[http2]>=0.27",
# ]
# ///
"""One-shot calibration sweep for the storage encoder.

Runs the cleanup pipeline (YellowRemover → BackgroundFlattener →
BackgroundSnapper → EdgeCleaner) on every JPEG in ``corpus/invoices/good/``, then
encodes each result under a sweep of candidate settings and reports
output bytes, background uniformity, and edge-halo per setting.

The strictest setting that satisfies (background stdev ≈ 0,
edge halo low, size ≤ 300 KB on every sample) becomes the production
setting baked into ``Encoder``.

This script is NOT shipped in the container. Parallel — fans out per-image
work across (cpu_count // 3) workers so it finishes in minutes, not hours.
"""

import io
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Make the single-file docscanner importable in BOTH parent and workers.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


SIZE_CAP_BYTES = 300_000

# Source scans are ~300 dpi (~8.4 MP at 2538×3324). Storing all of those
# pixels blows past the 300 KB cap regardless of encoder; downsample to
# 200 dpi (≈3.7 MP) for storage. 200 dpi is a real-world scanner default
# and reprints cleanly on laser without visible loss for invoice-grade
# content.
TARGET_DPI: int = 200
SOURCE_DPI: int = 300
DOWNSAMPLE_FACTOR: float = TARGET_DPI / SOURCE_DPI


@dataclass(frozen=True)
class Setting:
    name: str
    fmt: str  # "JPEG" | "PNG"
    quality: int = 0  # JPEG quality
    chroma: str = "4:2:0"  # "4:4:4" | "4:2:0"
    grayscale: bool = False


SETTINGS: list[Setting] = [
    *[
        Setting(f"jpeg_q{q}_color_{c}", "JPEG", quality=q, chroma=c, grayscale=False)
        for q in (92, 88, 85, 82, 78, 75, 70)
        for c in ("4:4:4", "4:2:0")
    ],
    *[
        Setting(f"jpeg_q{q}_gray", "JPEG", quality=q, chroma="n/a", grayscale=True)
        for q in (92, 88, 85, 82, 78, 75, 70)
    ],
    Setting("png_color", "PNG", grayscale=False),
    Setting("png_gray", "PNG", grayscale=True),
]


@dataclass(frozen=True)
class Result:
    path: Path
    setting_name: str
    n_bytes: int
    bg_stdev: float
    edge_halo: float
    payload: bytes | None  # only populated for the first sample (visual review)


def _encode(snapped: np.ndarray, setting: Setting) -> bytes:
    """Encode the snapped image under the given setting."""
    if setting.grayscale:
        arr = snapped if snapped.ndim == 2 else cv2.cvtColor(snapped, cv2.COLOR_BGR2GRAY)
        mode = "L"
    else:
        if snapped.ndim == 2:
            arr = cv2.cvtColor(snapped, cv2.COLOR_GRAY2RGB)
        else:
            arr = cv2.cvtColor(snapped, cv2.COLOR_BGR2RGB)
        mode = "RGB"
    img = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    match setting.fmt:
        case "JPEG":
            kw: dict = {
                "format": "JPEG",
                "quality": setting.quality,
                "optimize": True,
                "progressive": True,
            }
            if mode == "RGB":
                kw["subsampling"] = 0 if setting.chroma == "4:4:4" else 2
            img.save(buf, **kw)
        case "PNG":
            img.save(buf, format="PNG", optimize=True, compress_level=9)
        case other:
            raise ValueError(f"unknown format {other}")
    return buf.getvalue()


def _bg_stdev(snapped: np.ndarray, fg_mask: np.ndarray) -> float:
    bg = snapped[fg_mask == 0]
    return float(bg.std()) if bg.size else 0.0


def _edge_halo(decoded: np.ndarray, fg_mask: np.ndarray) -> float:
    """Mean deviation from pure 255 in the 2-pixel ring around foreground.

    The reference (snapped) image has pure-white background; an ideal
    encoding preserves that. JPEG halos around text leak grey into the
    ring, raising the score. Higher = more contamination of the clean
    white background.
    """
    if decoded.ndim == 3:
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
    fg_dilated = cv2.dilate(fg_mask, np.ones((5, 5), np.uint8))
    ring = (fg_dilated > 0) & (fg_mask == 0)
    if ring.sum() == 0:
        return 0.0
    deviation = 255 - decoded[ring].astype(np.int32)
    return float(np.abs(deviation).mean())


def _process_one(path_str: str, capture_payload: bool) -> list[Result]:
    """Worker entrypoint: prepare one image, encode under all settings."""
    # Pin BLAS / OpenCV thread counts so workers don't multiply the load.
    cv2.setNumThreads(1)
    sys.path.insert(0, str(PROJECT_ROOT))
    from docscanner import (
        BackgroundFlattener,
        BackgroundSnapper,
        EdgeCleaner,
        YellowRemover,
    )

    path = Path(path_str)
    bgr = cv2.imread(str(path))
    if bgr is None:
        return []
    cleaned = YellowRemover().remove(bgr)
    # Downsample BEFORE flatten/snap so the background-snap runs at storage
    # resolution and preserves the pure-255 guarantee in the encoded output.
    new_w = int(cleaned.shape[1] * DOWNSAMPLE_FACTOR)
    new_h = int(cleaned.shape[0] * DOWNSAMPLE_FACTOR)
    cleaned = cv2.resize(cleaned, (new_w, new_h), interpolation=cv2.INTER_AREA)
    flat = BackgroundFlattener().flatten(cleaned)
    snapped, fg_mask = BackgroundSnapper().snap(flat)
    snapped = EdgeCleaner().clean(snapped, fg_mask)
    bg_stdev = _bg_stdev(snapped, fg_mask)

    out: list[Result] = []
    for setting in SETTINGS:
        payload = _encode(snapped, setting)
        decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        halo = _edge_halo(decoded, fg_mask)
        out.append(
            Result(
                path=path,
                setting_name=setting.name,
                n_bytes=len(payload),
                bg_stdev=bg_stdev,
                edge_halo=halo,
                payload=payload if capture_payload else None,
            )
        )
    return out


def _worker_count() -> int:
    return max(1, (os.cpu_count() or 1) // 3)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: calibrate_storage.py <sample-dir> [<output-dir>]", file=sys.stderr)
        return 2
    sample_dir = Path(argv[1])
    out_dir = Path(argv[2]) if len(argv) > 2 else PROJECT_ROOT / "tmp" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_paths = sorted(sample_dir.glob("*.jpg"))
    if not sample_paths:
        print(f"no .jpg samples in {sample_dir}", file=sys.stderr)
        return 1

    workers = _worker_count()
    print(
        f"Calibrating {len(sample_paths)} sample(s) × {len(SETTINGS)} settings "
        f"using {workers} worker(s)…",
        file=sys.stderr,
    )

    by_setting: dict[str, list[Result]] = {s.name: [] for s in SETTINGS}
    completed = 0
    ctx = get_context("forkserver")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = {
            pool.submit(_process_one, str(p), i == 0): p
            for i, p in enumerate(sample_paths)
        }
        for fut in as_completed(futures):
            results = fut.result()
            for r in results:
                by_setting[r.setting_name].append(r)
                # Save first sample's encoded outputs for visual inspection.
                if r.payload is not None:
                    ext = "jpg" if "jpeg" in r.setting_name else "png"
                    (out_dir / f"{r.setting_name}.{ext}").write_bytes(r.payload)
            completed += 1
            if completed % 25 == 0 or completed == len(sample_paths):
                print(f"  …{completed}/{len(sample_paths)}", file=sys.stderr)

    # Summary table.
    print(
        f"\n{'setting':<28} {'max_kB':>8} {'mean_kB':>8} "
        f"{'over_cap':>9} {'max_halo':>9}"
    )
    rows: list[tuple[Setting, int, float, int, float]] = []
    for setting in SETTINGS:
        results = by_setting[setting.name]
        sizes = [r.n_bytes for r in results]
        halos = [r.edge_halo for r in results]
        over_cap = sum(1 for s in sizes if s > SIZE_CAP_BYTES)
        rows.append(
            (setting, max(sizes), statistics.mean(sizes), over_cap, max(halos))
        )
        print(
            f"{setting.name:<28} {max(sizes)/1024:>8.1f} "
            f"{statistics.mean(sizes)/1024:>8.1f} {over_cap:>9d} "
            f"{max(halos):>9.2f}"
        )

    # Hybrid strategy (per the user's spec): JPEG if it fits ≤300 KB on this
    # particular file, else fall back to fixed-setting PNG grayscale.
    # Pick the JPEG candidate that maximizes "files shipped as JPEG" subject
    # to halo ≤ 5 (visually clean).
    print(f"\n{'JPEG candidate':<28} {'jpeg_files':>10} {'png_files':>10} "
          f"{'hybrid_max_kB':>13} {'hybrid_mean_kB':>14}")
    by_path: dict[str, dict[str, Result]] = {}
    for setting in SETTINGS:
        for r in by_setting[setting.name]:
            by_path.setdefault(str(r.path), {})[setting.name] = r
    png_fallback = "png_gray"
    hybrid_rows: list[tuple[Setting, int, float, int, int]] = []
    for setting in SETTINGS:
        if setting.fmt != "JPEG":
            continue
        max_size = max((row[4] for row in rows if row[0].name == setting.name), default=0)
        max_halo = next(row[4] for row in rows if row[0].name == setting.name)
        if max_halo > 5.0:
            continue
        sizes_hybrid: list[int] = []
        n_jpeg = n_png = 0
        for path_str, settings_results in by_path.items():
            jpeg_size = settings_results[setting.name].n_bytes
            if jpeg_size <= SIZE_CAP_BYTES:
                sizes_hybrid.append(jpeg_size)
                n_jpeg += 1
            else:
                sizes_hybrid.append(settings_results[png_fallback].n_bytes)
                n_png += 1
        hybrid_rows.append(
            (setting, max(sizes_hybrid), statistics.mean(sizes_hybrid), n_jpeg, n_png)
        )
        print(
            f"{setting.name:<28} {n_jpeg:>10d} {n_png:>10d} "
            f"{max(sizes_hybrid)/1024:>13.1f} {statistics.mean(sizes_hybrid)/1024:>14.1f}"
        )

    # Winner: prefer the highest-quality JPEG (cleanest reproduction for files
    # that fit) — most files will fall to PNG anyway, so JPEG quality doesn't
    # cost much in storage, only in halo.
    if hybrid_rows:
        winner = max(hybrid_rows, key=lambda r: r[0].quality)
        s, hybrid_max, hybrid_mean, n_jpeg, n_png = winner
        print()
        print(f"WINNER (hybrid): JPEG={s.name}  PNG fallback={png_fallback}")
        print(
            f"  jpeg quality={s.quality} chroma={s.chroma} gray={s.grayscale}"
        )
        print(
            f"  files: {n_jpeg} JPEG / {n_png} PNG  |  "
            f"hybrid max={hybrid_max/1024:.1f} KB  mean={hybrid_mean/1024:.1f} KB"
        )
        return 0

    print("NO JPEG candidate had clean halos. Always-PNG fallback.")
    png_row = next(r for r in rows if r[0].name == png_fallback)
    print(
        f"PNG-only: max={png_row[1]/1024:.1f} KB mean={png_row[2]/1024:.1f} KB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
