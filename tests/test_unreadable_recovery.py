"""Population-level recovery test for the v1-failed corpus.

Asserts that the v2 OCR cascade (full-page region + alt PSMs + rotation)
recovers ≥ 70 of the files in `corpus/invoices/unreadable/` that v1 could not read.
Slow (~40 s); marked `slow` so the default `pytest tests/` skips it.
Run with `pytest -m slow` to include it.

Floor of 70 derived from the actual corpus:
  * Original 99-file population: baseline (v2) was 43 / 99.
  * User curated 8 obvious-junk files into `corpus/invoices/unreadable/junk/`,
    leaving 91 in the main directory.
  * v2 cascade recovers 73 / 91 = 80% of the remaining corpus.
  * Floor of 70 catches a regression of more than ~4 files without
    being so tight that a single tail flake fails CI.
"""

import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import cv2
import pytest

from docscanner import OcrEngine, OcrPreparer, Region, YellowRemover

pytestmark = pytest.mark.slow


REGEX = re.compile(r"R?INV/20\d{2}/\d{4,5}")
PRIMARY_REGIONS: tuple[Region, ...] = (Region(60, 0, 100, 25), Region(20, 30, 80, 75))
PRIMARY_CONFIG = "--psm 6 -l eng"
FALLBACK_REGIONS: tuple[tuple[Region, ...], ...] = ((Region(0, 0, 100, 100),),)
FALLBACK_CONFIGS: tuple[str, ...] = ("--psm 11 -l eng", "--psm 12 -l eng")
RECOVERY_FLOOR = 70


def _try_one(path_str: str) -> tuple[str, str | None, int]:
    """Worker: returns (filename, matched_name | None, rotation_degrees)."""
    cv2.setNumThreads(1)
    bgr = cv2.imread(path_str)
    if bgr is None:
        return (Path(path_str).name, None, 0)
    # YellowRemover requires color BGR; the unreadable PNGs are 1-bit
    # which cv2.imread reads as 3-channel grayscale. Pass through if it
    # raises (it shouldn't, but defensively).
    try:
        cleaned = YellowRemover().remove(bgr)
    except ValueError:
        cleaned = bgr
    binary = OcrPreparer().binarize(cleaned)
    engine = OcrEngine()
    result = engine.extract_with_fallbacks(
        binary,
        primary_regions=PRIMARY_REGIONS,
        primary_config=PRIMARY_CONFIG,
        fallback_regions=FALLBACK_REGIONS,
        fallback_configs=FALLBACK_CONFIGS,
        try_rotation=True,
        regex=REGEX,
    )
    if result is None:
        return (Path(path_str).name, None, 0)
    return (Path(path_str).name, result.match.name, result.rotation_degrees)


def test_v2_recovers_at_least_70_of_unreadable(project_root: Path) -> None:
    unreadable_dir = project_root / "corpus" / "invoices" / "unreadable"
    paths = sorted(p for p in unreadable_dir.glob("*.png") if p.is_file())
    if len(paths) < 80:
        pytest.skip(
            f"corpus/invoices/unreadable/ has {len(paths)} files; recovery floor "
            "was derived against the ~91-file curated corpus"
        )

    workers = max(1, (os.cpu_count() or 1) // 3)
    ctx = get_context("forkserver")
    hits: list[tuple[str, str, int]] = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = {pool.submit(_try_one, str(p)): p for p in paths}
        for fut in as_completed(futures):
            name, matched, rotation = fut.result()
            if matched is not None:
                hits.append((name, matched, rotation))

    print(f"\nRecovered {len(hits)}/{len(paths)} from corpus/invoices/unreadable/")
    if hits:
        rotated = [h for h in hits if h[2] != 0]
        print(f"  {len(rotated)} required rotation")
    assert len(hits) >= RECOVERY_FLOOR, (
        f"Recovery regression: {len(hits)} hits < floor {RECOVERY_FLOOR}. "
        "If this drop is intentional (e.g. the corpus changed), update "
        "RECOVERY_FLOOR. Otherwise inspect the cascade for regressions."
    )
