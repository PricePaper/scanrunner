"""Contract tests for StoragePreparer and Encoder (v3 layered pipeline).

These verify the production reproduction guarantee end-to-end on real
corpus/invoices/good/ samples: pure-white background, faithful foreground,
single calibrated encoder setting (8-level quantized PNG).

StoragePreparer in v3 takes the ORIGINAL bgr scan (no YellowRemover
preprocessing); the substrate-aware DocumentDecomposer handles yellow
paper natively via estimate_paper_tone.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from docscanner import (
    Encoder,
    ForegroundQuantizer,
    StoragePreparer,
)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class TestEncoder:
    def test_returns_png_for_any_input(self) -> None:
        gray = np.full((100, 100), 255, dtype=np.uint8)
        payload, mime = Encoder().encode(gray)
        assert mime == "image/png"
        assert payload[:8] == b"\x89PNG\r\n\x1a\n"

    def test_quantization_levels_are_calibrated_constant(self) -> None:
        """The chosen levels (8) and cap (300 KB) come from the
        2026-05-03 sweep on 168 corpus pages. Q8 was the only candidate
        hitting ≥90 % fit; if anyone changes either value, they must
        re-run the calibration sweep first.
        """
        assert ForegroundQuantizer.DEFAULT_LEVELS == 8
        assert Encoder.SIZE_CAP_BYTES == 300_000

    def test_deterministic_output_across_runs(self) -> None:
        rng = np.random.default_rng(seed=42)
        gray = rng.integers(220, 256, size=(500, 500), dtype=np.uint8)
        a, _ = Encoder().encode(gray)
        b, _ = Encoder().encode(gray)
        assert a == b, "Encoder output must be byte-identical for same input"

    def test_foreground_alphabet_is_quantized(self) -> None:
        """Q8 collapses the foreground (pixels < BACKGROUND_THRESHOLD)
        to 8 evenly-spaced bins. The decoded image must contain at most
        9 distinct foreground levels (8 bins + boundary).
        """
        # A grayscale ramp guarantees we hit lots of foreground levels.
        ramp = np.tile(np.arange(256, dtype=np.uint8), (100, 1))
        payload, _ = Encoder().encode(ramp)
        decoded = cv2.imdecode(
            np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
        )
        foreground = decoded[decoded < ForegroundQuantizer.BACKGROUND_THRESHOLD]
        assert foreground.size > 0
        assert len(np.unique(foreground)) <= 9, (
            f"foreground alphabet {len(np.unique(foreground))} > 9 "
            "— quantization didn't run"
        )


# ---------------------------------------------------------------------------
# StoragePreparer (integration over real samples)
# ---------------------------------------------------------------------------


class TestStoragePreparer:
    def test_produces_encoded_bytes_for_real_invoice(
        self, first_good_invoice: Path
    ) -> None:
        bgr = cv2.imread(str(first_good_invoice))
        payload, mime = StoragePreparer().prepare(bgr)
        assert mime == "image/png"
        assert len(payload) > 0

    def test_decoded_storage_image_is_majority_pure_white(
        self, first_good_invoice: Path
    ) -> None:
        """The reproduction contract: the OVERWHELMING majority of pixels
        in the decoded output are pure 255.

        v3 layered model produces an explicitly white canvas (paper is
        not a layer; whatever printed/handwritten don't claim is rendered
        as 255 by Document.composite()). At least 85 % of pixels must be
        pure 255 in the encoded output.
        """
        bgr = cv2.imread(str(first_good_invoice))
        payload, _ = StoragePreparer().prepare(bgr)
        decoded = cv2.imdecode(
            np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
        )
        pure_white_fraction: float = float((decoded == 255).sum() / decoded.size)
        assert pure_white_fraction >= 0.85, (
            f"only {pure_white_fraction:.1%} of pixels are pure 255 "
            "— the layered composite left too much foreground"
        )

    def test_storage_image_long_side_caps_at_1920(
        self, first_good_invoice: Path
    ) -> None:
        """Storage image's longer side must be ≤ 1920 px so Odoo's
        default ``base.image_autoresize_max_px = 1920x1920`` does not
        downsample-and-re-encode our v3 PNG server-side. See
        StoragePreparer.MAX_LONG_SIDE_PX.
        """
        bgr = cv2.imread(str(first_good_invoice))
        payload, _ = StoragePreparer().prepare(bgr)
        decoded = cv2.imdecode(
            np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
        )
        src_h, src_w = bgr.shape[:2]
        out_h, out_w = decoded.shape[:2]
        assert out_w < src_w and out_h < src_h
        assert max(out_h, out_w) <= StoragePreparer.MAX_LONG_SIDE_PX, (
            f"long side {max(out_h, out_w)}px exceeds Odoo's autoresize limit"
        )

    def test_foreground_uses_quantized_grayscale_palette(
        self, first_good_invoice: Path
    ) -> None:
        """Foreground must NOT collapse to pure black-or-white — Q8
        keeps multiple grayscale tones for signature legibility — but it
        also must not exceed the calibrated 8-level alphabet.
        """
        bgr = cv2.imread(str(first_good_invoice))
        payload, _ = StoragePreparer().prepare(bgr)
        decoded = cv2.imdecode(
            np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
        )
        foreground = decoded[decoded < ForegroundQuantizer.BACKGROUND_THRESHOLD]
        assert foreground.size > 100
        unique_levels = len(np.unique(foreground))
        assert 1 < unique_levels <= 9, (
            f"foreground has {unique_levels} unique levels — expected "
            "2..9 (Q8 alphabet plus near-background boundary)"
        )


# ---------------------------------------------------------------------------
# Parametrized sample sweep — guards against regressions on the population.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_all_samples_produce_valid_output(good_invoice_paths: list[Path]) -> None:
    """Every sample must round-trip through the storage pipeline.

    Marked @slow because the v3 layered model invokes DocTR per page
    (~5-15 s/page on CPU). Across 168 samples this is ~30 min. Run
    explicitly with `pytest -m slow`.

    The encoder is calibrated so ≥ 90 % of corpus pages fit ≤ 300 KB.
    We assert that fit ratio holds, and that every payload decodes.
    """
    prep = StoragePreparer()
    over_cap = 0
    for path in good_invoice_paths:
        bgr = cv2.imread(str(path))
        payload, mime = prep.prepare(bgr)
        assert mime == "image/png", f"{path.name}: unexpected mime {mime}"
        decoded = cv2.imdecode(
            np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
        )
        assert decoded is not None, f"{path.name}: failed to decode payload"
        if len(payload) > Encoder.SIZE_CAP_BYTES:
            over_cap += 1
    fit_ratio = (len(good_invoice_paths) - over_cap) / len(good_invoice_paths)
    assert fit_ratio >= 0.90, (
        f"only {fit_ratio:.1%} of {len(good_invoice_paths)} samples fit "
        f"≤ {Encoder.SIZE_CAP_BYTES} bytes (cap-fit calibration regressed)"
    )
