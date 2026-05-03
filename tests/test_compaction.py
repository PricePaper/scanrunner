"""Contract tests for the storage-compaction primitives.

`EdgeCrispener` snaps text-edge anti-aliasing halos to pure white but
leaves signature-body intermediates alone. `ForegroundQuantizer` collapses
foreground intensities to a small number of evenly-spaced bins so PNG
deflate compresses better. Together they cut PNG bytes substantially
without losing signature legibility.

Tests are written first — they should fail until the classes are
implemented in docscanner.py.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from docscanner import (
    BackgroundFlattener,
    BackgroundSnapper,
    EdgeCleaner,
    EdgeCrispener,
    ForegroundQuantizer,
    YellowRemover,
)


# ---------------------------------------------------------------------------
# EdgeCrispener
# ---------------------------------------------------------------------------


def _black_square_with_grey_halo(
    h: int = 200, w: int = 200, halo: int = 130
) -> np.ndarray:
    """White page with a black square wrapped in a 1-pixel grey halo
    immediately adjacent to the black core (the realistic anti-aliased
    text-edge geometry the production cleanup pipeline emits)."""
    page = np.full((h, w), 255, dtype=np.uint8)
    # Solid black core first.
    cv2.rectangle(page, (44, 44), (156, 156), 0, thickness=-1)
    # Then paint a 1-pixel-wide grey ring AROUND its outer edge — abuts
    # the black, no gap. Drawing in this order means the corners stay
    # halo-colored (the intent).
    cv2.rectangle(page, (43, 43), (157, 157), int(halo), thickness=1)
    return page


def _signature_blob(h: int = 200, w: int = 200, body: int = 130) -> np.ndarray:
    """White page with a large continuous mid-tone blob (pseudo-signature)."""
    page = np.full((h, w), 255, dtype=np.uint8)
    cv2.ellipse(
        page, (w // 2, h // 2), (60, 30), 0, 0, 360, int(body), thickness=-1
    )
    return page


class TestEdgeCrispener:
    def test_halo_around_black_collapses_to_white(self) -> None:
        page = _black_square_with_grey_halo(halo=130)
        # Spot-check halo pixels exist in the input.
        assert (page == 130).any(), "fixture should have halo pixels"
        out = EdgeCrispener().crispen(page)
        # The intermediate halo (row 43, cols 43..157) should be gone —
        # pixels there should be either 0 or 255.
        ring = out[43, 44:157]
        unique = set(int(v) for v in np.unique(ring))
        assert unique <= {0, 255}, (
            f"halo not crisped — unique values in ring: {unique}"
        )

    def test_signature_body_preserved(self) -> None:
        page = _signature_blob(body=130)
        out = EdgeCrispener().crispen(page)
        # Center of the signature should still be intermediate (~130).
        center: int = int(out[100, 100])
        assert 100 <= center <= 160, (
            f"signature body got crisped — center pixel = {center}"
        )

    def test_text_with_signature_keeps_signature_drops_halos(self) -> None:
        # White page, black anti-aliased text in upper region, signature
        # blob in lower. lineType=cv2.LINE_AA gives the realistic halo.
        page = np.full((400, 400), 255, dtype=np.uint8)
        cv2.putText(page, "INV/2026/05000", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3,
                    lineType=cv2.LINE_AA)
        cv2.ellipse(page, (200, 300), (80, 30), 0, 0, 360, 130, thickness=-1)
        out = EdgeCrispener().crispen(page)
        # Signature center preserved.
        sig_center: int = int(out[300, 200])
        assert 100 <= sig_center <= 160, (
            f"signature in mixed page got crisped: {sig_center}"
        )
        # Text region: count intermediate halo pixels (40 < v < 230).
        # Should drop substantially compared to input.
        text_band = page[60:90, 30:380]
        text_band_out = out[60:90, 30:380]
        n_halo_in: int = int(((text_band > 40) & (text_band < 230)).sum())
        n_halo_out: int = int(((text_band_out > 40) & (text_band_out < 230)).sum())
        assert n_halo_in > 0, "fixture text should have anti-aliased halo"
        assert n_halo_out < n_halo_in // 2, (
            f"text halo not reduced enough: {n_halo_in} → {n_halo_out}"
        )

    def test_real_invoice_drops_unique_intensity_count(
        self, first_good_invoice: Path
    ) -> None:
        """End-to-end: cleaned real invoice has fewer unique grayscale
        levels in the foreground after crispening."""
        bgr = cv2.imread(str(first_good_invoice))
        cleaned = YellowRemover().remove(bgr)
        flat = BackgroundFlattener().flatten(cleaned)
        snapped, fg_mask = BackgroundSnapper().snap(flat)
        snapped = EdgeCleaner().clean(snapped, fg_mask)
        before_unique: int = int(len(np.unique(snapped[fg_mask > 0])))
        crisped = EdgeCrispener().crispen(snapped)
        # Re-derive foreground mask on the crisped image (snap may have
        # turned halo pixels white, so the mask narrows naturally).
        fg_after = (crisped < 240).astype(np.uint8) * 255
        after_unique: int = int(len(np.unique(crisped[fg_after > 0])) or 1)
        # Crispening should strip enough halo intermediates that the
        # surviving foreground has a clearly tighter distribution.
        assert after_unique < before_unique, (
            f"unique foreground intensities did not shrink: "
            f"{before_unique} → {after_unique}"
        )


# ---------------------------------------------------------------------------
# ForegroundQuantizer
# ---------------------------------------------------------------------------


class TestForegroundQuantizer:
    def test_collapses_foreground_to_n_levels(self) -> None:
        # A continuous-tone foreground gradient on a white background.
        page = np.full((100, 256), 255, dtype=np.uint8)
        # Row 50 is a horizontal gradient from 0..254 across 256 cols.
        page[50, :] = np.arange(256, dtype=np.uint8)
        # 255 stays background. Foreground = pixels < 240 (roughly).
        out = ForegroundQuantizer(levels=16).quantize(page)
        # Background untouched.
        bg_out = out[0:40, :]
        assert (bg_out == 255).all(), "background must stay 255"
        # Foreground reduced to ≤16 distinct values.
        fg_out = out[50, :]
        n_unique: int = int(len(np.unique(fg_out)))
        assert n_unique <= 16, (
            f"foreground not quantized to ≤16 levels (got {n_unique})"
        )

    def test_pure_white_background_unchanged(self) -> None:
        page = np.full((100, 100), 255, dtype=np.uint8)
        out = ForegroundQuantizer(levels=16).quantize(page)
        assert (out == 255).all(), "all-white input must round-trip identical"

    def test_signature_intensity_roughly_preserved(self) -> None:
        # Synthetic signature blob at intensity ~120.
        page = np.full((200, 200), 255, dtype=np.uint8)
        cv2.ellipse(page, (100, 100), (60, 30), 0, 0, 360, 120, thickness=-1)
        out = ForegroundQuantizer(levels=16).quantize(page)
        # Mean intensity over the signature region should remain near
        # 120 within one quantization step (≈ 256/16 = 16).
        sig_pixels = out[(out < 240) & (out > 50)]
        assert sig_pixels.size > 0
        mean_after: float = float(sig_pixels.mean())
        assert abs(mean_after - 120) < 16, (
            f"signature mean intensity drifted too far: {mean_after:.1f}"
        )

    def test_levels_default_is_8(self) -> None:
        # Documentary test: locks the default so encoder size assumptions
        # in StoragePreparer's calibration match what runs in production.
        # 8 levels was the only candidate hitting ≥90 % fit ≤ 300 KB on
        # the 168-page good/ corpus (2026-05-03 sweep).
        assert ForegroundQuantizer.DEFAULT_LEVELS == 8

    def test_real_invoice_keeps_signature_appearance(
        self, first_good_invoice: Path
    ) -> None:
        """Round-tripping a real invoice through quantization preserves
        a signature-shaped region's mean intensity within one bin width."""
        bgr = cv2.imread(str(first_good_invoice))
        cleaned = YellowRemover().remove(bgr)
        flat = BackgroundFlattener().flatten(cleaned)
        snapped, _ = BackgroundSnapper().snap(flat)
        # Pick the darkest 10% of foreground (text is darkest), measure
        # its mean before / after quantization. Mean should drift by
        # less than one quantization bin width (≈256/16 = 16).
        fg_pixels = snapped[snapped < 240]
        if fg_pixels.size == 0:
            pytest.skip("sample has no foreground")
        threshold: int = int(np.percentile(fg_pixels, 10))
        dark_mask = snapped < threshold
        mean_before: float = float(snapped[dark_mask].mean())
        out = ForegroundQuantizer(levels=16).quantize(snapped)
        mean_after: float = float(out[dark_mask].mean())
        assert abs(mean_after - mean_before) < 16, (
            f"darkest 10% drifted by {abs(mean_after - mean_before):.1f}"
        )
