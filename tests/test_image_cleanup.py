"""Contract tests for the image-cleanup primitives.

These cover the four classes that produce the storage-bound image:
``YellowRemover``, ``BackgroundFlattener``, ``BackgroundSnapper``,
``EdgeCleaner``. The contracts they enforce together are the user's
reproduction guarantee: pure-white background, faithful foreground.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from docscanner import (
    BackgroundFlattener,
    BackgroundSnapper,
    EdgeCleaner,
    YellowRemover,
)


# ---------------------------------------------------------------------------
# Synthetic-image helpers
# ---------------------------------------------------------------------------


def _yellow_paper(h: int = 200, w: int = 200) -> np.ndarray:
    """A flat yellow page in BGR (close to invoice paper color)."""
    img = np.full((h, w, 3), (140, 230, 240), dtype=np.uint8)  # BGR
    return img


def _put_black_text(img: np.ndarray, text: str = "INV/2026/05000") -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    return out


def _put_blue_mark(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    cv2.line(out, (40, 150), (160, 150), (200, 50, 30), 6)  # BGR blue
    return out


# ---------------------------------------------------------------------------
# YellowRemover
# ---------------------------------------------------------------------------


class TestYellowRemover:
    def test_yellow_background_becomes_near_white(self) -> None:
        img = _yellow_paper()
        out = YellowRemover().remove(img)
        # Sample a clearly-paper region (no text).
        sample = out[10:50, 10:50]
        assert sample.mean() > 220, (
            f"Yellow paper should look near-white after removal, got mean {sample.mean():.1f}"
        )

    def test_black_text_is_preserved(self) -> None:
        img = _put_black_text(_yellow_paper())
        out = YellowRemover().remove(img)
        # The text region should still contain dark pixels.
        text_region = out[60:90, 10:200]
        assert text_region.min() < 60, (
            "Black text must survive yellow removal — found min "
            f"{text_region.min()} (expected dark pixels)."
        )

    def test_blue_ink_is_not_treated_as_yellow(self) -> None:
        img = _put_blue_mark(_yellow_paper())
        out = YellowRemover().remove(img)
        # The blue line should still be more blue than green/red.
        line_region = out[145:155, 50:150]
        b, g, r = cv2.split(line_region)
        assert b.mean() > r.mean() + 10, (
            "Blue ink must remain blue after yellow removal "
            f"(B={b.mean():.0f}, G={g.mean():.0f}, R={r.mean():.0f})."
        )

    def test_rejects_grayscale_input(self) -> None:
        gray = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError):
            YellowRemover().remove(gray)

    def test_real_invoice_loses_yellow_cast(self, first_good_invoice: Path) -> None:
        img = cv2.imread(str(first_good_invoice))
        assert img is not None, f"Could not read {first_good_invoice}"
        out = YellowRemover().remove(img)
        # Compare yellow-channel-imbalance: in BGR, yellow paper has
        # (B<G,B<R, G≈R). After removal a paper sample should be roughly
        # neutral (B≈G≈R within a small margin).
        h, w = out.shape[:2]
        # Sample a paper-only patch from the top margin.
        patch = out[10 : h // 30, w // 4 : 3 * w // 4]
        b, g, r = cv2.split(patch)
        imbalance_before = float(g.mean() - b.mean())
        # Same patch on the input.
        in_patch = img[10 : h // 30, w // 4 : 3 * w // 4]
        ib, ig, ir = cv2.split(in_patch)
        imbalance_in = float(ig.mean() - ib.mean())
        assert imbalance_before < imbalance_in, (
            "Yellow cast should be reduced — "
            f"before G−B={imbalance_in:.1f}, after G−B={imbalance_before:.1f}"
        )


# ---------------------------------------------------------------------------
# BackgroundFlattener
# ---------------------------------------------------------------------------


class TestBackgroundFlattener:
    def test_flattens_a_lightness_gradient(self) -> None:
        h, w = 300, 300
        # Left half darker than right half (simulated scanner shadow).
        img = np.full((h, w), 200, dtype=np.uint8)
        img[:, : w // 2] = 140
        out = BackgroundFlattener().flatten(img)
        left_mean = out[:, : w // 2].mean()
        right_mean = out[:, w // 2 :].mean()
        gap = abs(left_mean - right_mean)
        # Original gap is 60; after flattening it should narrow markedly.
        assert gap < 25, f"Flattener left a {gap:.1f} grayscale gap; expected <25"

    def test_color_image_returns_color(self) -> None:
        img = np.full((100, 100, 3), (200, 200, 200), dtype=np.uint8)
        out = BackgroundFlattener().flatten(img)
        assert out.ndim == 3 and out.shape[2] == 3


# ---------------------------------------------------------------------------
# BackgroundSnapper — the "clean white background" guarantee.
# ---------------------------------------------------------------------------


class TestBackgroundSnapper:
    def test_background_is_pure_white_after_snap(self) -> None:
        # Synthetic: noisy-grey paper background + black text.
        rng = np.random.default_rng(seed=42)
        bg = rng.integers(220, 240, size=(300, 600), dtype=np.uint8)
        cv2.putText(bg, "INV/2026/05000", (40, 200), cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, 0, 4)
        snapped, fg_mask = BackgroundSnapper().snap(bg)
        # All pixels NOT in the foreground mask must be exactly 255.
        bg_pixels = snapped[fg_mask == 0]
        assert bg_pixels.size > 0
        assert bg_pixels.min() == 255 and bg_pixels.max() == 255, (
            f"Background pixels must be pure 255, got "
            f"min={bg_pixels.min()} max={bg_pixels.max()}"
        )

    def test_foreground_keeps_grayscale_tone(self) -> None:
        """Signatures need their grayscale, not 1-bit reduction."""
        bg = np.full((300, 600), 230, dtype=np.uint8)
        # Synthetic "signature": a stroke with varying intensity.
        for x, intensity in zip(range(50, 550, 5), range(200, 0, -2)):
            cv2.circle(bg, (x, 150), 3, int(intensity), -1)
        snapped, fg_mask = BackgroundSnapper().snap(bg)
        fg_pixels = snapped[fg_mask > 0]
        unique = np.unique(fg_pixels)
        assert len(unique) > 4, (
            f"Foreground must keep grayscale tones (got only "
            f"{len(unique)} unique values: {unique[:8]})"
        )

    def test_real_invoice_background_is_white(self, first_good_invoice: Path) -> None:
        bgr = cv2.imread(str(first_good_invoice))
        cleaned = YellowRemover().remove(bgr)
        flat = BackgroundFlattener().flatten(cleaned)
        snapped, fg_mask = BackgroundSnapper().snap(flat)
        bg_pixels = snapped[fg_mask == 0]
        # On a real noisy scan the snap must still drive every paper pixel
        # to 255 — that is the reproduction contract.
        assert (bg_pixels == 255).all(), (
            f"Real-invoice background not snapped to white: "
            f"unique values present = {np.unique(bg_pixels)[:5]}"
        )


# ---------------------------------------------------------------------------
# EdgeCleaner
# ---------------------------------------------------------------------------


class TestEdgeCleaner:
    def test_removes_isolated_speckle_keeps_strokes(self) -> None:
        # White page with one connected stroke and several isolated dots.
        snapped = np.full((200, 200), 255, dtype=np.uint8)
        # A long horizontal "stroke" — large connected component.
        cv2.line(snapped, (20, 100), (180, 100), 0, 2)
        # Isolated single-pixel speckles.
        for x, y in [(30, 30), (60, 50), (120, 70), (170, 30)]:
            snapped[y, x] = 0
        fg_mask = (snapped < 200).astype(np.uint8) * 255
        out = EdgeCleaner().clean(snapped, fg_mask)
        # Speckles should be gone.
        for x, y in [(30, 30), (60, 50), (120, 70), (170, 30)]:
            assert out[y, x] == 255, f"Speckle at {(x, y)} survived"
        # Stroke should remain at most pixels along its length.
        stroke_pixels = out[99:102, 20:180]
        dark_in_stroke = (stroke_pixels < 200).sum()
        assert dark_in_stroke > 50, (
            f"Connected stroke was eroded — only {dark_in_stroke} dark pixels left"
        )
