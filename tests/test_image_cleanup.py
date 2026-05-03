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
    FaintInkRescuer,
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

    def test_anti_aliased_pen_stroke_on_yellow_paper_survives(self) -> None:
        """A dark pen stroke on yellow paper has an anti-aliased
        boundary where each edge pixel is a partial mix of ink and
        paper. Those mix pixels (Hue in the yellow band, Sat moderate,
        Val ~100-150) used to score as "yellow paper" and got snapped
        to white, eating the stroke from the inside out. After the
        VAL_MIN / SAT_MIN tightening they survive."""
        # Yellow paper background.
        paper = np.full((200, 400, 3), (50, 200, 230), dtype=np.uint8)  # BGR yellow
        # Anti-aliased dark pen stroke. Drawing with cv2.LINE_AA gives
        # the realistic edge mix.
        cv2.line(paper, (50, 100), (350, 100), (10, 10, 10), thickness=4,
                 lineType=cv2.LINE_AA)
        out = YellowRemover().remove(paper)
        # In the original, the stroke band has many pixels with V≈100-150
        # (the anti-aliased mix). After yellow removal those pixels
        # MUST still be visibly darker than the background — meaning
        # the stroke retained its body, not just its hard core.
        # We check a row through the stroke and assert plenty of dark
        # / mid-tone pixels survive.
        out_gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        stroke_band = out_gray[97:103, 100:300]
        # "Visibly stroke" = pixel below 200 (background after removal
        # is 255 or near-255).
        non_white = int((stroke_band < 200).sum())
        # The stroke is 4 px tall × 200 px wide = ~800 stroke-region
        # pixels in the band. Demand at least half survive as
        # "darker than background" rather than getting whitened.
        assert non_white >= 400, (
            f"YellowRemover ate too much of the pen stroke: only "
            f"{non_white} pixels in the stroke band remained dark"
        )

    def test_real_invoice_pen_signature_survives_yellow_removal(
        self, project_root: Path
    ) -> None:
        """Regression: stage-by-stage probe of inv/good/INV-2026-05020/0002
        showed pen ink (handwritten note + signature on yellow paper)
        was being mostly destroyed by YellowRemover before any other
        stage ran.

        The damage zone is exactly [50, 200) — anti-aliased pen-stroke
        edges where partial-coverage mixes of dark ink and yellow paper
        score as "yellow paper" by HSV (hue in band, sat moderate, val
        ~80-150) and get snapped to pure white. Pre-fix: ~74 % of
        these pixels were destroyed (230 495 → 59 056). Post-fix
        threshold tightening (VAL_MIN, SAT_MIN) preserves them.

        Contract: at least 60 % of the pre-removal pixels in [50, 200)
        survive yellow removal.
        """
        sample = (
            project_root / "inv" / "good"
            / "INV-2026-05020_id-973756_aid-430394_Customer_Invoice-20260429_150716_0002.jpg"
        )
        if not sample.exists():
            pytest.skip(f"sample not present: {sample.name}")
        bgr = cv2.imread(str(sample))
        gray_before = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        midtone_before = int(((gray_before >= 50) & (gray_before < 200)).sum())
        out = YellowRemover().remove(bgr)
        gray_after = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        midtone_after = int(((gray_after >= 50) & (gray_after < 200)).sum())
        retention = midtone_after / max(midtone_before, 1)
        assert retention >= 0.6, (
            f"YellowRemover ate the anti-aliased ink edges: "
            f"midtone retention = {retention:.0%} "
            f"({midtone_before} → {midtone_after})"
        )

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

    def test_pen_ink_against_white_paper_survives(self) -> None:
        """Pen ink (intensity ~200) on near-white paper should be
        classified as foreground. The earlier dance around faint-ink
        survival was a YellowRemover bug; the snap itself just needs
        to preserve clearly-darker-than-paper strokes."""
        page = np.full((400, 400), 252, dtype=np.uint8)
        cv2.line(page, (50, 200), (350, 200), 200, thickness=3)
        snapped, fg_mask = BackgroundSnapper().snap(page)
        line_band = snapped[198:203, 50:350]
        non_white = int((line_band < 250).sum())
        total = int(line_band.size)
        retention = non_white / total
        assert retention >= 0.5, (
            f"pen ink survived only {retention:.0%} of the line region"
        )


# ---------------------------------------------------------------------------
# EdgeCleaner
# ---------------------------------------------------------------------------


class TestFaintInkRescuer:
    """Second-pass rescue: faint pixels adjacent to detected dark
    strokes get reclassified as foreground; isolated faint regions with
    no nearby anchor are left as background."""

    def test_faint_extension_of_dark_stroke_is_rescued(self) -> None:
        # Dark core (intensity 30) plus very faint trailing extension
        # (intensity 235) immediately adjacent — like a signature stroke
        # whose pen lifted to near-paper-tone. With paper near 250 and
        # primary snap C=20 catching <230, the 235 extension is invisible
        # to pass 1; rescue (C=8 inside neighborhood) catches < 242 and
        # picks it up.
        page = np.full((300, 600), 250, dtype=np.uint8)
        cv2.line(page, (50, 150), (250, 150), 30, thickness=4)   # dark core
        cv2.line(page, (250, 150), (550, 150), 235, thickness=4)  # faint extension
        snapped, fg_mask = BackgroundSnapper().snap(page)
        light_in_mask_before = int(((snapped < 245) & (snapped > 225)).sum())
        rescued, rescued_mask = FaintInkRescuer().rescue(snapped, fg_mask, page)
        light_in_mask_after = int(((rescued < 245) & (rescued > 225)).sum())
        assert light_in_mask_after > light_in_mask_before * 2, (
            f"rescue did not strengthen the faint extension: "
            f"{light_in_mask_before} → {light_in_mask_after}"
        )

    def test_isolated_faint_region_is_NOT_rescued(self) -> None:
        # A faint blob (intensity 235) alone in the middle of the page
        # — no dark anchor nearby. Pass 1 misses it (235 > 230 threshold)
        # and pass 2 needs an existing-foreground neighborhood to fire,
        # which doesn't exist. Should stay snapped to white.
        page = np.full((400, 400), 250, dtype=np.uint8)
        cv2.ellipse(page, (200, 200), (40, 20), 0, 0, 360, 235, thickness=-1)
        snapped, fg_mask = BackgroundSnapper().snap(page)
        rescued, rescued_mask = FaintInkRescuer().rescue(snapped, fg_mask, page)
        # Almost no foreground after rescue (only stragglers from
        # adaptive-threshold edge artifacts, if any).
        rescued_fg_count = int((rescued < 240).sum())
        assert rescued_fg_count < 200, (
            f"isolated faint blob got falsely rescued ({rescued_fg_count} "
            "foreground pixels — expected < 200)"
        )

    def test_iterative_rescue_extends_a_realistic_faint_chain(self) -> None:
        """A short faint extension (~50 px) of a dark stroke should be
        rescued via iteration. Reach is intentionally bounded
        (MAX_ITERATIONS × NEIGHBORHOOD_RADIUS = 60 px) — enough for
        signature stroke halos but not enough for the rescue to walk
        across paper into adjacent shadows and snowball them into
        blotches."""
        page = np.full((200, 200), 250, dtype=np.uint8)
        cv2.line(page, (30, 100), (60, 100), 30, thickness=4)         # dark anchor
        cv2.line(page, (60, 100), (110, 100), 235, thickness=4)        # 50 px faint extension
        snapped, fg_mask = BackgroundSnapper().snap(page)
        rescued, rescued_mask = FaintInkRescuer().rescue(snapped, fg_mask, page)
        # Sample chain near the FAR end (column ~95); should be rescued.
        far_end = rescued[98:103, 90:108]
        far_end_fg = int((far_end < 245).sum())
        assert far_end_fg > 20, (
            f"iteration did not extend rescue along the chain: only "
            f"{far_end_fg} pixels rescued near the far end"
        )

    def test_isolated_faint_chain_NOT_bridged(self) -> None:
        """An isolated faint chain with no dark anchor at all should
        not be rescued, even with iteration. We only grow from real
        detected ink — never bridge across pure paper."""
        page = np.full((200, 400), 250, dtype=np.uint8)
        # A faint chain only — no dark anchor at the start.
        cv2.line(page, (50, 100), (330, 100), 235, thickness=4)
        snapped, fg_mask = BackgroundSnapper().snap(page)
        rescued, _ = FaintInkRescuer().rescue(snapped, fg_mask, page)
        # Should remain almost entirely background.
        rescued_fg_count = int((rescued < 240).sum())
        assert rescued_fg_count < 500, (
            f"isolated faint chain falsely bridged ({rescued_fg_count} "
            "pixels rescued — expected near-zero with no dark anchor)"
        )

    def test_speckle_in_textured_paper_does_NOT_snowball(self) -> None:
        """Realistic regression: a speckle in a slightly-darker paper
        region with internal noise/texture must not seed iterative
        rescue. Without a seed-area filter, the rescuer's permissive
        threshold catches paper texture pixels in the speckle's
        neighborhood, then iterates outward into a large blob.

        Real ink is a large connected component (≥ 20 px); paper
        shadow speckle is sub-10-px noise. Filter seeds by area."""
        rng = np.random.default_rng(seed=42)
        # Noisy paper-shadow region: mean ~232, std ~6 (realistic for
        # scanner shadow on yellow paper that didn't fully clear
        # YellowRemover).
        page = rng.integers(220, 245, size=(400, 400), dtype=np.uint8)
        # A few isolated 1-2 px speckle dots — what BackgroundSnapper
        # might pick up as "foreground" in such a region.
        for x, y in [(50, 50), (120, 80), (200, 200), (300, 150), (350, 300)]:
            page[y, x] = 80
        snapped, fg_mask = BackgroundSnapper().snap(page)
        rescued, _ = FaintInkRescuer().rescue(snapped, fg_mask, page)
        rescued_fg = (rescued < 240).astype(np.uint8) * 255
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(rescued_fg)
        biggest = (
            max(int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_labels))
            if n_labels > 1 else 0
        )
        assert biggest < 100, (
            f"speckle blob snowballed to {biggest} pixels — rescuer is "
            "amplifying paper-shadow noise into blotches"
        )

    def test_real_invoice_grows_light_ink_count(
        self, first_good_invoice: Path
    ) -> None:
        bgr = cv2.imread(str(first_good_invoice))
        cleaned = YellowRemover().remove(bgr)
        flat = BackgroundFlattener().flatten(cleaned)
        snapped, fg_mask = BackgroundSnapper().snap(flat)
        before_light = int(((snapped < 240) & (snapped > 200)).sum())
        rescued, _ = FaintInkRescuer().rescue(snapped, fg_mask, flat)
        after_light = int(((rescued < 240) & (rescued > 200)).sum())
        # Real invoices have anti-aliased text edges; rescue should
        # capture additional faint pixels adjacent to the dark cores.
        assert after_light > before_light, (
            f"rescue had no effect on real sample: "
            f"{before_light} → {after_light}"
        )


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
