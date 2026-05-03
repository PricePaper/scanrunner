"""Contract tests for InkRegionDetector.

Wraps DocTR's pretrained text-region detector. Returns a boolean mask
sized to the input where True covers detected text/handwriting regions
(plus padding to capture stroke edges) and False elsewhere.

Used downstream as a "permission slip" mask: pixels outside detected
ink regions get force-snapped to white, killing the marbled paper-
texture speckle that the snapper + rescuer would otherwise produce
on margin paper.

The DocTR model load is slow (~3-5 s first call). Detector class
caches the predictor at module level after first instantiation, so
across all tests the load happens once.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from docscanner import InkRegionDetector


@pytest.fixture(scope="session")
def detector() -> InkRegionDetector:
    """Session-scoped: model loads once for the whole test run."""
    return InkRegionDetector()


class TestInkRegionDetector:
    def test_mask_shape_matches_input(self, detector: InkRegionDetector) -> None:
        page = np.full((400, 600, 3), 255, dtype=np.uint8)
        cv2.putText(page, "INV/2026/05000", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3,
                    lineType=cv2.LINE_AA)
        mask = detector.detect(page)
        assert mask.shape == (400, 600)
        assert mask.dtype == bool

    def test_mask_covers_text_region(self, detector: InkRegionDetector) -> None:
        page = np.full((400, 600, 3), 255, dtype=np.uint8)
        # Text is roughly at y=170..210, x=50..480 after putText.
        cv2.putText(page, "INV/2026/05000", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3,
                    lineType=cv2.LINE_AA)
        mask = detector.detect(page)
        # The text region should be substantially covered by the mask.
        text_band = mask[170:215, 50:480]
        coverage = float(text_band.mean())
        assert coverage > 0.5, (
            f"text region only {coverage:.0%} covered — DocTR didn't see it"
        )

    def test_mask_mostly_false_in_empty_margin(
        self, detector: InkRegionDetector
    ) -> None:
        page = np.full((400, 600, 3), 255, dtype=np.uint8)
        cv2.putText(page, "INV/2026/05000", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3,
                    lineType=cv2.LINE_AA)
        mask = detector.detect(page)
        # Top margin (y=0..50) is empty paper.
        margin = mask[:50, :]
        assert margin.mean() < 0.05, (
            f"empty margin has {margin.mean():.0%} mask coverage — "
            "DocTR is hallucinating text where there isn't any"
        )

    def test_blank_page_yields_mostly_empty_mask(
        self, detector: InkRegionDetector
    ) -> None:
        page = np.full((400, 600, 3), 255, dtype=np.uint8)
        mask = detector.detect(page)
        assert mask.mean() < 0.05, (
            f"blank page has {mask.mean():.0%} mask coverage — false positives"
        )

    def test_real_invoice_signature_region_is_covered(
        self, detector: InkRegionDetector, project_root: Path
    ) -> None:
        """Real-world contract: a sample with a clear handwritten
        signature gets a mask that covers the signature region."""
        sample = (
            project_root / "corpus" / "invoices" / "good"
            / "INV-2026-05015_id-973750_aid-430545_Customer_Invoice-20260430_084849_0020.jpg"
        )
        if not sample.exists():
            pytest.skip(f"sample not present: {sample.name}")
        bgr = cv2.imread(str(sample))
        mask = detector.detect(bgr)
        # The signature on this page is roughly in the lower-left quadrant.
        # Demand non-trivial coverage there.
        h, w = bgr.shape[:2]
        sig_quadrant = mask[h // 2 :, : w // 2]
        coverage = float(sig_quadrant.mean())
        assert coverage > 0.05, (
            f"signature quadrant only {coverage:.1%} covered — "
            "DocTR missed the signature on this real sample"
        )
