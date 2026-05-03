"""Contract tests for OcrEngine on real invoice samples.

v3.5 backend: DocTR detection + recognition (no Tesseract). The engine
takes a BGR image (no separate binarization step), invokes DocTR's
``ocr_predictor`` once per region crop, and matches the configured
regex against the recognized text. Rotation cascade unchanged: try
0° → 90° → 180° → 270° on the original until a hit lands.
"""

import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from docscanner import (
    Invoice,
    OcrEngine,
    OcrMatch,
    OcrResult,
    Region,
    YellowRemover,
)


# ---------------------------------------------------------------------------
# OcrEngine — end-to-end on real samples
# ---------------------------------------------------------------------------


def _expected_invoice_name(path: Path) -> str:
    """Filenames look like INV-2026-05000_id-...; convert to INV/2026/05000."""
    parts = path.name.split("_", 1)[0]
    # parts like 'INV-2026-05000' or 'RINV-2026-12345'
    pieces = parts.split("-")
    if len(pieces) != 3:
        return ""
    prefix, year, num = pieces
    return f"{prefix}/{year}/{num}"


def _to_bgr(gray: np.ndarray) -> np.ndarray:
    """DocTR expects color input. Tests render synthetic pages on grayscale
    canvases for compactness; convert to 3-channel BGR before passing in."""
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class TestOcrEngine:
    def test_extracts_invoice_number_from_real_sample(
        self,
        first_good_invoice: Path,
    ) -> None:
        bgr = cv2.imread(str(first_good_invoice))
        cleaned = YellowRemover().remove(bgr)
        regions = (Region(60, 0, 100, 25), Region(20, 30, 80, 70))
        regex = re.compile(r"R?INV/20\d{2}/\d{4,5}")
        engine = OcrEngine()
        match = engine.extract(cleaned, regions, regex)
        assert match is not None, (
            f"Expected to extract invoice number from {first_good_invoice.name}"
        )
        expected = _expected_invoice_name(first_good_invoice)
        assert match.name == expected, (
            f"Got {match.name!r} expected {expected!r}"
        )

    def test_returns_none_for_unreadable(
        self, unreadable_paths: list[Path]
    ) -> None:
        # A truly unreadable scan should return None for our regex.
        regex = re.compile(r"R?INV/20\d{2}/\d{4,5}")
        engine = OcrEngine()
        misses = 0
        for path in unreadable_paths[:5]:  # 5 is enough — these are designed-fail
            bgr = cv2.imread(str(path))
            if bgr is None:
                continue
            regions = (Region(60, 0, 100, 25), Region(20, 30, 80, 70))
            match = engine.extract(bgr, regions, regex)
            if match is None:
                misses += 1
        # Most unreadable samples should not yield a hit. We don't require
        # 100 % miss because some failed-scans actually contain the right
        # text; just demand at least one is genuinely unreadable.
        assert misses >= 1, "Expected at least one unreadable to truly fail"

    def test_so_in_text_is_not_matched_as_invoice(self) -> None:
        """A page whose body says 'SO/2026/12345' must not be misread."""
        page = np.full((1200, 1600), 255, dtype=np.uint8)
        cv2.putText(page, "Source: SO/2026/12345", (100, 600),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, 0, 5)
        regex = re.compile(r"R?INV/20\d{2}/\d{4,5}")
        engine = OcrEngine()
        regions = (Region(0, 0, 100, 100),)
        match = engine.extract(_to_bgr(page), regions, regex)
        assert match is None, f"SO/… must not match the INV regex (got {match})"

    def test_first_region_wins_when_match_present(self) -> None:
        """If both regions hold an invoice number, the first one is used."""
        page = np.full((2200, 1700), 255, dtype=np.uint8)
        # Top-right header — invoice number that should win.
        cv2.putText(page, "INV/2026/05000", (1100, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, 0, 5)
        # Body — invoice number that must NOT win.
        cv2.putText(page, "INV/2026/99999", (200, 1500),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, 0, 5)
        regex = re.compile(r"R?INV/20\d{2}/\d{4,5}")
        engine = OcrEngine()
        regions = (Region(50, 0, 100, 20), Region(0, 50, 60, 100))
        match = engine.extract(_to_bgr(page), regions, regex)
        assert match is not None
        assert match.name == "INV/2026/05000", (
            f"First region (top-right) should win, got {match.name}"
        )
        assert match.region == regions[0]


# ---------------------------------------------------------------------------
# OcrEngine.extract_with_fallbacks — cascade contract
# ---------------------------------------------------------------------------


def _make_invoice_page(text_in_top_right: str | None,
                       text_in_body: str | None) -> np.ndarray:
    """Synthetic invoice-shaped page (1700×2200 white BGR). Optional text
    in top-right header and/or body. Stroke thickness = 5 — DocTR's
    detector drops the leading "I" from Hershey-Simplex glyphs at the
    default thickness=4 because the verticals collapse on bicubic
    upscale; an extra pixel of stroke keeps it intact.
    """
    page = np.full((2200, 1700), 255, dtype=np.uint8)
    if text_in_top_right:
        cv2.putText(page, text_in_top_right, (1100, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, 0, 5)
    if text_in_body:
        cv2.putText(page, text_in_body, (200, 1500),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, 0, 5)
    return _to_bgr(page)


class TestExtractWithFallbacks:
    PRIMARY = (Region(50, 0, 100, 20),)               # top-right only
    BODY_ONLY = (Region(0, 30, 60, 80),)              # body region
    FULL_PAGE = (Region(0, 0, 100, 100),)             # whole page
    REGEX = re.compile(r"R?INV/20\d{2}/\d{4,5}")

    def test_primary_wins_no_rotation(self) -> None:
        page = _make_invoice_page("INV/2026/05000", None)
        engine = OcrEngine()
        result = engine.extract_with_fallbacks(
            page,
            primary_regions=self.PRIMARY,
            fallback_regions=(self.FULL_PAGE,),
            try_rotation=True,
            regex=self.REGEX,
        )
        assert result is not None
        assert result.match.name == "INV/2026/05000"
        assert result.rotation_degrees == 0

    def test_fallback_region_wins_when_primary_misses(self) -> None:
        # Number is in the body, not the top-right primary region.
        page = _make_invoice_page(None, "INV/2026/05001")
        engine = OcrEngine()
        result = engine.extract_with_fallbacks(
            page,
            primary_regions=self.PRIMARY,           # body excluded
            fallback_regions=(self.FULL_PAGE,),     # full page picks it up
            try_rotation=False,
            regex=self.REGEX,
        )
        assert result is not None
        assert result.match.name == "INV/2026/05001"
        assert result.rotation_degrees == 0

    def test_rotation_180_wins(self) -> None:
        page = _make_invoice_page("INV/2026/05002", None)
        # Rotate 180° so primary OCR fails until the cascade rotates it back.
        page_rot = np.rot90(page, k=2)
        engine = OcrEngine()
        result = engine.extract_with_fallbacks(
            page_rot,
            primary_regions=self.PRIMARY,
            fallback_regions=(),
            try_rotation=True,
            regex=self.REGEX,
        )
        assert result is not None
        assert result.match.name == "INV/2026/05002"
        assert result.rotation_degrees == 180

    def test_all_miss_returns_none(self) -> None:
        page = _make_invoice_page(None, "Source: SO/2026/77777")  # SO never matches INV
        engine = OcrEngine()
        result = engine.extract_with_fallbacks(
            page,
            primary_regions=self.PRIMARY,
            fallback_regions=(self.FULL_PAGE,),
            try_rotation=True,
            regex=self.REGEX,
        )
        assert result is None

    def test_no_fallbacks_no_rotation_matches_extract_behavior(self) -> None:
        """With empty fallbacks and rotation off, cascade is just `extract`."""
        page = _make_invoice_page("INV/2026/05003", None)
        engine = OcrEngine()
        result = engine.extract_with_fallbacks(
            page,
            primary_regions=self.PRIMARY,
            fallback_regions=(),
            try_rotation=False,
            regex=self.REGEX,
        )
        direct = engine.extract(page, self.PRIMARY, self.REGEX)
        assert result is not None and direct is not None
        assert result.match == direct
        assert result.rotation_degrees == 0
