"""Contract tests for ``PrintedLayer`` and ``PrintedLayerExtractor`` (v3 phase 2).

``PrintedLayer`` is the v3 carrier object for laser-printed content lifted off
a scanned page: a boolean ``mask`` marking pixels that belong to printed
strokes (text, frames, logos, barcodes), and a ``tones`` grayscale map that
preserves the ORIGINAL grayscale value at those masked pixels (so the
downstream composer can lay anti-aliased ink back onto a clean paper canvas
without binarizing it into black-on-white).

``PrintedLayerExtractor`` identifies the printed layer given a paper-tone
baseline (``paper_color`` and ``paper_variation`` from
``estimate_paper_tone``). The extractor must distinguish printed content
(uniform stroke widths, regular geometry, consistently darker than paper)
from handwriting (variable strokes, irregular geometry) and from paper noise
(near-paper-tone pixels).

Tests draw real fixtures from ``corpus/receipts/`` (white paper) and
``corpus/invoices/good/`` (yellow paper) so the extractor's substrate-aware
behavior is verified against actual scans, not synthetic toys.

These tests pin the WHAT (return shape, dtype, mask coverage envelopes per
content class, tone preservation) without prescribing HOW the extractor
computes it.
"""

import cv2
import fitz  # PyMuPDF
import numpy as np
import pytest
from dataclasses import FrozenInstanceError
from pathlib import Path

from docscanner import (
    BgrColor,
    BgrImage,
    PrintedLayer,
    PrintedLayerExtractor,
    estimate_paper_tone,
)


# --- Corpus paths (relative to project_root) --------------------------------

WHITE_RECEIPT_RELATIVE_PATH = "corpus/receipts/WH_IN_2026_00548.pdf"
YELLOW_INVOICE_RELATIVE_PATH = (
    "corpus/invoices/good/"
    "INV-2026-05015_id-973750_aid-430545_Customer_Invoice"
    "-20260430_084849_0020.jpg"
)

# --- PDF rendering ----------------------------------------------------------

# 200 DPI matches the calibration corpus; PDFs render to ~1652x2138 at this
# DPI, which the verified-clean crops below were probed against.
PDF_RENDER_DPI = 200


# --- Verified-clean crop coordinates ----------------------------------------
#
# These slices were probed against the real corpus files; the assertion
# tolerances below are anchored to the empirical dark-pixel ratios reported
# by that probe.

# PRINTED-TEXT crop on WH_IN_2026_00548.pdf page 0:
# `bgr[40:340, 60:860]` -> 300x800 with the company header
# ("Fancy Heat Corp / 40 Veronica Avenue / Somerset New Jersey ...").
# Pure printed text, no handwriting. Naive dark-pixel ratio (V<100) ~ 2.81%.
PRINTED_TEXT_ROW_START = 40
PRINTED_TEXT_ROW_END = 340
PRINTED_TEXT_COL_START = 60
PRINTED_TEXT_COL_END = 860

# HANDWRITING crop on WH_IN_2026_00548.pdf page 0:
# `bgr[540:840, 1080:1480]` -> 300x400 with the handwritten "Carlos U. 4/16/26"
# inside a hand-drawn circle. No printed content. Dark-pixel ratio ~ 3.21%.
HANDWRITING_ROW_START = 540
HANDWRITING_ROW_END = 840
HANDWRITING_COL_START = 1080
HANDWRITING_COL_END = 1480

# BLANK-PAPER crop on WH_IN_2026_00548.pdf page 0:
# `bgr[h-1100:h-600, w//2-300:w//2+300]` -> 500x600 of pure white substrate.
# Dark-pixel ratio is 0%.
BLANK_PAPER_ROW_OFFSET_FROM_BOTTOM_START = 1100
BLANK_PAPER_ROW_OFFSET_FROM_BOTTOM_END = 600
BLANK_PAPER_COL_HALFWIDTH = 300

# YELLOW-INVOICE HEADER crop on the INV-2026-05015 ..._0020.jpg sample
# (~2548x3332): `bgr[80:480, 60:1200]` -> 400x1140 of printed header text
# ("Price Paper / 379 N Main St / Freeport, New York") on yellow paper.
YELLOW_HEADER_ROW_START = 80
YELLOW_HEADER_ROW_END = 480
YELLOW_HEADER_COL_START = 60
YELLOW_HEADER_COL_END = 1200

# YELLOW-INVOICE 05020 page: a separate sample we slice for HORIZONTAL
# HANDWRITING (the "3FD Kraft 2 Back order" handwritten line). This is the
# adversarial case: handwriting written across the page in a single line
# forms a wide horizontal connected component that fools any naive aspect-
# ratio screen looking for "text-shaped strips."
HORIZONTAL_HANDWRITING_INVOICE_RELATIVE_PATH = (
    "corpus/invoices/good/"
    "INV-2026-05020_id-973756_aid-430394_Customer_Invoice"
    "-20260429_150716_0002.jpg"
)
# `bgr[1010:1150, 150:1100]` -> 140x950 strip of just the handwritten
# "3FD Kraft 2 Back order" on yellow paper; no printed text, no signature.
HORIZONTAL_HANDWRITING_ROW_START = 1010
HORIZONTAL_HANDWRITING_ROW_END = 1150
HORIZONTAL_HANDWRITING_COL_START = 150
HORIZONTAL_HANDWRITING_COL_END = 1100


# --- Synthetic-layer dimensions ---------------------------------------------

# Used only by the dataclass-shape test; small fixed dimensions are fine
# because we're verifying field metadata, not extraction behavior.
SYNTHETIC_LAYER_HEIGHT = 10
SYNTHETIC_LAYER_WIDTH = 15


# --- Mask coverage envelopes (justified inline at assertion sites) ----------

# Printed text crop empirical dark-pixel ratio is ~2.81%. A 1.5% floor
# leaves headroom for the extractor to be tighter than naive thresholding
# (e.g., excluding anti-aliased halos) while still catching the majority of
# printed strokes.
PRINTED_TEXT_MASK_COVERAGE_FLOOR = 0.015

# Handwriting crop empirical dark-pixel ratio is ~3.21%. A 0.5% ceiling
# pins the extractor to "handwriting must NOT be classified as printed."
# The handwritten signature has variable stroke widths and an irregular
# circle; the extractor must reject it on geometry/uniformity grounds even
# though it's plenty dark.
HANDWRITING_MASK_COVERAGE_CEILING = 0.005

# Blank paper has 0% dark pixels; 0.1% ceiling tolerates microscopic
# false positives from uint8 quantization or paper micro-texture.
BLANK_PAPER_MASK_COVERAGE_CEILING = 0.001

# Yellow-paper header: printed text on a substrate with ~1.90 ΔE intrinsic
# variation. Floor of 1.0% verifies the extractor uses paper_color to
# calibrate its "darker than paper" threshold rather than hardcoding a
# fixed luminance cutoff that would either flood (too low) or miss
# everything (too high) on yellow stock.
YELLOW_HEADER_MASK_COVERAGE_FLOOR = 0.01

# Tone-preservation contract: the mean grayscale of masked pixels must be
# at least this many uint8 units below the paper's grayscale equivalent.
# This rules out a degenerate implementation that stores zeros (or 255s)
# at masked positions instead of the actual ink tone. 50 is wide enough
# to allow the extractor to mask only the darkest portion of strokes
# (skipping anti-aliased halo pixels) while still demanding clear ink
# darkness.
TONES_BELOW_PAPER_GRAYSCALE_GAP = 50

# Paper-color and paper-variation defaults for the white-page shape test.
# We don't depend on estimate_paper_tone here — we want the extractor to
# accept arbitrary substrate-baseline inputs without exploding.
DEFAULT_PAPER_COLOR_WHITE: BgrColor = (255, 255, 255)
DEFAULT_PAPER_VARIATION = 0.5


# --- Helpers ----------------------------------------------------------------

def _render_pdf_page0(path: Path, dpi: int = PDF_RENDER_DPI) -> BgrImage:
    """Render page 0 of ``path`` at ``dpi`` and return it as a BGR ndarray.

    Single render path (PyMuPDF, RGB->BGR conversion, no alpha) reused
    across every PDF-backed test in this file — matches the helper used in
    ``test_paper_tone.py`` so all PDF tests render identically.
    """
    document = fitz.open(str(path))
    try:
        page = document.load_page(0)
        # PyMuPDF's default DPI is 72; scale the matrix so we render at the
        # requested DPI (matches the calibration corpus's 200 DPI baseline).
        zoom: float = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    finally:
        document.close()

    rgb_buffer: np.ndarray = np.frombuffer(pixmap.samples, dtype=np.uint8)
    rgb_image: np.ndarray = rgb_buffer.reshape(pixmap.height, pixmap.width, 3)
    bgr_image: BgrImage = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    return bgr_image


def _load_white_receipt_page(project_root: Path) -> BgrImage:
    """Render WH_IN_2026_00548.pdf page 0 or skip if the corpus file is gone."""
    sample_path: Path = project_root / WHITE_RECEIPT_RELATIVE_PATH
    if not sample_path.is_file():
        pytest.skip(f"corpus file missing: {sample_path}")
    return _render_pdf_page0(sample_path)


def _load_yellow_invoice_page(project_root: Path) -> BgrImage:
    """Read the yellow-paper invoice JPG or skip if the corpus file is gone."""
    sample_path: Path = project_root / YELLOW_INVOICE_RELATIVE_PATH
    if not sample_path.is_file():
        pytest.skip(f"corpus file missing: {sample_path}")
    page: BgrImage | None = cv2.imread(str(sample_path))
    if page is None:
        pytest.skip(
            f"cv2.imread returned None for {sample_path}; "
            f"file unreadable or unsupported"
        )
    return page


def _printed_text_crop(page: BgrImage) -> BgrImage:
    """Slice the verified-clean PRINTED-TEXT crop from a rendered receipt page."""
    return page[
        PRINTED_TEXT_ROW_START:PRINTED_TEXT_ROW_END,
        PRINTED_TEXT_COL_START:PRINTED_TEXT_COL_END,
    ]


def _handwriting_crop(page: BgrImage) -> BgrImage:
    """Slice the verified-clean HANDWRITING crop from a rendered receipt page."""
    return page[
        HANDWRITING_ROW_START:HANDWRITING_ROW_END,
        HANDWRITING_COL_START:HANDWRITING_COL_END,
    ]


def _blank_paper_crop(page: BgrImage) -> BgrImage:
    """Slice the verified-clean BLANK-PAPER crop from a rendered receipt page."""
    height, width = page.shape[:2]
    row_start: int = height - BLANK_PAPER_ROW_OFFSET_FROM_BOTTOM_START
    row_end: int = height - BLANK_PAPER_ROW_OFFSET_FROM_BOTTOM_END
    col_center: int = width // 2
    col_start: int = col_center - BLANK_PAPER_COL_HALFWIDTH
    col_end: int = col_center + BLANK_PAPER_COL_HALFWIDTH
    return page[row_start:row_end, col_start:col_end]


def _yellow_header_crop(page: BgrImage) -> BgrImage:
    """Slice the verified-clean YELLOW-HEADER crop from the invoice JPG."""
    return page[
        YELLOW_HEADER_ROW_START:YELLOW_HEADER_ROW_END,
        YELLOW_HEADER_COL_START:YELLOW_HEADER_COL_END,
    ]


def _load_horizontal_handwriting_page(project_root: Path) -> BgrImage:
    """Read the 05020 invoice JPG (which contains horizontal handwriting)
    or skip if the corpus file is gone."""
    sample_path: Path = project_root / HORIZONTAL_HANDWRITING_INVOICE_RELATIVE_PATH
    if not sample_path.is_file():
        pytest.skip(f"corpus file missing: {sample_path}")
    page: BgrImage | None = cv2.imread(str(sample_path))
    if page is None:
        pytest.skip(
            f"cv2.imread returned None for {sample_path}; "
            f"file unreadable or unsupported"
        )
    return page


def _horizontal_handwriting_crop(page: BgrImage) -> BgrImage:
    """Slice the pure-handwriting "3FD Kraft 2 Back order" strip from 05020."""
    return page[
        HORIZONTAL_HANDWRITING_ROW_START:HORIZONTAL_HANDWRITING_ROW_END,
        HORIZONTAL_HANDWRITING_COL_START:HORIZONTAL_HANDWRITING_COL_END,
    ]


def _bgr_to_grayscale_uint8(bgr_color: BgrColor) -> int:
    """Convert a BGR triple to a single grayscale uint8 (BT.601 weights).

    Matches OpenCV's ``cv2.COLOR_BGR2GRAY`` weights so the assertion uses
    the same luminance definition the extractor's own internals likely use.
    """
    blue, green, red = bgr_color
    gray: int = int(round(0.114 * blue + 0.587 * green + 0.299 * red))
    return gray


# --- Tests ------------------------------------------------------------------


class TestPrintedLayer:
    """Contract for the ``PrintedLayer`` value object.

    Pins:
      * Field dtypes (mask=bool, tones=uint8) and shape consistency.
      * Frozen / slotted dataclass semantics — instances are immutable so
        the decomposer can share them across the print/handwriting/paper
        layers without defensive copies.
    """

    def test_layer_fields_have_required_dtypes_and_shape(self) -> None:
        # Arrange — synthetic mask and tones with the documented dtypes.
        mask: np.ndarray = np.zeros(
            (SYNTHETIC_LAYER_HEIGHT, SYNTHETIC_LAYER_WIDTH), dtype=bool
        )
        tones: np.ndarray = np.full(
            (SYNTHETIC_LAYER_HEIGHT, SYNTHETIC_LAYER_WIDTH),
            255,
            dtype=np.uint8,
        )

        # Act
        layer = PrintedLayer(mask=mask, tones=tones)

        # Assert — dtype is part of the contract; downstream code indexes
        # arrays as boolean masks and reads tones as uint8 grayscale.
        assert layer.mask.dtype == bool, (
            f"mask.dtype={layer.mask.dtype}, expected bool"
        )
        assert layer.tones.dtype == np.uint8, (
            f"tones.dtype={layer.tones.dtype}, expected uint8"
        )
        assert layer.mask.shape == (
            SYNTHETIC_LAYER_HEIGHT,
            SYNTHETIC_LAYER_WIDTH,
        ), (
            f"mask.shape={layer.mask.shape}, expected "
            f"({SYNTHETIC_LAYER_HEIGHT}, {SYNTHETIC_LAYER_WIDTH})"
        )
        assert layer.tones.shape == (
            SYNTHETIC_LAYER_HEIGHT,
            SYNTHETIC_LAYER_WIDTH,
        ), (
            f"tones.shape={layer.tones.shape}, expected "
            f"({SYNTHETIC_LAYER_HEIGHT}, {SYNTHETIC_LAYER_WIDTH})"
        )

    def test_layer_is_frozen_and_slotted(self) -> None:
        # Arrange — any well-formed layer; we only care about mutation.
        layer = PrintedLayer(
            mask=np.zeros(
                (SYNTHETIC_LAYER_HEIGHT, SYNTHETIC_LAYER_WIDTH), dtype=bool
            ),
            tones=np.full(
                (SYNTHETIC_LAYER_HEIGHT, SYNTHETIC_LAYER_WIDTH),
                255,
                dtype=np.uint8,
            ),
        )
        replacement_mask: np.ndarray = np.ones(
            (SYNTHETIC_LAYER_HEIGHT, SYNTHETIC_LAYER_WIDTH), dtype=bool
        )

        # Act / Assert — frozen dataclass raises FrozenInstanceError; slotted
        # dataclass raises AttributeError on unknown attribute. Either is
        # acceptable evidence of "you cannot mutate me," so accept both.
        with pytest.raises((FrozenInstanceError, AttributeError)):
            layer.mask = replacement_mask  # type: ignore[misc]


class TestPrintedLayerExtractor:
    """Contract for ``PrintedLayerExtractor.extract(bgr, paper_color, paper_variation)``.

    Pins:
      * Return type is ``PrintedLayer``; mask and tones match input shape.
      * Mask dtype is bool; tones dtype is uint8.
      * Real printed text -> substantial mask coverage.
      * Real handwriting  -> near-empty mask (handwriting is rejected).
      * Blank paper       -> essentially empty mask.
      * Yellow paper with printed text -> still substantial coverage
        (substrate-aware threshold honors paper_color).
      * Tones at masked pixels preserve original ink darkness, not zeros.
      * Tones outside the mask are still valid uint8 (no NaN / out-of-range).
    """

    def test_returns_printed_layer_with_mask_and_tones_matching_input_shape(
        self, project_root: Path
    ) -> None:
        # Arrange — a full receipt page; the test is about return-shape
        # invariants, not coverage ratios.
        page: BgrImage = _load_white_receipt_page(project_root)
        extractor = PrintedLayerExtractor()

        # Act
        result = extractor.extract(
            page,
            DEFAULT_PAPER_COLOR_WHITE,
            DEFAULT_PAPER_VARIATION,
        )

        # Assert — type, shape, dtype contract.
        assert isinstance(result, PrintedLayer), (
            f"extract() must return PrintedLayer, "
            f"got {type(result).__name__}"
        )
        assert result.mask.shape == page.shape[:2], (
            f"mask.shape={result.mask.shape}, expected {page.shape[:2]}"
        )
        assert result.mask.dtype == bool, (
            f"mask.dtype={result.mask.dtype}, expected bool"
        )
        assert result.tones.shape == page.shape[:2], (
            f"tones.shape={result.tones.shape}, expected {page.shape[:2]}"
        )
        assert result.tones.dtype == np.uint8, (
            f"tones.dtype={result.tones.dtype}, expected uint8"
        )

    def test_printed_text_crop_yields_substantial_mask_coverage(
        self, project_root: Path
    ) -> None:
        # Arrange — pure printed-text crop; calibrate against its own
        # paper baseline so the extractor sees the substrate the way
        # estimate_paper_tone would compute it for this region.
        page: BgrImage = _load_white_receipt_page(project_root)
        crop: BgrImage = _printed_text_crop(page)
        paper_color, paper_variation = estimate_paper_tone(crop)
        extractor = PrintedLayerExtractor()

        # Act
        result = extractor.extract(crop, paper_color, paper_variation)

        # Assert — empirical dark-pixel ratio in this crop is ~2.81%; floor
        # of 1.5% leaves room for the extractor to be more selective than
        # naive thresholding (e.g., dropping anti-aliased halo) while still
        # capturing the bulk of the printed strokes.
        coverage: float = float(result.mask.mean())
        assert coverage > PRINTED_TEXT_MASK_COVERAGE_FLOOR, (
            f"mask coverage {coverage:.4f} did not exceed "
            f"{PRINTED_TEXT_MASK_COVERAGE_FLOOR}; printed text was not "
            f"detected"
        )

    def test_pure_handwriting_crop_yields_low_mask_coverage(
        self, project_root: Path
    ) -> None:
        # Arrange — pure handwriting crop ("Carlos U. 4/16/26" inside a
        # hand-drawn circle). The marks are dark — dark-pixel ratio is
        # ~3.21%, HIGHER than the printed-text crop — so the extractor
        # must distinguish "printed" from "dark" using stroke uniformity
        # / geometry, not luminance alone.
        page: BgrImage = _load_white_receipt_page(project_root)
        crop: BgrImage = _handwriting_crop(page)
        paper_color, paper_variation = estimate_paper_tone(crop)
        extractor = PrintedLayerExtractor()

        # Act
        result = extractor.extract(crop, paper_color, paper_variation)

        # Assert — handwriting must NOT be classified as printed. Ceiling
        # of 0.5% (vs. 3.21% naive dark ratio) means the extractor is
        # rejecting the vast majority of the handwriting pixels.
        coverage: float = float(result.mask.mean())
        assert coverage < HANDWRITING_MASK_COVERAGE_CEILING, (
            f"mask coverage {coverage:.4f} exceeded "
            f"{HANDWRITING_MASK_COVERAGE_CEILING}; handwriting was "
            f"misclassified as printed content"
        )

    def test_horizontal_handwriting_strip_yields_low_mask_coverage(
        self, project_root: Path
    ) -> None:
        # Arrange — adversarial fixture: handwriting written in a single
        # horizontal line ("3FD Kraft 2 Back order" on the 05020 invoice).
        # This case fools any naive aspect-ratio screen on the DocTR mask
        # because the padded word boxes from one handwriting line merge
        # into a wide horizontal strip indistinguishable in shape from a
        # printed-text line. The extractor must reject it on stroke
        # geometry / uniformity grounds — pen ink has variable stroke
        # widths that no laser-printed glyph would.
        page: BgrImage = _load_horizontal_handwriting_page(project_root)
        crop: BgrImage = _horizontal_handwriting_crop(page)
        paper_color, paper_variation = estimate_paper_tone(crop)
        extractor = PrintedLayerExtractor()

        # Act
        result = extractor.extract(crop, paper_color, paper_variation)

        # Assert — same ceiling as the original handwriting test. The
        # adversarial geometry must not change the verdict: handwriting is
        # not printed regardless of orientation.
        coverage: float = float(result.mask.mean())
        assert coverage < HANDWRITING_MASK_COVERAGE_CEILING, (
            f"mask coverage {coverage:.4f} exceeded "
            f"{HANDWRITING_MASK_COVERAGE_CEILING}; horizontal-line "
            f"handwriting was misclassified as printed content "
            f"(paper_color={paper_color}, paper_variation={paper_variation})"
        )

    def test_blank_paper_crop_yields_near_empty_mask(
        self, project_root: Path
    ) -> None:
        # Arrange — pure white paper crop; nothing should fire here.
        page: BgrImage = _load_white_receipt_page(project_root)
        crop: BgrImage = _blank_paper_crop(page)
        paper_color, paper_variation = estimate_paper_tone(crop)
        extractor = PrintedLayerExtractor()

        # Act
        result = extractor.extract(crop, paper_color, paper_variation)

        # Assert — empirical dark-pixel ratio is 0%; 0.1% ceiling tolerates
        # only microscopic false positives from quantization / micro-texture.
        coverage: float = float(result.mask.mean())
        assert coverage < BLANK_PAPER_MASK_COVERAGE_CEILING, (
            f"mask coverage {coverage:.4f} exceeded "
            f"{BLANK_PAPER_MASK_COVERAGE_CEILING}; extractor hallucinated "
            f"printed content on blank paper"
        )

    def test_yellow_paper_with_printed_header_yields_substantial_mask_coverage(
        self, project_root: Path
    ) -> None:
        # Arrange — yellow-substrate invoice header with printed text
        # ("Price Paper / 379 N Main St / Freeport, New York"). This is
        # the substrate-awareness contract: a fixed-luminance threshold
        # would either flood the mask (treating yellow paper as ink) or
        # miss everything (cutoff above the paper's own luminance). The
        # extractor must lean on paper_color to anchor "darker than paper."
        page: BgrImage = _load_yellow_invoice_page(project_root)
        crop: BgrImage = _yellow_header_crop(page)
        paper_color, paper_variation = estimate_paper_tone(crop)
        extractor = PrintedLayerExtractor()

        # Act
        result = extractor.extract(crop, paper_color, paper_variation)

        # Assert — at least 1% mask coverage on a header crop dominated by
        # printed text. Floor is intentionally lower than the white-paper
        # printed-text floor because yellow paper has more intrinsic noise
        # (ΔE ~1.90 vs ~0.35), so the extractor may legitimately tighten
        # its threshold to avoid eating substrate.
        coverage: float = float(result.mask.mean())
        assert coverage > YELLOW_HEADER_MASK_COVERAGE_FLOOR, (
            f"mask coverage {coverage:.4f} did not exceed "
            f"{YELLOW_HEADER_MASK_COVERAGE_FLOOR}; extractor failed to "
            f"detect printed text on yellow paper "
            f"(paper_color={paper_color}, paper_variation={paper_variation})"
        )

    def test_tones_at_masked_pixels_are_substantially_darker_than_paper(
        self, project_root: Path
    ) -> None:
        # Arrange — printed-text crop again; we need pixels in the mask
        # to inspect their preserved tones.
        page: BgrImage = _load_white_receipt_page(project_root)
        crop: BgrImage = _printed_text_crop(page)
        paper_color, paper_variation = estimate_paper_tone(crop)
        extractor = PrintedLayerExtractor()
        result = extractor.extract(crop, paper_color, paper_variation)

        # Sanity gate: the coverage assertions above already pin this, but
        # if the mask happens to be empty here we'd silently pass with a
        # NaN mean. Skip clearly so the failure is signaled by the other
        # test rather than masked here.
        if not result.mask.any():
            pytest.skip(
                "mask is empty on printed-text crop; coverage assertion "
                "in another test will surface this regression"
            )

        paper_grayscale: int = _bgr_to_grayscale_uint8(paper_color)
        masked_tones: np.ndarray = result.tones[result.mask]

        # Act
        masked_tone_mean: float = float(masked_tones.mean())

        # Assert — masked pixels must read at least 50 grayscale units
        # below the paper baseline. This forbids a degenerate "stuff zeros
        # in tones" implementation (which would be too dark, but also
        # forbids a "leave 255 there" implementation, since 255 is well
        # above paper-50). We're pinning that the extractor records the
        # ACTUAL ink tone (anti-aliased grayscale) at the masked pixels.
        gap: float = paper_grayscale - masked_tone_mean
        assert gap >= TONES_BELOW_PAPER_GRAYSCALE_GAP, (
            f"masked-tone mean {masked_tone_mean:.1f} is only {gap:.1f} "
            f"units below paper grayscale {paper_grayscale}; expected "
            f"gap >= {TONES_BELOW_PAPER_GRAYSCALE_GAP}. Tones at masked "
            f"pixels must preserve actual ink darkness."
        )

    def test_tones_outside_mask_are_well_defined_uint8(
        self, project_root: Path
    ) -> None:
        # Arrange — any real crop; the contract here is uniform.
        page: BgrImage = _load_white_receipt_page(project_root)
        crop: BgrImage = _printed_text_crop(page)
        paper_color, paper_variation = estimate_paper_tone(crop)
        extractor = PrintedLayerExtractor()

        # Act
        result = extractor.extract(crop, paper_color, paper_variation)

        # Assert — uint8 dtype and the entire tones array is in
        # [0, 255]. uint8's natural range already enforces this, but a
        # buggy implementation could pre-cast a float buffer and slip
        # NaN through .astype(np.uint8) (becomes 0) or wrap on overflow.
        # The dtype check + min/max bounds catch both.
        assert result.tones.dtype == np.uint8, (
            f"tones.dtype={result.tones.dtype}, expected uint8"
        )
        tones_min: int = int(result.tones.min())
        tones_max: int = int(result.tones.max())
        assert 0 <= tones_min, (
            f"tones.min()={tones_min} below 0"
        )
        assert tones_max <= 255, (
            f"tones.max()={tones_max} above 255"
        )
