"""Contract tests for ``HandwrittenLayer`` and ``HandwrittenLayerExtractor``
(v3 phase 3).

``HandwrittenLayer`` is the v3 carrier object for handwritten ink (pen,
pencil, marker, signature ink) lifted off a scanned page. It mirrors
``PrintedLayer`` in shape: a boolean ``mask`` marking pixels that belong
to handwritten strokes (signatures, notes, check marks, scribbled
quantities/codes), and a ``tones`` grayscale map that preserves the
ORIGINAL grayscale value at those masked pixels. Pen ink has rich tonal
variation — pressure highs/lows, fade at stroke endings, varying
opacity along a single curve — and the composer downstream must render
it with that character intact rather than binarizing it into
black-on-white.

``HandwrittenLayerExtractor`` identifies the handwritten layer given:
  1. A paper-tone baseline (``paper_color`` and ``paper_variation`` from
     ``estimate_paper_tone``) so the "darker than paper" decision is
     substrate-aware (works on white receipts AND yellow invoices).
  2. A ``printed_mask`` input naming the pixels the printed-layer
     extractor already claimed. Handwriting only looks at pixels printed
     didn't take. This is the precedence rule that prevents the same
     pixel from being claimed by two layers — whatever printed already
     has stays in printed; whatever's left and looks like ink becomes
     handwriting.

Tests draw real fixtures from ``corpus/receipts/`` (white paper, cursive
signature) and ``corpus/invoices/good/`` (yellow paper, horizontal
handwritten quantity/code line) so the extractor's substrate- and
precedence-aware behavior is verified against actual scans.

These tests pin the WHAT (return shape, dtype, mask coverage envelopes
per content class, precedence-with-printed-mask, tone preservation)
without prescribing HOW the extractor computes it.
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
    HandwrittenLayer,
    HandwrittenLayerExtractor,
    PrintedLayerExtractor,
    estimate_paper_tone,
)


# --- Corpus paths (relative to project_root) --------------------------------

WHITE_RECEIPT_RELATIVE_PATH = "corpus/receipts/WH_IN_2026_00548.pdf"

# 05020 invoice page contains the "3FD Kraft 2 Back order" handwritten
# strip on yellow paper. Same file used by test_printed_layer.py for the
# adversarial horizontal-handwriting case.
HORIZONTAL_HANDWRITING_INVOICE_RELATIVE_PATH = (
    "corpus/invoices/good/"
    "INV-2026-05020_id-973756_aid-430394_Customer_Invoice"
    "-20260429_150716_0002.jpg"
)


# --- PDF rendering ----------------------------------------------------------

# 200 DPI matches the calibration corpus; PDFs render to ~1652x2138 at this
# DPI, which the verified-clean crops below were probed against.
PDF_RENDER_DPI = 200


# --- Verified-clean crop coordinates (mirrored from test_printed_layer.py) --
#
# These slices were probed against the real corpus files; the assertion
# tolerances below are anchored to the empirical dark-pixel ratios
# reported by that probe. Names match test_printed_layer.py exactly so
# the two sibling test files describe the SAME crops, simplifying
# cross-file reasoning when the decomposer wires both extractors.

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

# HORIZONTAL_HANDWRITING crop on the 05020 invoice JPG (~2548x3324):
# `bgr[1010:1150, 150:1100]` -> 140x950 strip of just the handwritten
# "3FD Kraft 2 Back order" on yellow paper; no printed text, no signature.
HORIZONTAL_HANDWRITING_ROW_START = 1010
HORIZONTAL_HANDWRITING_ROW_END = 1150
HORIZONTAL_HANDWRITING_COL_START = 150
HORIZONTAL_HANDWRITING_COL_END = 1100


# --- Precedence-test sub-region for the CARLOS handwriting crop -------------
#
# The "printed-mask covers handwriting" precedence test (test 10) needs a
# rectangular region inside the 300x400 CARLOS crop that the printed_mask
# will pre-claim. Rows 100..200 / cols 100..300 covers a meaningful slice
# of the signature glyphs so the test only passes if the extractor really
# honors precedence (vs. just happening to mask outside that rect).
PRINTED_PRECEDENCE_RECT_ROW_START = 100
PRINTED_PRECEDENCE_RECT_ROW_END = 200
PRINTED_PRECEDENCE_RECT_COL_START = 100
PRINTED_PRECEDENCE_RECT_COL_END = 300


# --- Synthetic-layer dimensions ---------------------------------------------

# Used only by the dataclass-shape test; small fixed dimensions are fine
# because we're verifying field metadata, not extraction behavior.
SYNTHETIC_LAYER_HEIGHT = 10
SYNTHETIC_LAYER_WIDTH = 15


# --- Mask coverage envelopes (justified inline at assertion sites) ----------

# Pure-handwriting crops (CARLOS signature + horizontal "3FD Kraft 2 Back
# order") have empirical dark-pixel ratios ~3.21% / similar order. A 0.5%
# floor demands the extractor catch at least a meaningful fraction of the
# handwriting ink while leaving headroom to be tighter than naive
# thresholding (e.g., excluding anti-aliased halos around stroke edges,
# rejecting paper micro-noise that crossed a fixed luminance cutoff).
HANDWRITING_MASK_COVERAGE_FLOOR = 0.005

# When the printed_mask already covers the printed text, the handwriting
# extractor must claim essentially nothing on a printed-only crop —
# precedence rule. 0.5% ceiling tolerates microscopic mis-attributions
# at stroke edges where the printed_mask might be 1-2 pixels short of the
# actual printed glyph.
PRINTED_TEXT_WITH_PRECEDENCE_MASK_COVERAGE_CEILING = 0.005

# Blank paper has 0% dark pixels; 0.1% ceiling tolerates microscopic
# false positives from uint8 quantization or paper micro-texture. The
# extractor must NOT hallucinate handwriting on bare substrate.
BLANK_PAPER_MASK_COVERAGE_CEILING = 0.001

# Tone-preservation contract: the mean grayscale of masked pixels must be
# at least this many uint8 units below the paper's grayscale equivalent.
# This rules out a degenerate implementation that stores 255s or paper-
# tone values at masked positions instead of the actual pen-ink tone. 50
# is wide enough to allow the extractor to mask only the darker portion
# of strokes (skipping anti-aliased halo pixels) while still demanding
# clear ink darkness.
TONES_BELOW_PAPER_GRAYSCALE_GAP = 50


# --- Default substrate baseline for the shape test --------------------------

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
    ``test_paper_tone.py`` and ``test_printed_layer.py`` so all PDF tests
    render identically.
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


def _load_horizontal_handwriting_page(project_root: Path) -> BgrImage:
    """Read the 05020 invoice JPG (which contains the horizontal handwritten
    "3FD Kraft 2 Back order" strip) or skip if the corpus file is gone."""
    sample_path: Path = (
        project_root / HORIZONTAL_HANDWRITING_INVOICE_RELATIVE_PATH
    )
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
    """Slice the verified-clean HANDWRITING (CARLOS signature) crop."""
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


def _horizontal_handwriting_crop(page: BgrImage) -> BgrImage:
    """Slice the pure-handwriting "3FD Kraft 2 Back order" strip from 05020."""
    return page[
        HORIZONTAL_HANDWRITING_ROW_START:HORIZONTAL_HANDWRITING_ROW_END,
        HORIZONTAL_HANDWRITING_COL_START:HORIZONTAL_HANDWRITING_COL_END,
    ]


def _empty_printed_mask_for(image: BgrImage) -> np.ndarray:
    """Build an all-False printed_mask matching ``image`` H/W. Used when a
    test wants to verify standalone handwriting-extractor behavior with
    no precedence claims yet recorded."""
    return np.zeros(image.shape[:2], dtype=bool)


def _bgr_to_grayscale_uint8(bgr_color: BgrColor) -> int:
    """Convert a BGR triple to a single grayscale uint8 (BT.601 weights).

    Matches OpenCV's ``cv2.COLOR_BGR2GRAY`` weights so the assertion uses
    the same luminance definition the extractor's own internals likely use.
    """
    blue, green, red = bgr_color
    gray: int = int(round(0.114 * blue + 0.587 * green + 0.299 * red))
    return gray


# --- Tests ------------------------------------------------------------------


class TestHandwrittenLayer:
    """Contract for the ``HandwrittenLayer`` value object.

    Pins:
      * Field dtypes (mask=bool, tones=uint8) and shape consistency.
      * Frozen / slotted dataclass semantics — instances are immutable so
        the decomposer can share them across the print/handwriting/paper
        layers without defensive copies.

    Mirrors ``TestPrintedLayer`` in test_printed_layer.py so the two
    carrier objects have identical structural guarantees.
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
        layer = HandwrittenLayer(mask=mask, tones=tones)

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
        layer = HandwrittenLayer(
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


class TestHandwrittenLayerExtractor:
    """Contract for ``HandwrittenLayerExtractor.extract(bgr, paper_color,
    paper_variation, printed_mask)``.

    Pins:
      * Return type is ``HandwrittenLayer``; mask and tones match input
        shape; mask dtype is bool; tones dtype is uint8.
      * Real handwriting (cursive signature on white, horizontal strip
        on yellow) -> substantial mask coverage.
      * Printed text whose pixels are already claimed by ``printed_mask``
        -> near-empty handwriting mask (precedence rule: don't re-claim
        what printed already has).
      * Blank paper -> essentially empty mask.
      * Tones at masked pixels preserve original ink darkness, not zeros
        or paper-tone fill.
      * Tones outside the mask are still valid uint8 (no NaN /
        out-of-range).
      * ``printed_mask`` precedence is enforced even when underlying
        pixels look like ink — the two layers' masks must never overlap.
    """

    def test_returns_handwritten_layer_with_mask_and_tones_matching_input_shape(
        self, project_root: Path
    ) -> None:
        # Arrange — a full receipt page; the test is about return-shape
        # invariants, not coverage ratios.
        page: BgrImage = _load_white_receipt_page(project_root)
        printed_mask: np.ndarray = _empty_printed_mask_for(page)
        extractor = HandwrittenLayerExtractor()

        # Act
        result = extractor.extract(
            page,
            DEFAULT_PAPER_COLOR_WHITE,
            DEFAULT_PAPER_VARIATION,
            printed_mask,
        )

        # Assert — type, shape, dtype contract.
        assert isinstance(result, HandwrittenLayer), (
            f"extract() must return HandwrittenLayer, "
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

    def test_pure_handwriting_crop_yields_substantial_mask_coverage(
        self, project_root: Path
    ) -> None:
        # Arrange — pure handwriting crop ("Carlos U. 4/16/26" inside a
        # hand-drawn circle); empty printed_mask so the extractor sees no
        # prior claims and is judged purely on whether it identifies the
        # ink. Substrate baseline comes from estimate_paper_tone on the
        # crop itself so the "darker than paper" decision is calibrated.
        page: BgrImage = _load_white_receipt_page(project_root)
        crop: BgrImage = _handwriting_crop(page)
        paper_color, paper_variation = estimate_paper_tone(crop)
        printed_mask: np.ndarray = _empty_printed_mask_for(crop)
        extractor = HandwrittenLayerExtractor()

        # Act
        result = extractor.extract(
            crop, paper_color, paper_variation, printed_mask
        )

        # Assert — empirical dark-pixel ratio in this crop is ~3.21%; floor
        # of 0.5% leaves ample room for the extractor to tighten on stroke
        # cores (skipping anti-aliased halos) while still capturing a
        # meaningful slice of the handwriting.
        coverage: float = float(result.mask.mean())
        assert coverage > HANDWRITING_MASK_COVERAGE_FLOOR, (
            f"mask coverage {coverage:.4f} did not exceed "
            f"{HANDWRITING_MASK_COVERAGE_FLOOR}; cursive handwriting was "
            f"not detected (paper_color={paper_color}, "
            f"paper_variation={paper_variation})"
        )

    def test_horizontal_handwriting_strip_yields_substantial_mask_coverage(
        self, project_root: Path
    ) -> None:
        # Arrange — yellow-paper "3FD Kraft 2 Back order" strip; empty
        # printed_mask. This is the load-bearing test for "the leak from
        # PrintedLayer must land in HandwrittenLayer instead." If
        # PrintedLayerExtractor mistakenly absorbs "3FD" into its mask,
        # the decomposer downstream should NOT pass that as printed_mask
        # here — but for THIS unit test, we test the standalone behavior
        # with empty printed_mask so the handwriting extractor is judged
        # purely on whether it catches handwriting on yellow stock.
        page: BgrImage = _load_horizontal_handwriting_page(project_root)
        crop: BgrImage = _horizontal_handwriting_crop(page)
        paper_color, paper_variation = estimate_paper_tone(crop)
        printed_mask: np.ndarray = _empty_printed_mask_for(crop)
        extractor = HandwrittenLayerExtractor()

        # Act
        result = extractor.extract(
            crop, paper_color, paper_variation, printed_mask
        )

        # Assert — same floor as the white-paper handwriting test. The
        # substrate change must not change the verdict: handwriting is
        # handwriting regardless of paper color, as long as paper_color
        # is supplied so the extractor can anchor its threshold.
        coverage: float = float(result.mask.mean())
        assert coverage > HANDWRITING_MASK_COVERAGE_FLOOR, (
            f"mask coverage {coverage:.4f} did not exceed "
            f"{HANDWRITING_MASK_COVERAGE_FLOOR}; horizontal-strip "
            f"handwriting on yellow paper was not detected "
            f"(paper_color={paper_color}, "
            f"paper_variation={paper_variation})"
        )

    def test_pure_printed_text_with_matching_printed_mask_yields_near_empty(
        self, project_root: Path
    ) -> None:
        # Arrange — white-paper printed-text crop ("Fancy Heat Corp"
        # header). Build the printed_mask by running the real
        # PrintedLayerExtractor against the same crop — this is exactly
        # how the decomposer will wire it. The handwriting extractor
        # should then claim essentially nothing because the printed
        # extractor has already taken every printed pixel.
        page: BgrImage = _load_white_receipt_page(project_root)
        crop: BgrImage = _printed_text_crop(page)
        paper_color, paper_variation = estimate_paper_tone(crop)
        printed_layer = PrintedLayerExtractor().extract(
            crop, paper_color, paper_variation
        )
        extractor = HandwrittenLayerExtractor()

        # Act
        result = extractor.extract(
            crop, paper_color, paper_variation, printed_layer.mask
        )

        # Assert — printed-text precedence: ceiling 0.5% tolerates only
        # microscopic mis-attributions at stroke edges where the
        # printed_mask might be 1-2 px short of the actual glyph, plus
        # paper micro-noise. A hit anywhere near the printed-text
        # coverage floor (~1.5% in test_printed_layer.py) would mean the
        # handwriting extractor is double-claiming ink and the
        # composer would draw the glyph twice.
        coverage: float = float(result.mask.mean())
        assert coverage < PRINTED_TEXT_WITH_PRECEDENCE_MASK_COVERAGE_CEILING, (
            f"mask coverage {coverage:.4f} exceeded "
            f"{PRINTED_TEXT_WITH_PRECEDENCE_MASK_COVERAGE_CEILING}; "
            f"handwriting extractor double-claimed printed text already "
            f"covered by printed_mask"
        )

    def test_blank_paper_yields_near_empty_mask(
        self, project_root: Path
    ) -> None:
        # Arrange — pure white paper crop; nothing should fire here even
        # with an empty printed_mask (no precedence claims to honor; the
        # extractor must simply not hallucinate ink on bare substrate).
        page: BgrImage = _load_white_receipt_page(project_root)
        crop: BgrImage = _blank_paper_crop(page)
        paper_color, paper_variation = estimate_paper_tone(crop)
        printed_mask: np.ndarray = _empty_printed_mask_for(crop)
        extractor = HandwrittenLayerExtractor()

        # Act
        result = extractor.extract(
            crop, paper_color, paper_variation, printed_mask
        )

        # Assert — empirical dark-pixel ratio is 0%; 0.1% ceiling tolerates
        # only microscopic false positives from quantization / micro-texture.
        coverage: float = float(result.mask.mean())
        assert coverage < BLANK_PAPER_MASK_COVERAGE_CEILING, (
            f"mask coverage {coverage:.4f} exceeded "
            f"{BLANK_PAPER_MASK_COVERAGE_CEILING}; extractor hallucinated "
            f"handwriting on blank paper"
        )

    def test_tones_at_masked_pixels_preserve_ink_tone(
        self, project_root: Path
    ) -> None:
        # Arrange — handwriting crop again; we need pixels in the mask
        # to inspect their preserved tones.
        page: BgrImage = _load_white_receipt_page(project_root)
        crop: BgrImage = _handwriting_crop(page)
        paper_color, paper_variation = estimate_paper_tone(crop)
        printed_mask: np.ndarray = _empty_printed_mask_for(crop)
        extractor = HandwrittenLayerExtractor()
        result = extractor.extract(
            crop, paper_color, paper_variation, printed_mask
        )

        # Sanity gate: the coverage assertions above already pin this, but
        # if the mask happens to be empty here we'd silently pass with a
        # NaN mean. Skip clearly so the failure is signaled by the other
        # test rather than masked here.
        if not result.mask.any():
            pytest.skip(
                "mask is empty on handwriting crop; coverage assertion "
                "in another test will surface this regression"
            )

        paper_grayscale: int = _bgr_to_grayscale_uint8(paper_color)
        masked_tones: np.ndarray = result.tones[result.mask]

        # Act
        masked_tone_mean: float = float(masked_tones.mean())

        # Assert — masked pixels must read at least 50 grayscale units
        # below the paper baseline. This forbids a degenerate "stuff
        # paper-tone in tones" implementation (which would defeat the
        # whole point of the tones array — preserving pen-pressure
        # variation), and equally forbids a "leave 255 there"
        # implementation. We're pinning that the extractor records the
        # ACTUAL ink tone (anti-aliased grayscale) at the masked pixels.
        gap: float = paper_grayscale - masked_tone_mean
        assert gap >= TONES_BELOW_PAPER_GRAYSCALE_GAP, (
            f"masked-tone mean {masked_tone_mean:.1f} is only {gap:.1f} "
            f"units below paper grayscale {paper_grayscale}; expected "
            f"gap >= {TONES_BELOW_PAPER_GRAYSCALE_GAP}. Tones at masked "
            f"pixels must preserve actual pen-ink darkness."
        )

    def test_tones_outside_mask_are_well_defined_uint8(
        self, project_root: Path
    ) -> None:
        # Arrange — any real crop; the contract here is uniform.
        page: BgrImage = _load_white_receipt_page(project_root)
        crop: BgrImage = _handwriting_crop(page)
        paper_color, paper_variation = estimate_paper_tone(crop)
        printed_mask: np.ndarray = _empty_printed_mask_for(crop)
        extractor = HandwrittenLayerExtractor()

        # Act
        result = extractor.extract(
            crop, paper_color, paper_variation, printed_mask
        )

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

    def test_printed_mask_takes_precedence_even_when_pixels_look_like_ink(
        self, project_root: Path
    ) -> None:
        # Arrange — handwriting crop ("Carlos U. 4/16/26"). Construct a
        # printed_mask that pre-claims a rectangular region overlapping
        # the signature glyphs. Even though those pixels look exactly
        # like handwritten ink (because they ARE handwritten ink), the
        # extractor must respect the precedence rule: pixels already
        # claimed by printed never appear in the handwritten mask.
        # Otherwise the composer would render the same stroke twice and
        # the per-pixel ownership invariant of the v3 layer model would
        # collapse.
        page: BgrImage = _load_white_receipt_page(project_root)
        crop: BgrImage = _handwriting_crop(page)
        paper_color, paper_variation = estimate_paper_tone(crop)

        printed_mask: np.ndarray = _empty_printed_mask_for(crop)
        printed_mask[
            PRINTED_PRECEDENCE_RECT_ROW_START:PRINTED_PRECEDENCE_RECT_ROW_END,
            PRINTED_PRECEDENCE_RECT_COL_START:PRINTED_PRECEDENCE_RECT_COL_END,
        ] = True
        extractor = HandwrittenLayerExtractor()

        # Act
        result = extractor.extract(
            crop, paper_color, paper_variation, printed_mask
        )

        # Assert — strict zero overlap between handwritten mask and
        # printed_mask. Anything else means the same pixel is claimed
        # by two layers.
        overlap_pixel_count: int = int((result.mask & printed_mask).sum())
        assert overlap_pixel_count == 0, (
            f"{overlap_pixel_count} pixels appear in BOTH the handwritten "
            f"mask AND the input printed_mask; precedence rule violated. "
            f"Pixels already claimed by the printed layer must never "
            f"appear in the handwritten layer."
        )
