"""Contract tests for ``Document`` and ``DocumentDecomposer`` (v3 phase 4).

``Document`` is the v3 layered storage model for a single scanned page.
A Document is the composite of a ``PrintedLayer`` (laser toner) and a
``HandwrittenLayer`` (pen ink) over an implicit white-paper canvas.
Paper is not an explicit layer: whatever the two layers don't claim is
paper, rendered as white. The ``composite()`` method is the canonical
way to render the Document back into a single grayscale image — it
starts with a white canvas, lays down handwritten tones at the
handwritten mask, and lays down printed tones at the printed mask
(printed takes precedence on any overlap).

``DocumentDecomposer`` is the orchestrator that turns a raw BGR scan
into a ``Document``. It wires together
``estimate_paper_tone`` -> ``PrintedLayerExtractor`` ->
``HandwrittenLayerExtractor`` and enforces the precedence rule that no
single pixel may be claimed by both layers (handwriting only sees
pixels printed didn't claim).

Tests draw real fixtures from ``corpus/receipts/`` (white paper,
cursive signature) and ``corpus/invoices/good/`` (yellow paper,
horizontal handwritten quantity/code line) so the decomposer's
substrate-aware, precedence-honoring behavior is verified end-to-end
against actual scans.

These tests pin the WHAT (composite rendering rules, precedence,
per-layer coverage envelopes on real pages, return types) without
prescribing HOW the orchestration is implemented.
"""

import cv2
import fitz  # PyMuPDF
import numpy as np
import pytest
from dataclasses import FrozenInstanceError
from pathlib import Path

from docscanner import (
    BgrImage,
    Document,
    DocumentDecomposer,
    HandwrittenLayer,
    HandwrittenLayerExtractor,
    PrintedLayer,
    PrintedLayerExtractor,
)


# --- Corpus paths (relative to project_root) --------------------------------

WHITE_RECEIPT_RELATIVE_PATH = "corpus/receipts/WH_IN_2026_00548.pdf"

# 05020 invoice: yellow paper with substantial printed body content AND
# substantial handwritten content ("3FD Kraft 2 Back order" + signature).
# Same file used by the printed/handwritten layer tests.
YELLOW_INVOICE_RELATIVE_PATH = (
    "corpus/invoices/good/"
    "INV-2026-05020_id-973756_aid-430394_Customer_Invoice"
    "-20260429_150716_0002.jpg"
)


# --- PDF rendering ----------------------------------------------------------

# 200 DPI matches the calibration corpus (~1652x2138 for the WH_IN receipt).
# Same value used by test_printed_layer.py / test_handwritten_layer.py so
# all PDF-backed tests render identically.
PDF_RENDER_DPI = 200


# --- Synthetic-Document dimensions ------------------------------------------
#
# Used by the Document value-object tests. Small fixed dimensions are fine
# because we're verifying composite rendering rules and dataclass metadata,
# not extraction behavior.

SYNTHETIC_DOC_HEIGHT = 10
SYNTHETIC_DOC_WIDTH = 15

# Re-frozen test (`test_document_is_frozen_and_slotted`) tries to assign a
# different shape; keep it distinct from SYNTHETIC_DOC_* so the assertion
# can't accidentally pass against the original value.
REASSIGN_ATTEMPT_SHAPE: tuple[int, int] = (5, 5)


# --- Synthetic painted-region coordinates -----------------------------------
#
# These rectangles are used inside the SYNTHETIC_DOC_HEIGHT x
# SYNTHETIC_DOC_WIDTH canvas to paint mask=True regions for the composite()
# rendering tests. They're small enough to inspect by eye but big enough
# to give the assertions multiple pixels to check (so a one-pixel off-by-
# one in the implementation would still fail the test).

PRINTED_REGION_ROW_START = 2
PRINTED_REGION_ROW_END = 4   # 2 rows tall
PRINTED_REGION_COL_START = 3
PRINTED_REGION_COL_END = 6   # 3 cols wide

HANDWRITTEN_REGION_ROW_START = 6
HANDWRITTEN_REGION_ROW_END = 8   # 2 rows tall
HANDWRITTEN_REGION_COL_START = 4
HANDWRITTEN_REGION_COL_END = 9   # 5 cols wide

# Single pixel where BOTH layers claim ownership; used by the precedence
# test to verify that printed wins over handwritten in composite().
OVERLAP_PIXEL_ROW = 5
OVERLAP_PIXEL_COL = 7


# --- Synthetic tone values --------------------------------------------------
#
# Pure white = paper canvas; the two ink tones are deliberately distinct
# from each other and from paper so the assertions can identify which
# layer landed where in the rendered composite.

PAPER_TONE_VALUE = 255
PRINTED_INK_TONE_VALUE = 50    # dark printed glyph
HANDWRITTEN_INK_TONE_VALUE = 80  # mid-gray pen ink
PRINTED_INK_OVERLAP_TONE = 50    # printed value at the overlap pixel
HANDWRITTEN_INK_OVERLAP_TONE = 120  # handwritten value at the overlap pixel


# --- Mask coverage envelopes (justified inline at assertion sites) ----------

# White receipt has plenty of printed content (header, item lines, totals,
# barcode). Floor of 0.5% is well below the printed-text crop's ~2.81%
# naive dark-pixel ratio after weighting for the page-wide whitespace.
DECOMPOSED_PRINTED_COVERAGE_FLOOR = 0.005

# White receipt has the Carlos signature + circle in roughly one corner.
# Floor of 0.1% is conservative — the signature is small relative to the
# whole page but still leaves a measurable fingerprint in the mask.
DECOMPOSED_HANDWRITTEN_COVERAGE_FLOOR = 0.001

# Yellow invoice has substantial content of BOTH kinds. Same 0.5% floor
# applies independently to each layer; the substrate change must not
# zero out either layer.
YELLOW_INVOICE_PRINTED_COVERAGE_FLOOR = 0.005
YELLOW_INVOICE_HANDWRITTEN_COVERAGE_FLOOR = 0.005


# --- Composite-vs-union rounding tolerance ----------------------------------
#
# The composite()-vs-(printed|handwritten) test allows a small mismatch
# because some inked pixels could legitimately have tone == 255 in
# degenerate cases (anti-aliased edges where the extractor preserved the
# halo). 1% of total inked pixels is loose enough to absorb that without
# letting a real bug — e.g., dropping handwritten pixels entirely — slip
# through.
COMPOSITE_UNION_ROUNDING_TOLERANCE_FRACTION = 0.01


# --- Helpers ----------------------------------------------------------------

def _render_pdf_page0(path: Path, dpi: int = PDF_RENDER_DPI) -> BgrImage:
    """Render page 0 of ``path`` at ``dpi`` and return it as a BGR ndarray.

    Single render path (PyMuPDF, RGB->BGR conversion, no alpha) reused
    across every PDF-backed test in this file — matches the helper used
    in ``test_printed_layer.py`` and ``test_handwritten_layer.py`` so all
    PDF tests render identically.
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
    """Read the 05020 invoice JPG (yellow paper, printed body + handwriting)
    or skip if the corpus file is gone."""
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


def _empty_layer(
    height: int = SYNTHETIC_DOC_HEIGHT,
    width: int = SYNTHETIC_DOC_WIDTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (mask, tones) for an empty layer of the given dimensions.

    Mask is all-False; tones are all-paper (255). Used as the default
    "no-content" filler so synthetic Documents can isolate one layer's
    behavior at a time.
    """
    mask: np.ndarray = np.zeros((height, width), dtype=bool)
    tones: np.ndarray = np.full(
        (height, width), PAPER_TONE_VALUE, dtype=np.uint8
    )
    return mask, tones


def _empty_printed_layer() -> PrintedLayer:
    """Build a PrintedLayer with no claimed pixels, sized for synthetic tests."""
    mask, tones = _empty_layer()
    return PrintedLayer(mask=mask, tones=tones)


def _empty_handwritten_layer() -> HandwrittenLayer:
    """Build a HandwrittenLayer with no claimed pixels, sized for synthetic tests."""
    mask, tones = _empty_layer()
    return HandwrittenLayer(mask=mask, tones=tones)


def _printed_layer_with_painted_region(
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    tone_value: int,
) -> PrintedLayer:
    """Build a PrintedLayer where the given rectangle is masked True with
    ``tone_value`` painted into tones; everywhere else is mask=False and
    tones=paper-white. Lets the composite() tests assert pixel-level
    rendering against a known-region painted pattern.
    """
    mask, tones = _empty_layer()
    mask[row_start:row_end, col_start:col_end] = True
    tones[row_start:row_end, col_start:col_end] = tone_value
    return PrintedLayer(mask=mask, tones=tones)


def _handwritten_layer_with_painted_region(
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    tone_value: int,
) -> HandwrittenLayer:
    """Mirror of ``_printed_layer_with_painted_region`` for HandwrittenLayer.

    Same rationale: keep the composite() tests' setup explicit but compact.
    """
    mask, tones = _empty_layer()
    mask[row_start:row_end, col_start:col_end] = True
    tones[row_start:row_end, col_start:col_end] = tone_value
    return HandwrittenLayer(mask=mask, tones=tones)


# --- Tests ------------------------------------------------------------------


class TestDocument:
    """Contract for the ``Document`` value object and its ``composite()``
    rendering method.

    Pins:
      * Field types (shape tuple, PrintedLayer, HandwrittenLayer).
      * Frozen / slotted dataclass semantics — Documents are immutable so
        the orchestration pipeline can pass them across stages without
        defensive copies.
      * ``composite()`` rendering rules:
          - empty Document -> pure white canvas (the implicit-paper rule);
          - printed mask -> printed tones land at those pixels;
          - handwritten mask -> handwritten tones land at those pixels;
          - on overlap, printed wins (precedence rule, defensive even
            though extractors should prevent overlap upstream);
          - output is uint8 with shape == doc.shape.
    """

    def test_document_fields_have_required_shape_and_layer_types(self) -> None:
        # Arrange — both layers must share the Document's declared shape.
        printed: PrintedLayer = _empty_printed_layer()
        handwritten: HandwrittenLayer = _empty_handwritten_layer()
        declared_shape: tuple[int, int] = (
            SYNTHETIC_DOC_HEIGHT,
            SYNTHETIC_DOC_WIDTH,
        )

        # Act
        doc = Document(
            shape=declared_shape, printed=printed, handwritten=handwritten
        )

        # Assert — field types + shape are part of the contract; downstream
        # consumers index doc.shape as (H, W) and dispatch on layer type.
        assert doc.shape == declared_shape, (
            f"doc.shape={doc.shape}, expected {declared_shape}"
        )
        assert isinstance(doc.printed, PrintedLayer), (
            f"doc.printed type={type(doc.printed).__name__}, "
            f"expected PrintedLayer"
        )
        assert isinstance(doc.handwritten, HandwrittenLayer), (
            f"doc.handwritten type={type(doc.handwritten).__name__}, "
            f"expected HandwrittenLayer"
        )

    def test_document_is_frozen_and_slotted(self) -> None:
        # Arrange — any well-formed Document; we only care about mutation.
        doc = Document(
            shape=(SYNTHETIC_DOC_HEIGHT, SYNTHETIC_DOC_WIDTH),
            printed=_empty_printed_layer(),
            handwritten=_empty_handwritten_layer(),
        )

        # Act / Assert — frozen dataclass raises FrozenInstanceError; a
        # slotted dataclass raises AttributeError on unknown attributes.
        # Either is acceptable evidence of "you cannot mutate me," matching
        # the precedent established by PrintedLayer / HandwrittenLayer
        # in test_printed_layer.py and test_handwritten_layer.py.
        with pytest.raises((FrozenInstanceError, AttributeError)):
            doc.shape = REASSIGN_ATTEMPT_SHAPE  # type: ignore[misc]

    def test_composite_with_empty_layers_is_pure_white(self) -> None:
        # Arrange — both layers all-False with tones=255; the composite
        # must therefore be the implicit-paper canvas: every pixel 255.
        # This test pins the "paper is not a layer; whatever isn't claimed
        # is white" rule.
        doc = Document(
            shape=(SYNTHETIC_DOC_HEIGHT, SYNTHETIC_DOC_WIDTH),
            printed=_empty_printed_layer(),
            handwritten=_empty_handwritten_layer(),
        )

        # Act
        rendered: np.ndarray = doc.composite()

        # Assert — pure white canvas at the declared shape.
        expected_canvas: np.ndarray = np.full(
            (SYNTHETIC_DOC_HEIGHT, SYNTHETIC_DOC_WIDTH),
            PAPER_TONE_VALUE,
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(
            rendered,
            expected_canvas,
            err_msg=(
                f"composite() of empty Document was not pure white; "
                f"unique values present: {np.unique(rendered).tolist()}"
            ),
        )

    def test_composite_renders_printed_tones_at_printed_mask(self) -> None:
        # Arrange — printed layer paints a small rectangle with
        # PRINTED_INK_TONE_VALUE; handwritten layer is empty. Composite
        # must show the printed tone in the rectangle and pure paper
        # elsewhere.
        printed: PrintedLayer = _printed_layer_with_painted_region(
            PRINTED_REGION_ROW_START,
            PRINTED_REGION_ROW_END,
            PRINTED_REGION_COL_START,
            PRINTED_REGION_COL_END,
            PRINTED_INK_TONE_VALUE,
        )
        doc = Document(
            shape=(SYNTHETIC_DOC_HEIGHT, SYNTHETIC_DOC_WIDTH),
            printed=printed,
            handwritten=_empty_handwritten_layer(),
        )

        # Act
        rendered: np.ndarray = doc.composite()

        # Assert — every pixel in the printed rectangle reads as the
        # printed ink tone; everything outside reads as paper.
        printed_region: np.ndarray = rendered[
            PRINTED_REGION_ROW_START:PRINTED_REGION_ROW_END,
            PRINTED_REGION_COL_START:PRINTED_REGION_COL_END,
        ]
        assert np.all(printed_region == PRINTED_INK_TONE_VALUE), (
            f"printed region values: {np.unique(printed_region).tolist()}, "
            f"expected uniformly {PRINTED_INK_TONE_VALUE}"
        )
        # Build a "everything outside the printed rectangle" mask to
        # verify the rest is paper. Done with a boolean mask rather than
        # row/col slices so the L-shape around the rectangle is covered.
        outside_mask: np.ndarray = np.ones(
            (SYNTHETIC_DOC_HEIGHT, SYNTHETIC_DOC_WIDTH), dtype=bool
        )
        outside_mask[
            PRINTED_REGION_ROW_START:PRINTED_REGION_ROW_END,
            PRINTED_REGION_COL_START:PRINTED_REGION_COL_END,
        ] = False
        outside_values: np.ndarray = rendered[outside_mask]
        assert np.all(outside_values == PAPER_TONE_VALUE), (
            f"pixels outside the printed region were not paper; "
            f"unique outside values: {np.unique(outside_values).tolist()}"
        )

    def test_composite_renders_handwritten_tones_at_handwritten_mask(self) -> None:
        # Arrange — symmetric to the printed-only test: handwritten layer
        # paints a small rectangle with HANDWRITTEN_INK_TONE_VALUE;
        # printed layer is empty. Pins that handwritten tones survive
        # into the rendered output (a degenerate implementation that
        # only renders printed would pass the previous test but fail
        # this one).
        handwritten: HandwrittenLayer = _handwritten_layer_with_painted_region(
            HANDWRITTEN_REGION_ROW_START,
            HANDWRITTEN_REGION_ROW_END,
            HANDWRITTEN_REGION_COL_START,
            HANDWRITTEN_REGION_COL_END,
            HANDWRITTEN_INK_TONE_VALUE,
        )
        doc = Document(
            shape=(SYNTHETIC_DOC_HEIGHT, SYNTHETIC_DOC_WIDTH),
            printed=_empty_printed_layer(),
            handwritten=handwritten,
        )

        # Act
        rendered: np.ndarray = doc.composite()

        # Assert — handwritten ink tone inside the rectangle, paper outside.
        handwritten_region: np.ndarray = rendered[
            HANDWRITTEN_REGION_ROW_START:HANDWRITTEN_REGION_ROW_END,
            HANDWRITTEN_REGION_COL_START:HANDWRITTEN_REGION_COL_END,
        ]
        assert np.all(handwritten_region == HANDWRITTEN_INK_TONE_VALUE), (
            f"handwritten region values: "
            f"{np.unique(handwritten_region).tolist()}, expected "
            f"uniformly {HANDWRITTEN_INK_TONE_VALUE}"
        )
        outside_mask: np.ndarray = np.ones(
            (SYNTHETIC_DOC_HEIGHT, SYNTHETIC_DOC_WIDTH), dtype=bool
        )
        outside_mask[
            HANDWRITTEN_REGION_ROW_START:HANDWRITTEN_REGION_ROW_END,
            HANDWRITTEN_REGION_COL_START:HANDWRITTEN_REGION_COL_END,
        ] = False
        outside_values: np.ndarray = rendered[outside_mask]
        assert np.all(outside_values == PAPER_TONE_VALUE), (
            f"pixels outside the handwritten region were not paper; "
            f"unique outside values: {np.unique(outside_values).tolist()}"
        )

    def test_composite_printed_takes_precedence_over_handwritten_on_overlap(
        self,
    ) -> None:
        # Arrange — both layers claim a single overlap pixel with
        # different tone values. Even though the upstream extractors'
        # precedence rule should prevent overlap in production, the
        # composite() method must implement printed-wins-over-handwritten
        # consistently as a defense-in-depth invariant: a malformed
        # Document built by hand (e.g., a test fixture, or a future
        # caller bypassing the decomposer) must still render with the
        # documented precedence.
        printed_mask: np.ndarray = np.zeros(
            (SYNTHETIC_DOC_HEIGHT, SYNTHETIC_DOC_WIDTH), dtype=bool
        )
        printed_mask[OVERLAP_PIXEL_ROW, OVERLAP_PIXEL_COL] = True
        printed_tones: np.ndarray = np.full(
            (SYNTHETIC_DOC_HEIGHT, SYNTHETIC_DOC_WIDTH),
            PAPER_TONE_VALUE,
            dtype=np.uint8,
        )
        printed_tones[OVERLAP_PIXEL_ROW, OVERLAP_PIXEL_COL] = (
            PRINTED_INK_OVERLAP_TONE
        )
        printed = PrintedLayer(mask=printed_mask, tones=printed_tones)

        handwritten_mask: np.ndarray = np.zeros(
            (SYNTHETIC_DOC_HEIGHT, SYNTHETIC_DOC_WIDTH), dtype=bool
        )
        handwritten_mask[OVERLAP_PIXEL_ROW, OVERLAP_PIXEL_COL] = True
        handwritten_tones: np.ndarray = np.full(
            (SYNTHETIC_DOC_HEIGHT, SYNTHETIC_DOC_WIDTH),
            PAPER_TONE_VALUE,
            dtype=np.uint8,
        )
        handwritten_tones[OVERLAP_PIXEL_ROW, OVERLAP_PIXEL_COL] = (
            HANDWRITTEN_INK_OVERLAP_TONE
        )
        handwritten = HandwrittenLayer(
            mask=handwritten_mask, tones=handwritten_tones
        )

        doc = Document(
            shape=(SYNTHETIC_DOC_HEIGHT, SYNTHETIC_DOC_WIDTH),
            printed=printed,
            handwritten=handwritten,
        )

        # Act
        rendered: np.ndarray = doc.composite()

        # Assert — at the overlap pixel, the printed tone wins.
        # PRINTED_INK_OVERLAP_TONE != HANDWRITTEN_INK_OVERLAP_TONE so this
        # cannot accidentally pass on a "handwritten wins" implementation.
        rendered_overlap_value: int = int(
            rendered[OVERLAP_PIXEL_ROW, OVERLAP_PIXEL_COL]
        )
        assert rendered_overlap_value == PRINTED_INK_OVERLAP_TONE, (
            f"composite()[{OVERLAP_PIXEL_ROW}, {OVERLAP_PIXEL_COL}] = "
            f"{rendered_overlap_value}, expected {PRINTED_INK_OVERLAP_TONE} "
            f"(printed must win over handwritten on overlap)"
        )

    def test_composite_returns_uint8_with_correct_shape(self) -> None:
        # Arrange — any Document; the contract here is uniform.
        doc = Document(
            shape=(SYNTHETIC_DOC_HEIGHT, SYNTHETIC_DOC_WIDTH),
            printed=_empty_printed_layer(),
            handwritten=_empty_handwritten_layer(),
        )

        # Act
        rendered: np.ndarray = doc.composite()

        # Assert — shape and dtype contract. Downstream consumers (storage
        # writers, OCR feeders) read the composite as a uint8 grayscale
        # array sized to the original page; deviations break those
        # consumers silently.
        assert rendered.shape == doc.shape, (
            f"composite().shape={rendered.shape}, expected {doc.shape}"
        )
        assert rendered.dtype == np.uint8, (
            f"composite().dtype={rendered.dtype}, expected uint8"
        )


class TestDocumentDecomposer:
    """Contract for ``DocumentDecomposer.decompose(bgr) -> Document``.

    Pins:
      * Returns a ``Document`` whose declared shape matches the input's
        (H, W).
      * Returns real PrintedLayer / HandwrittenLayer instances whose
        masks/tones also match the input shape.
      * Honors strict per-pixel precedence: the printed and handwritten
        masks NEVER overlap on any real scan.
      * On the white receipt: substantial printed coverage, measurable
        handwritten coverage (Carlos signature + circle).
      * On the yellow invoice: substantial coverage in BOTH layers
        (printed body + horizontal handwritten line + signature).
      * ``composite()`` faithfully renders both layers without dropping
        pixels — the count of inked pixels in the composite matches the
        union of the two layer masks within rounding tolerance.
    """

    def test_decompose_returns_document_with_shape_matching_input(
        self, project_root: Path
    ) -> None:
        # Arrange
        bgr: BgrImage = _load_white_receipt_page(project_root)
        decomposer = DocumentDecomposer()

        # Act
        doc: Document = decomposer.decompose(bgr)

        # Assert — the Document must declare the exact (H, W) of the
        # input. Any rescale or padding would silently break downstream
        # consumers that align on coordinates from the original scan.
        assert doc.shape == bgr.shape[:2], (
            f"doc.shape={doc.shape}, expected {bgr.shape[:2]}"
        )

    def test_decompose_returns_document_with_real_layer_instances(
        self, project_root: Path
    ) -> None:
        # Arrange
        bgr: BgrImage = _load_white_receipt_page(project_root)
        decomposer = DocumentDecomposer()
        expected_layer_shape: tuple[int, int] = bgr.shape[:2]

        # Act
        doc: Document = decomposer.decompose(bgr)

        # Assert — both layers must be the v3 carrier types (not bare
        # ndarrays or dicts) and their mask/tones arrays must match the
        # input H/W so the precedence-overlap and composite assertions
        # downstream have a meaningful coordinate system.
        assert isinstance(doc.printed, PrintedLayer), (
            f"doc.printed type={type(doc.printed).__name__}, "
            f"expected PrintedLayer"
        )
        assert isinstance(doc.handwritten, HandwrittenLayer), (
            f"doc.handwritten type={type(doc.handwritten).__name__}, "
            f"expected HandwrittenLayer"
        )
        assert doc.printed.mask.shape == expected_layer_shape, (
            f"doc.printed.mask.shape={doc.printed.mask.shape}, "
            f"expected {expected_layer_shape}"
        )
        assert doc.printed.tones.shape == expected_layer_shape, (
            f"doc.printed.tones.shape={doc.printed.tones.shape}, "
            f"expected {expected_layer_shape}"
        )
        assert doc.handwritten.mask.shape == expected_layer_shape, (
            f"doc.handwritten.mask.shape={doc.handwritten.mask.shape}, "
            f"expected {expected_layer_shape}"
        )
        assert doc.handwritten.tones.shape == expected_layer_shape, (
            f"doc.handwritten.tones.shape={doc.handwritten.tones.shape}, "
            f"expected {expected_layer_shape}"
        )

    def test_decompose_layers_have_zero_mask_overlap(
        self, project_root: Path
    ) -> None:
        # Arrange — full white receipt page; the decomposer must apply
        # the precedence rule end-to-end so that no single pixel ends up
        # in both layer masks.
        bgr: BgrImage = _load_white_receipt_page(project_root)
        decomposer = DocumentDecomposer()

        # Act
        doc: Document = decomposer.decompose(bgr)

        # Assert — strict zero overlap. Any non-zero count means the
        # composer would draw the same stroke twice, the per-pixel
        # ownership invariant of the v3 layer model collapses, and any
        # downstream code that sums layer coverages would over-count.
        overlap_pixel_count: int = int(
            (doc.printed.mask & doc.handwritten.mask).sum()
        )
        assert overlap_pixel_count == 0, (
            f"{overlap_pixel_count} pixels appear in BOTH the printed "
            f"and handwritten masks; precedence rule violated end-to-end "
            f"by the decomposer"
        )

    def test_decompose_white_receipt_yields_substantial_printed_coverage(
        self, project_root: Path
    ) -> None:
        # Arrange — the white receipt has plenty of printed content:
        # company header, line items, totals, barcode. The decomposer
        # must surface all of that into doc.printed.mask.
        bgr: BgrImage = _load_white_receipt_page(project_root)
        decomposer = DocumentDecomposer()

        # Act
        doc: Document = decomposer.decompose(bgr)

        # Assert — at least 0.5% of the page is claimed by printed. This
        # is well below the printed-text crop's ~2.81% naive ratio after
        # weighting for the page-wide whitespace; the floor exists to
        # catch a regression where the extractor is wired correctly but
        # outputs an empty mask for an entire page.
        coverage: float = float(doc.printed.mask.mean())
        assert coverage > DECOMPOSED_PRINTED_COVERAGE_FLOOR, (
            f"printed coverage {coverage:.4f} did not exceed "
            f"{DECOMPOSED_PRINTED_COVERAGE_FLOOR}; decomposer failed "
            f"to detect printed content on the white receipt"
        )

    def test_decompose_white_receipt_yields_some_handwritten_coverage(
        self, project_root: Path
    ) -> None:
        # Arrange — same input. The Carlos signature + hand-drawn circle
        # is small relative to the whole page but must still leave a
        # measurable footprint in doc.handwritten.mask.
        bgr: BgrImage = _load_white_receipt_page(project_root)
        decomposer = DocumentDecomposer()

        # Act
        doc: Document = decomposer.decompose(bgr)

        # Assert — at least 0.1% of the page is claimed by handwriting.
        # Conservative because the signature is a small region; the
        # important regression to catch is "handwriting layer is empty
        # on a page that clearly has handwriting."
        coverage: float = float(doc.handwritten.mask.mean())
        assert coverage > DECOMPOSED_HANDWRITTEN_COVERAGE_FLOOR, (
            f"handwritten coverage {coverage:.4f} did not exceed "
            f"{DECOMPOSED_HANDWRITTEN_COVERAGE_FLOOR}; decomposer failed "
            f"to detect the Carlos signature on the white receipt"
        )

    def test_decompose_yellow_invoice_yields_both_layers_populated(
        self, project_root: Path
    ) -> None:
        # Arrange — the 05020 invoice is the dual-content fixture: yellow
        # paper, substantial printed body, plus the horizontal "3FD Kraft
        # 2 Back order" handwritten line and a signature. Both layers
        # must come back populated; if either is empty the substrate
        # adaptation has broken.
        bgr: BgrImage = _load_yellow_invoice_page(project_root)
        decomposer = DocumentDecomposer()

        # Act
        doc: Document = decomposer.decompose(bgr)

        # Assert — independent floors on each layer. Asserting one at a
        # time so the failure message identifies which side broke.
        printed_coverage: float = float(doc.printed.mask.mean())
        handwritten_coverage: float = float(doc.handwritten.mask.mean())
        assert printed_coverage > YELLOW_INVOICE_PRINTED_COVERAGE_FLOOR, (
            f"printed coverage {printed_coverage:.4f} did not exceed "
            f"{YELLOW_INVOICE_PRINTED_COVERAGE_FLOOR}; decomposer failed "
            f"to detect printed content on the yellow invoice"
        )
        assert handwritten_coverage > YELLOW_INVOICE_HANDWRITTEN_COVERAGE_FLOOR, (
            f"handwritten coverage {handwritten_coverage:.4f} did not "
            f"exceed {YELLOW_INVOICE_HANDWRITTEN_COVERAGE_FLOOR}; "
            f"decomposer failed to detect handwritten content "
            f"('3FD Kraft 2 Back order' + signature) on the yellow "
            f"invoice"
        )

    def test_decompose_composite_pixel_count_matches_layer_union(
        self, project_root: Path
    ) -> None:
        # Arrange — full white receipt page. Decompose and render the
        # composite; the count of non-white pixels in the composite
        # should equal the count of pixels in the union of the two
        # layer masks (within rounding tolerance, since the extractors
        # may legitimately preserve some anti-aliased edge pixels at
        # tone == 255 in degenerate cases). This pins that composite()
        # faithfully renders both layers without dropping pixels.
        bgr: BgrImage = _load_white_receipt_page(project_root)
        decomposer = DocumentDecomposer()
        doc: Document = decomposer.decompose(bgr)

        # Act
        rendered: np.ndarray = doc.composite()
        non_white_pixel_count: int = int((rendered != PAPER_TONE_VALUE).sum())
        union_pixel_count: int = int(
            (doc.printed.mask | doc.handwritten.mask).sum()
        )

        # Sanity gate: if the union is somehow zero we'd divide by zero
        # below AND the page-coverage assertions in earlier tests would
        # already be failing. Skip cleanly so the failure is signaled
        # there rather than masked here.
        if union_pixel_count == 0:
            pytest.skip(
                "layer-mask union is empty on white receipt; coverage "
                "assertions in other tests will surface this regression"
            )

        # Assert — the two counts must match within
        # COMPOSITE_UNION_ROUNDING_TOLERANCE_FRACTION of the union size.
        # Using absolute difference (not signed) because either direction
        # is a bug: composite drawing extra pixels would be just as bad
        # as dropping them.
        absolute_difference: int = abs(
            non_white_pixel_count - union_pixel_count
        )
        allowed_difference: float = (
            COMPOSITE_UNION_ROUNDING_TOLERANCE_FRACTION * union_pixel_count
        )
        assert absolute_difference <= allowed_difference, (
            f"composite() non-white pixel count "
            f"({non_white_pixel_count}) differs from layer-mask union "
            f"({union_pixel_count}) by {absolute_difference} pixels, "
            f"exceeding the allowed "
            f"{COMPOSITE_UNION_ROUNDING_TOLERANCE_FRACTION:.0%} tolerance "
            f"({allowed_difference:.1f} pixels). composite() may be "
            f"dropping or duplicating pixels relative to the layer masks."
        )
