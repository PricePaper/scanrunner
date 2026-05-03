"""Contract tests for ``estimate_paper_tone`` — the v3 paper-tone primitive.

``estimate_paper_tone`` returns the dominant paper-substrate color and a
scalar ``tone_variation`` (CIE Lab ΔE units) describing how much the paper
population deviates from that base color. Both the print-layer extractor
and the handwriting extractor consume this primitive as a contrast-threshold
input: a marbled / textured sheet must tolerate more deviation before a
pixel is classified as "ink" than a pristine white sheet does.

Tests draw paper-substrate samples from ``corpus/receipts/`` (white) and
``corpus/invoices/good/`` (yellow). Synthetic input is reserved for the
marbled-vs-uniform monotonicity case where controlled means are required.

These tests pin the WHAT (median-of-substrate, ink excluded, variation
monotone in noise) without prescribing HOW the function computes it.
"""

import cv2
import fitz  # PyMuPDF
import numpy as np
import pytest
from pathlib import Path

from docscanner import estimate_paper_tone, BgrImage, BgrColor


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
# Crops were probed against the real corpus files; means/std reported are
# the empirical values used to choose the assertion tolerances below.

# WHITE_PAPER crop on WH_IN_2026_00548.pdf page 0 (rendered ~1652x2138):
# `bgr[h-1000:h-600, w//2-200:w//2+200]` -> 400x400 of pure white substrate.
# Empirical: BGR ~ (255, 255, 255), ΔE_lab std ~ 0.35.
WHITE_PAPER_ROW_OFFSET_FROM_BOTTOM_START = 1000
WHITE_PAPER_ROW_OFFSET_FROM_BOTTOM_END = 600
WHITE_PAPER_COL_HALFWIDTH = 200

# HEADER_TEXT crop on WH_IN_2026_00548.pdf page 0:
# `bgr[40:440, 60:660]` -> 400x600 with company name, address, dates,
# barcode -> printed text on top of paper.
# Naive whole-crop mean ~ BGR (243, 243, 243); a correct paper extractor
# must rise well above that by excluding ink.
HEADER_ROW_START = 40
HEADER_ROW_END = 440
HEADER_COL_START = 60
HEADER_COL_END = 660

# YELLOW_PAPER crop on the INV-2026-05015 ..._0020.jpg sample (~2548x3332):
# `bgr[h//2-150:h//2+150, w-380:w-80]` -> 300x300 of clean yellow margin.
# Empirical: BGR ~ (152, 228, 237), ΔE_lab std ~ 1.90.
YELLOW_PAPER_ROW_HALFHEIGHT = 150
YELLOW_PAPER_COL_OFFSET_FROM_RIGHT_START = 380
YELLOW_PAPER_COL_OFFSET_FROM_RIGHT_END = 80


# --- Tolerance budgets (justified inline at assertion sites) ----------------

# White-paper crop: every BGR channel must read at least this bright.
# Empirical channel means are ~255; floor of 250 leaves headroom for any
# substrate-population filter that trims a small fraction of darker pixels.
WHITE_PAPER_CHANNEL_FLOOR = 250

# Variation ceiling on the white-paper crop. Crop's intrinsic ΔE std is
# ~0.35; ceiling of 1.0 is a generous bound that still catches ink leakage.
WHITE_PAPER_VARIATION_CEILING = 1.0

# Yellow-paper acceptance envelope (BGR). Empirical mean is (152, 228, 237);
# bounds widened by ~12 around each channel to allow for legitimate
# substrate-population filtering choices.
YELLOW_PAPER_B_MIN, YELLOW_PAPER_B_MAX = 140, 165
YELLOW_PAPER_G_MIN, YELLOW_PAPER_G_MAX = 220, 240
YELLOW_PAPER_R_MIN, YELLOW_PAPER_R_MAX = 225, 245

# Variation ceiling on the yellow-paper crop. Crop's intrinsic ΔE std is
# ~1.90; ceiling of 5.0 allows for picky population selection that may
# discard outliers and tighten the residual variance differently than a
# raw std would.
YELLOW_PAPER_VARIATION_CEILING = 5.0

# Header crop: each channel must rise above this floor. The naive mean of
# the header crop (text + paper) is ~243 per channel — already light
# because most pixels are paper. Floor of 248 forces the implementation to
# actually exclude ink rather than just average everything.
HEADER_PAPER_CHANNEL_FLOOR = 248

# Marbled-vs-uniform synthetic comparison.
PAGE_HEIGHT = 400
PAGE_WIDTH = 600
MARBLED_MEAN = 245           # target mean luminance, near-white
MARBLED_NOISE_SIGMA = 8.0    # std-dev of additive Gaussian texture
MARBLED_NOISE_LO = 220       # clip floor — keep it within "still paper"
MARBLED_NOISE_HI = 255       # clip ceiling — uint8 max
MARBLED_SEED = 20260503      # deterministic noise; reproducible across runs
VARIATION_SEPARATION_MIN = 0.5  # marbled minus uniform, ΔE units


# --- Helpers ----------------------------------------------------------------

def _render_pdf_page0(path: Path, dpi: int = PDF_RENDER_DPI) -> BgrImage:
    """Render page 0 of ``path`` at ``dpi`` and return it as a BGR ndarray.

    Reused across every PDF-backed test so all callers go through one
    well-defined render path (PyMuPDF, RGB->BGR conversion, no alpha).
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


def _white_paper_crop(page: BgrImage) -> BgrImage:
    """Slice the verified-clean WHITE_PAPER crop from a rendered receipt page."""
    height, width = page.shape[:2]
    row_start: int = height - WHITE_PAPER_ROW_OFFSET_FROM_BOTTOM_START
    row_end: int = height - WHITE_PAPER_ROW_OFFSET_FROM_BOTTOM_END
    col_center: int = width // 2
    col_start: int = col_center - WHITE_PAPER_COL_HALFWIDTH
    col_end: int = col_center + WHITE_PAPER_COL_HALFWIDTH
    return page[row_start:row_end, col_start:col_end]


def _header_text_crop(page: BgrImage) -> BgrImage:
    """Slice the verified-clean HEADER_TEXT crop (text + paper)."""
    return page[
        HEADER_ROW_START:HEADER_ROW_END,
        HEADER_COL_START:HEADER_COL_END,
    ]


def _yellow_paper_crop(page: BgrImage) -> BgrImage:
    """Slice the verified-clean YELLOW_PAPER crop from the invoice JPG."""
    height, width = page.shape[:2]
    row_center: int = height // 2
    row_start: int = row_center - YELLOW_PAPER_ROW_HALFHEIGHT
    row_end: int = row_center + YELLOW_PAPER_ROW_HALFHEIGHT
    col_start: int = width - YELLOW_PAPER_COL_OFFSET_FROM_RIGHT_START
    col_end: int = width - YELLOW_PAPER_COL_OFFSET_FROM_RIGHT_END
    return page[row_start:row_end, col_start:col_end]


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


def _solid_page(bgr_color: tuple[int, int, int]) -> BgrImage:
    """Build a uniform PAGE_HEIGHT x PAGE_WIDTH page of a single BGR color.

    Used only by the marbled-vs-uniform synthetic comparison; that test
    requires controlled means to isolate the variation signal.
    """
    page: BgrImage = np.full(
        (PAGE_HEIGHT, PAGE_WIDTH, 3), bgr_color, dtype=np.uint8
    )
    return page


def _marbled_page() -> BgrImage:
    """Near-white page with low-amplitude Gaussian texture (marbled stock)."""
    rng = np.random.default_rng(MARBLED_SEED)
    noise: np.ndarray = rng.normal(
        loc=MARBLED_MEAN,
        scale=MARBLED_NOISE_SIGMA,
        size=(PAGE_HEIGHT, PAGE_WIDTH, 3),
    )
    clipped: np.ndarray = np.clip(noise, MARBLED_NOISE_LO, MARBLED_NOISE_HI)
    page: BgrImage = clipped.astype(np.uint8)
    return page


class TestEstimatePaperTone:
    """Contract for ``estimate_paper_tone(bgr) -> (BgrColor, float)``.

    Verifies:
      * Return type and value-range invariants on a real white-paper crop.
      * Real white-receipt substrate -> near-white color, low variation.
      * Real yellow-invoice substrate -> yellow envelope, low variation.
      * On a real header crop (text + paper) the result tracks paper, not
        the ink-inclusive naive mean.
      * ``tone_variation`` is monotone in actual paper texture
        (synthetic, controlled-mean comparison).
      * The function handles both white and yellow substrates without
        being hardcoded to one.
    """

    # 1. Type contract -------------------------------------------------------

    def test_returns_three_int_tuple_and_finite_nonnegative_float(
        self, project_root: Path
    ) -> None:
        # Arrange — real white-paper crop from the receipts corpus.
        page: BgrImage = _load_white_receipt_page(project_root)
        crop: BgrImage = _white_paper_crop(page)

        # Act
        paper_color, tone_variation = estimate_paper_tone(crop)

        # Assert — shape of the return value
        assert isinstance(paper_color, tuple), (
            f"paper_color must be a tuple, got {type(paper_color).__name__}"
        )
        assert len(paper_color) == 3, (
            f"paper_color must have 3 channels, got {len(paper_color)}"
        )
        for channel_index, channel_value in enumerate(paper_color):
            # ints, not numpy scalars — downstream code uses these as JSON /
            # tuple keys and python-int comparisons.
            assert isinstance(channel_value, int), (
                f"channel {channel_index} must be int, "
                f"got {type(channel_value).__name__}"
            )
            assert 0 <= channel_value <= 255, (
                f"channel {channel_index} value {channel_value} "
                f"outside uint8 range"
            )

        # Assert — variation is a real, finite, non-negative scalar
        assert isinstance(tone_variation, float), (
            f"tone_variation must be float, "
            f"got {type(tone_variation).__name__}"
        )
        assert np.isfinite(tone_variation), (
            f"tone_variation {tone_variation} must be finite"
        )
        assert tone_variation >= 0.0, (
            f"tone_variation {tone_variation} must be >= 0"
        )

    # 2. White receipt paper -------------------------------------------------

    def test_white_receipt_paper_returns_near_white_with_low_variation(
        self, project_root: Path
    ) -> None:
        # Arrange — verified-clean WHITE_PAPER crop from page 0 of
        # WH_IN_2026_00548.pdf. Empirical channel means ~255, ΔE std ~0.35.
        page: BgrImage = _load_white_receipt_page(project_root)
        crop: BgrImage = _white_paper_crop(page)

        # Act
        paper_color, tone_variation = estimate_paper_tone(crop)

        # Assert — every channel must read as near-white. Floor of 250
        # leaves room for substrate-population filters that may trim a
        # small fraction of darker pixels but should not pull the median
        # appreciably away from 255 on a clean crop.
        for channel_index, channel_value in enumerate(paper_color):
            assert channel_value >= WHITE_PAPER_CHANNEL_FLOOR, (
                f"channel {channel_index}={channel_value} dropped below "
                f"{WHITE_PAPER_CHANNEL_FLOOR}; paper_color={paper_color}"
            )

        # Assert — intrinsic variation in the crop is ~0.35 ΔE; a generous
        # ceiling of 1.0 still catches any ink-leakage regression.
        assert tone_variation < WHITE_PAPER_VARIATION_CEILING, (
            f"tone_variation={tone_variation}, expected < "
            f"{WHITE_PAPER_VARIATION_CEILING}"
        )

    # 3. Yellow invoice paper ------------------------------------------------

    def test_yellow_invoice_paper_returns_yellow_with_low_variation(
        self, project_root: Path
    ) -> None:
        # Arrange — verified-clean YELLOW_PAPER crop from the invoice JPG.
        # Empirical mean BGR ~ (152, 228, 237), ΔE std ~1.90.
        page: BgrImage = _load_yellow_invoice_page(project_root)
        crop: BgrImage = _yellow_paper_crop(page)

        # Act
        paper_color, tone_variation = estimate_paper_tone(crop)
        blue, green, red = paper_color

        # Assert — paper color falls inside the yellow envelope. Bounds
        # are ~12 wide around each empirical channel mean, allowing
        # legitimate population-filter choices but rejecting any drift
        # toward neutral/white or toward saturated ink.
        assert YELLOW_PAPER_B_MIN <= blue <= YELLOW_PAPER_B_MAX, (
            f"B={blue} outside [{YELLOW_PAPER_B_MIN}, "
            f"{YELLOW_PAPER_B_MAX}]; paper_color={paper_color}"
        )
        assert YELLOW_PAPER_G_MIN <= green <= YELLOW_PAPER_G_MAX, (
            f"G={green} outside [{YELLOW_PAPER_G_MIN}, "
            f"{YELLOW_PAPER_G_MAX}]; paper_color={paper_color}"
        )
        assert YELLOW_PAPER_R_MIN <= red <= YELLOW_PAPER_R_MAX, (
            f"R={red} outside [{YELLOW_PAPER_R_MIN}, "
            f"{YELLOW_PAPER_R_MAX}]; paper_color={paper_color}"
        )

        # Assert — yellow paper has more intrinsic noise than white
        # (~1.90 ΔE), so we allow up to 5.0 ΔE before flagging trouble.
        assert tone_variation < YELLOW_PAPER_VARIATION_CEILING, (
            f"tone_variation={tone_variation}, expected < "
            f"{YELLOW_PAPER_VARIATION_CEILING}"
        )

    # 4. Header crop (text + paper) — paper wins over ink mean --------------

    def test_header_with_text_returns_paper_color_not_ink_average(
        self, project_root: Path
    ) -> None:
        # Arrange — HEADER_TEXT crop from page 0: company name, address,
        # dates, barcode on top of white paper. Naive whole-crop BGR mean
        # is already ~(243, 243, 243) because most pixels are paper, so
        # a floor at 248 specifically demands the implementation EXCLUDE
        # ink, not just average everything.
        page: BgrImage = _load_white_receipt_page(project_root)
        crop: BgrImage = _header_text_crop(page)

        # Act
        paper_color, _tone_variation = estimate_paper_tone(crop)

        # Assert — paper color rises above the naive mean. This is the
        # core "ignore ink" contract: ink pixels (dark text, barcode bars)
        # must NOT be members of the paper-substrate population.
        for channel_index, channel_value in enumerate(paper_color):
            assert channel_value >= HEADER_PAPER_CHANNEL_FLOOR, (
                f"channel {channel_index}={channel_value} dropped below "
                f"{HEADER_PAPER_CHANNEL_FLOOR} (naive mean ~243); ink "
                f"pixels likely leaked into paper-substrate population; "
                f"paper_color={paper_color}"
            )

    # 5. Marbled vs uniform — variation is monotone in noise ----------------

    def test_marbled_paper_has_higher_tone_variation_than_uniform_paper(
        self,
    ) -> None:
        # Synthetic input is required here: we need two pages with the
        # SAME approximate mean (~245), one perfectly flat, one with
        # low-amplitude Gaussian texture. Real corpus crops can't
        # guarantee equal means, so the variation signal would be
        # entangled with mean differences.

        # Arrange
        uniform_page: BgrImage = _solid_page(
            (MARBLED_MEAN, MARBLED_MEAN, MARBLED_MEAN)
        )
        marbled_page: BgrImage = _marbled_page()

        # Act
        _uniform_color, uniform_variation = estimate_paper_tone(uniform_page)
        _marbled_color, marbled_variation = estimate_paper_tone(marbled_page)

        # Assert — variation must be strictly, MEASURABLY higher on the
        # textured page. A trivial > would let two near-zero numbers pass
        # by floating-point noise; we require a gap of at least
        # VARIATION_SEPARATION_MIN ΔE so downstream extractors see a real
        # signal to widen their thresholds on.
        separation: float = marbled_variation - uniform_variation
        assert separation > VARIATION_SEPARATION_MIN, (
            f"marbled_variation={marbled_variation}, "
            f"uniform_variation={uniform_variation}, "
            f"separation={separation} (need > {VARIATION_SEPARATION_MIN})"
        )

    # 6. Substrate-agnostic smoke test --------------------------------------

    def test_handles_both_white_and_yellow_substrates(
        self, project_root: Path
    ) -> None:
        # Arrange — both real substrates back-to-back in one test, to
        # confirm the implementation is not silently hardcoded to one
        # paper color.
        white_page: BgrImage = _load_white_receipt_page(project_root)
        white_crop: BgrImage = _white_paper_crop(white_page)
        yellow_page: BgrImage = _load_yellow_invoice_page(project_root)
        yellow_crop: BgrImage = _yellow_paper_crop(yellow_page)

        # Act
        white_color, white_variation = estimate_paper_tone(white_crop)
        yellow_color, yellow_variation = estimate_paper_tone(yellow_crop)

        # Assert — both runs return the documented types. We don't pin
        # numeric envelopes here (the dedicated tests above do that);
        # this is a regression net for "the function still runs on both
        # substrates without exploding."
        assert isinstance(white_color, tuple) and len(white_color) == 3, (
            f"white_color must be a 3-tuple, got {white_color!r}"
        )
        assert isinstance(yellow_color, tuple) and len(yellow_color) == 3, (
            f"yellow_color must be a 3-tuple, got {yellow_color!r}"
        )
        for color in (white_color, yellow_color):
            for channel_value in color:
                assert isinstance(channel_value, int), (
                    f"channel must be int, got "
                    f"{type(channel_value).__name__} in {color!r}"
                )
                assert 0 <= channel_value <= 255, (
                    f"channel {channel_value} outside uint8 range "
                    f"in {color!r}"
                )
        for variation in (white_variation, yellow_variation):
            assert isinstance(variation, float), (
                f"tone_variation must be float, got "
                f"{type(variation).__name__}"
            )
            assert np.isfinite(variation), (
                f"tone_variation {variation} not finite"
            )
            assert variation >= 0.0, (
                f"tone_variation {variation} negative"
            )
