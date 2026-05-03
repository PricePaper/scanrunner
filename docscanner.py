#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "opencv-python-headless>=4.10",
#   "Pillow>=11.0",
#   "numpy>=2.0",
#   "pytesseract>=0.3.13",
#   "python-magic>=0.4.27",
#   "PyYAML>=6.0",
#   "watchdog>=5.0",
#   "httpx[http2]>=0.27",
#   "python-doctr>=1.0",
#   "torch>=2.4",
# ]
# ///
"""scanrunner v2.0 — clean-room rewrite.

A single-file daemon that watches an inbox directory for scanned invoices
(and, in the future, other document types), OCRs the document number, links
the original to its Odoo record, stores a cleaned-up reproduction-quality
attachment, and archives the source under a structured tree.

All production code lives in this file. Tests live under ``tests/``.
"""

import argparse
import base64
import hashlib
import io
import logging
import os
import re
import signal
import smtplib
import sqlite3
import shutil
import sys
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from email.message import EmailMessage
from fnmatch import fnmatch
from multiprocessing import get_context
from pathlib import Path
from typing import Any, ClassVar, Self, override

import cv2
import httpx
import magic
import numpy as np
import pytesseract  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]
from PIL import Image
from watchdog.events import FileClosedEvent, FileSystemEventHandler
from watchdog.observers import Observer

type RegionPct = tuple[int, int, int, int]
"""(x1%, y1%, x2%, y2%) — inclusive percentages of image width / height."""

type BgrImage = np.ndarray
"""OpenCV BGR uint8 image (H, W, 3)."""

type BgrColor = tuple[int, int, int]
"""A single BGR uint8 color triple (B, G, R), each channel in [0, 255]."""

type GrayImage = np.ndarray
"""Single-channel uint8 image (H, W)."""


# -----------------------------------------------------------------------------
# Image cleanup primitives — used by Invoice.preprocess (yellow paper) and by
# the storage pass for every document type (background snap, edge clean).
# -----------------------------------------------------------------------------


class YellowRemover:
    """Replace the yellow paper cast with near-white.

    HSV-mask based: pixels whose hue lands in the yellow band and whose
    saturation/value indicate "paper, not ink" get redirected to the local
    lightness mean (close to white once the cast is gone). Black text, blue
    or red ink, signatures, and check marks are left untouched.

    Used **only** by ``Invoice.preprocess``; other document types may not
    be on yellow paper and skip this step.
    """

    HUE_LOW: ClassVar[int] = 15
    HUE_HIGH: ClassVar[int] = 45
    SAT_MIN: ClassVar[int] = 100
    """Pure yellow paper has S≈140-180 on this scanner; partial-coverage
    edge mixes (where dark ink shows through paper) drop to S≈60-90.
    100 cleanly distinguishes paper from edge-mix pixels — the latter
    must NOT be classified as paper or pen-stroke anti-aliasing gets
    eaten and signatures arrive at the encoder as fragments."""

    VAL_MIN: ClassVar[int] = 170
    """Pure yellow paper has V≈210-230. Anti-aliased ink edges have
    V≈100-150. The old VAL_MIN=80 caught the edges (V≥80 trivially)
    and snapped them white — destroying ~74 % of pixels in [50, 200)
    on the inv/good/INV-2026-05020/0002 sample. 170 keeps confident
    paper in the mask, drops edge mixes."""

    def remove(self, bgr: BgrImage) -> BgrImage:
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("YellowRemover requires a BGR color image")
        hsv: np.ndarray = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h: np.ndarray
        s: np.ndarray
        v: np.ndarray
        h, s, v = cv2.split(hsv)
        # Confidently-yellow paper: push to pure white, killing the cast.
        # Pixels darker than VAL_MIN (likely text shadowing) or low-saturation
        # (already neutral) are left alone — BackgroundSnapper handles those.
        yellow_mask: np.ndarray = (
            (h >= self.HUE_LOW)
            & (h <= self.HUE_HIGH)
            & (s >= self.SAT_MIN)
            & (v >= self.VAL_MIN)
        )
        out: BgrImage = bgr.copy()
        out[yellow_mask] = (255, 255, 255)
        return out


class BackgroundFlattener:
    """Flatten illumination by dividing through a large-kernel background estimate.

    Removes scanner shadowing and uneven exposure without touching strokes.
    """

    KERNEL: ClassVar[int] = 51

    def flatten(self, img: BgrImage | GrayImage) -> BgrImage | GrayImage:
        if img.ndim == 2:
            return self._flatten_channel(img)
        channels: list[GrayImage] = [
            self._flatten_channel(img[:, :, c]) for c in range(img.shape[2])
        ]
        return np.stack(channels, axis=-1)

    def _flatten_channel(self, ch: GrayImage) -> GrayImage:
        kernel: np.ndarray = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.KERNEL, self.KERNEL)
        )
        background: np.ndarray = cv2.morphologyEx(ch, cv2.MORPH_CLOSE, kernel)
        background = cv2.medianBlur(background, self.KERNEL)
        background = np.where(background == 0, 1, background).astype(np.float32)
        norm: np.ndarray = (ch.astype(np.float32) / background) * 255.0
        return np.clip(norm, 0, 255).astype(np.uint8)


class BackgroundSnapper:
    """Force background pixels to pure white; preserve foreground tone.

    Builds a foreground mask from the flattened image (anything sufficiently
    darker than the local background = ink/signature/check-mark). Background
    pixels are snapped to 255 so JPEG/PNG encoders don't fan grey "scanner
    shadow" noise across the page when reprinting on white paper.

    Foreground pixels keep their grayscale value so signatures and check
    marks reproduce with proper stroke weight.
    """

    THRESHOLD_OFFSET: ClassVar[int] = 35
    """Pixels darker than (local_background - this) are foreground.

    35 keeps routine paper texture quiet AND preserves anti-aliased
    ink edges (which arrive at 100-150 V values, comfortably below
    local_mean - 35). An earlier "drop to 20" attempt was chasing a
    faint-ink-survival problem that turned out to be in YellowRemover;
    once that was fixed the snapper could go back to a clean 35
    without losing real ink, and the page-wide gray speckle that came
    with the 20 setting goes away.
    """

    def snap(self, img: BgrImage | GrayImage) -> tuple[GrayImage, GrayImage]:
        """Return (snapped grayscale image, foreground mask)."""
        gray: GrayImage = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Adaptive threshold: foreground = pixels noticeably darker than their
        # neighborhood mean. Block size large enough to span a stroke without
        # spanning whole columns of text.
        fg_mask: GrayImage = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=51,
            C=self.THRESHOLD_OFFSET,
        )
        snapped: GrayImage = gray.copy()
        snapped[fg_mask == 0] = 255
        return snapped, fg_mask


class InkRegionDetector:
    """Locates text/handwriting regions on a page via DocTR's pretrained
    detector.

    Used as a "permission slip" mask downstream: pixels OUTSIDE detected
    ink regions get force-snapped to white, which kills the marbled
    paper-texture speckle that the snapper + rescuer would otherwise
    produce on margin paper. Pixels INSIDE detected regions go through
    the existing precision pipeline unchanged.

    The DocTR model loads on first ``detect()`` call and is cached at
    the class level — across a worker's lifetime the model is loaded
    exactly once. Each ProcessPool worker pays the load once on its
    first file (~3-5 s); subsequent files in that worker are fast.
    """

    PADDING_PX: ClassVar[int] = 18
    """Pixels to pad each detected word bbox before stamping into the
    mask. DocTR returns tight boxes; padding captures stroke edges
    (anti-aliased halos that fall just outside the glyph) and adjacent
    short marks (commas, accents, the dot of a check-mark)."""

    _model: ClassVar[Any | None] = None

    @classmethod
    def _get_model(cls) -> Any:
        if cls._model is None:
            # Defer the heavy import so module load doesn't drag torch in.
            from doctr.models import detection_predictor
            cls._model = detection_predictor(pretrained=True)
        return cls._model

    def detect(self, bgr: BgrImage) -> np.ndarray:
        """Return a boolean mask of shape ``bgr.shape[:2]`` covering all
        detected text/ink regions plus ``PADDING_PX`` of slack on each side."""
        # DocTR expects RGB.
        rgb: np.ndarray = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        results: list[dict[str, np.ndarray]] = self._get_model()([rgb])
        h: int
        w: int
        h, w = bgr.shape[:2]
        mask: np.ndarray = np.zeros((h, w), dtype=np.uint8)
        words: np.ndarray = results[0].get("words", np.empty((0, 5)))
        pad: int = self.PADDING_PX
        for x_min, y_min, x_max, y_max, _conf in words:
            x1: int = max(0, int(x_min * w) - pad)
            y1: int = max(0, int(y_min * h) - pad)
            x2: int = min(w, int(x_max * w) + pad)
            y2: int = min(h, int(y_max * h) + pad)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
        return mask > 0


class FaintInkRescuer:
    """Iterative second-pass snap that rescues faint ink adjacent to detected strokes.

    The primary ``BackgroundSnapper`` may miss the lightest pixels of a
    pen-stroke halo. This class dilates the existing foreground mask
    (anywhere ink was already detected → the stroke "neighborhood"),
    re-runs adaptive threshold inside that neighborhood with a more
    permissive offset, and writes the rescued faint pixels'
    flattened-grayscale tone back into the snapped image. The pass
    repeats — the just-rescued faint pixels become anchors for the next
    iteration, letting a long chain of progressively-fainter ink grow
    outward in waves.

    Seed-area filter: only connected components ≥ MIN_SEED_AREA can
    anchor a rescue. Stops paper-shadow speckle (~1-5 px components)
    from snowballing into gray blotches over multiple iterations.
    Isolated faint regions with no detected anchor stay snapped to
    white — each iteration only grows from existing detected ink,
    never bridges across pure paper.
    """

    NEIGHBORHOOD_RADIUS: ClassVar[int] = 20
    """Pixels within this many of an existing foreground pixel are
    eligible for rescue per iteration."""

    RESCUE_THRESHOLD_OFFSET: ClassVar[int] = 8
    """Adaptive-threshold C used inside the neighborhood. Smaller than
    BackgroundSnapper.THRESHOLD_OFFSET — once we know we're near real
    ink, we can be more aggressive about catching its faint ends."""

    BLOCK_SIZE: ClassVar[int] = 51

    MAX_ITERATIONS: ClassVar[int] = 3
    """Hard cap on rescue passes. Three passes (~60 px reach) covers
    real signature stroke halos without letting rescue walk across the
    page into adjacent paper-shadow regions."""

    MIN_SEED_AREA: ClassVar[int] = 20
    """Connected components in the input fg_mask smaller than this
    cannot anchor a rescue. Real ink strokes (≥50 px) and check marks
    pass; sub-10-px speckle drops out."""

    def rescue(
        self,
        snapped: GrayImage,
        fg_mask: GrayImage,
        flat: BgrImage | GrayImage,
    ) -> tuple[GrayImage, GrayImage]:
        """Return (updated snapped, updated foreground mask)."""
        flat_gray: GrayImage = (
            flat if flat.ndim == 2 else cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
        )
        n_labels: int
        labels: np.ndarray
        stats: np.ndarray
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            fg_mask, connectivity=8
        )
        seed_mask: GrayImage = np.zeros_like(fg_mask)
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] >= self.MIN_SEED_AREA:
                seed_mask[labels == i] = 255
        kernel: np.ndarray = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.NEIGHBORHOOD_RADIUS * 2 + 1, self.NEIGHBORHOOD_RADIUS * 2 + 1),
        )
        permissive_mask: GrayImage = cv2.adaptiveThreshold(
            flat_gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=self.BLOCK_SIZE,
            C=self.RESCUE_THRESHOLD_OFFSET,
        )
        out: GrayImage = snapped.copy()
        current_mask: GrayImage = fg_mask.copy()
        iter_anchor: GrayImage = seed_mask.copy()
        for _ in range(self.MAX_ITERATIONS):
            neighborhood: GrayImage = cv2.dilate(iter_anchor, kernel)
            new_fg: np.ndarray = (
                (permissive_mask > 0)
                & (neighborhood > 0)
                & (current_mask == 0)
            )
            if not new_fg.any():
                break
            out[new_fg] = flat_gray[new_fg]
            new_fg_u8: GrayImage = new_fg.astype(np.uint8) * 255
            current_mask = current_mask | new_fg_u8
            iter_anchor = iter_anchor | new_fg_u8
        return out, current_mask


class EdgeCleaner:
    """Drop isolated foreground speckle (paper fiber, toner spray); keep strokes."""

    MIN_COMPONENT_AREA: ClassVar[int] = 8

    def clean(self, snapped: GrayImage, fg_mask: GrayImage) -> GrayImage:
        """Remove tiny foreground components and write the result back to ``snapped``."""
        n_labels: int
        labels: np.ndarray
        stats: np.ndarray
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            fg_mask, connectivity=8
        )
        out: GrayImage = snapped.copy()
        for label in range(1, n_labels):
            area: int = stats[label, cv2.CC_STAT_AREA]
            if area < self.MIN_COMPONENT_AREA:
                out[labels == label] = 255
        return out


class EdgeCrispener:
    """Snap text-edge anti-aliasing halos to pure white; keep signature bodies.

    A halo pixel is intermediate-tone AND adjacent (within a few pixels)
    to a strong-black region. A signature-body pixel is intermediate-tone
    AND surrounded by other intermediate-tone pixels. The first kind costs
    PNG bytes for cosmetic anti-aliasing the eye barely notices; the
    second kind is the actual document content we promised to preserve.
    """

    DARK_THRESHOLD: ClassVar[int] = 60        # pixels ≤ this count as "dark stroke"
    INTERMEDIATE_LOW: ClassVar[int] = 40
    INTERMEDIATE_HIGH: ClassVar[int] = 230
    HALO_DILATE_RADIUS: ClassVar[int] = 3     # halo extends this far from dark
    SIGNATURE_WINDOW: ClassVar[int] = 9       # local context window
    SIGNATURE_MIN_INTERMEDIATE_FRACTION: ClassVar[float] = 0.35

    def crispen(self, snapped: GrayImage) -> GrayImage:
        """Return ``snapped`` with halos snapped to 255, bodies untouched."""
        dark_mask: GrayImage = (snapped <= self.DARK_THRESHOLD).astype(np.uint8) * 255
        kernel: np.ndarray = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.HALO_DILATE_RADIUS * 2 + 1, self.HALO_DILATE_RADIUS * 2 + 1),
        )
        near_dark: GrayImage = cv2.dilate(dark_mask, kernel)
        intermediate_mask: GrayImage = (
            (snapped > self.INTERMEDIATE_LOW)
            & (snapped < self.INTERMEDIATE_HIGH)
        ).astype(np.uint8) * 255
        # Local fraction of intermediates in a window. A signature body
        # has a high fraction (its neighborhood is mostly itself); a halo
        # has a low fraction (its neighborhood is dark + white, with the
        # halo a thin ring).
        kernel_window: np.ndarray = np.ones(
            (self.SIGNATURE_WINDOW, self.SIGNATURE_WINDOW), dtype=np.float32
        ) / float(self.SIGNATURE_WINDOW * self.SIGNATURE_WINDOW)
        local_intermediate_frac: np.ndarray = cv2.filter2D(
            (intermediate_mask > 0).astype(np.float32), -1, kernel_window,
        )
        signature_body: np.ndarray = (
            local_intermediate_frac >= self.SIGNATURE_MIN_INTERMEDIATE_FRACTION
        )
        halo_mask: np.ndarray = (
            (intermediate_mask > 0)
            & (near_dark > 0)
            & ~signature_body
        )
        out: GrayImage = snapped.copy()
        out[halo_mask] = 255
        return out


class ForegroundQuantizer:
    """Collapse foreground intensities to N evenly-spaced bins.

    PNG deflate compresses far better when the foreground alphabet is
    small. 16 levels (default) keeps signature legibility while reducing
    storage; 8 risks visible banding on ink gradients.
    """

    DEFAULT_LEVELS: ClassVar[int] = 16
    BACKGROUND_THRESHOLD: ClassVar[int] = 240   # pixels ≥ this are paper

    def __init__(self, levels: int = DEFAULT_LEVELS) -> None:
        if not 2 <= levels <= 256:
            raise ValueError(f"levels must be in [2, 256], got {levels}")
        self._levels: int = levels
        # Pre-compute the bin centers (evenly spaced 0..255).
        step: float = 255.0 / (levels - 1)
        self._lut: np.ndarray = np.arange(256, dtype=np.float32)
        self._lut = (np.round(self._lut / step) * step).astype(np.uint8)

    def quantize(self, snapped: GrayImage) -> GrayImage:
        """Quantize foreground pixels; leave near-paper pixels exactly 255."""
        background: np.ndarray = snapped >= self.BACKGROUND_THRESHOLD
        out: GrayImage = cv2.LUT(snapped, self._lut)
        out[background] = 255
        return out


# -----------------------------------------------------------------------------
# Storage encoder — fixed calibrated settings (no runtime quality ladder).
# -----------------------------------------------------------------------------
#
# Calibrated 2026-05-01 against 168 samples from inv/good/:
#   * downsample factor 2/3 (≈300 dpi → 200 dpi for storage)
#   * primary: JPEG quality 85, grayscale  (halo ≤ 1.99, ~28% of files fit ≤300 KB)
#   * fallback: PNG grayscale, optimize=True, compress_level=9
#       (mean 317 KB, max 505 KB across the 168 samples)
#
# Format pick is deterministic per file: encode JPEG once; if it exceeds
# the size cap, encode PNG once. Same input → same output. No quality
# ladder, no iteration to hit the cap.


class Encoder:
    """JPEG-or-PNG encoder using the single calibrated setting per format.

    Per the user's spec: ship JPEG when ≤ ``SIZE_CAP_BYTES``, else fall back
    to PNG-grayscale. The format pick is the only branch — neither
    quality nor compression-level varies at runtime.
    """

    JPEG_QUALITY: ClassVar[int] = 85
    SIZE_CAP_BYTES: ClassVar[int] = 300_000

    def encode(self, snapped: GrayImage) -> tuple[bytes, str]:
        """Return (payload, mimetype) for the given storage-ready grayscale image."""
        jpeg: bytes = self._encode_jpeg(snapped)
        if len(jpeg) <= self.SIZE_CAP_BYTES:
            return jpeg, "image/jpeg"
        return self._encode_png(snapped), "image/png"

    def _encode_jpeg(self, gray: GrayImage) -> bytes:
        img: Image.Image = Image.fromarray(gray, mode="L")
        buf: io.BytesIO = io.BytesIO()
        img.save(
            buf,
            format="JPEG",
            quality=self.JPEG_QUALITY,
            optimize=True,
            progressive=True,
        )
        return buf.getvalue()

    def _encode_png(self, gray: GrayImage) -> bytes:
        img: Image.Image = Image.fromarray(gray, mode="L")
        buf: io.BytesIO = io.BytesIO()
        img.save(buf, format="PNG", optimize=True, compress_level=9)
        return buf.getvalue()


# --- estimate_paper_tone -----------------------------------------------------
# Substrate-population selection uses an adaptive luminance percentile rather
# than a fixed V threshold so the same code handles white, cream, and yellow
# papers without per-substrate tuning. The 60th-percentile cutoff keeps the
# brightest 40% of pixels: on a mostly-paper crop that is the paper, on a
# crop with ink the dark ink pixels fall below the cutoff and are excluded.
PAPER_SUBSTRATE_V_PERCENTILE: int = 60


def estimate_paper_tone(bgr: BgrImage) -> tuple[BgrColor, float]:
    """Estimate the dominant paper-substrate color and tone variation.

    Returns ``(paper_color, tone_variation)`` where ``paper_color`` is
    the median BGR of the substrate population (ink excluded) and
    ``tone_variation`` is the population's ΔE_lab spread (Euclidean
    distance in OpenCV Lab space, approximating ΔE76).
    """
    # Adaptive luminance cutoff: the brightest PAPER_SUBSTRATE_V_PERCENTILE-th
    # percentile defines "substrate" regardless of the paper's actual hue.
    hsv: np.ndarray = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    value_channel: np.ndarray = hsv[:, :, 2]
    value_threshold = float(
        np.percentile(value_channel, PAPER_SUBSTRATE_V_PERCENTILE)
    )
    substrate_mask: np.ndarray = value_channel >= value_threshold

    substrate_pixels: np.ndarray = bgr[substrate_mask]

    # Per-channel median over the substrate population — robust to the
    # remaining handful of darker pixels that may have squeaked above the
    # percentile floor (e.g., faint anti-aliasing fringes).
    median_bgr: np.ndarray = np.median(substrate_pixels, axis=0)
    paper_color: BgrColor = (
        int(median_bgr[0]),
        int(median_bgr[1]),
        int(median_bgr[2]),
    )

    # Lab ΔE spread: convert just the substrate pixels to Lab and take the
    # std-dev of each pixel's Euclidean distance to the population centroid.
    # OpenCV Lab packs L/a/b into uint8; we cast to float for the math.
    substrate_lab: np.ndarray = cv2.cvtColor(
        substrate_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB,
    ).reshape(-1, 3).astype(np.float32)
    lab_centroid: np.ndarray = substrate_lab.mean(axis=0)
    delta_e: np.ndarray = np.linalg.norm(substrate_lab - lab_centroid, axis=1)
    tone_variation = float(delta_e.std())

    return paper_color, tone_variation


@dataclass(frozen=True, slots=True)
class PrintedLayer:
    """v3 carrier: laser-printed content lifted off a scanned page.

    ``mask`` marks pixels that belong to printed strokes (text, frames,
    logos, barcodes); ``tones`` preserves the original grayscale at those
    pixels so the downstream composer can lay anti-aliased ink back onto
    a clean paper canvas without binarizing it into pure black-on-white.

    Attributes:
        mask: Boolean ndarray of shape ``(H, W)``. ``True`` where a pixel
            belongs to printed content, ``False`` elsewhere.
        tones: ``uint8`` ndarray of shape ``(H, W)``. Grayscale ink tones
            preserved from the original scan at masked positions; values
            outside the mask are unspecified but must be valid ``uint8``.
    """

    mask: np.ndarray
    tones: np.ndarray


class PrintedLayerExtractor:
    """Extract the printed-content layer from a scanned page.

    Given a BGR scan and a paper-tone baseline (``paper_color`` and
    ``paper_variation`` from :func:`estimate_paper_tone`), identify the
    pixels that belong to laser-printed content (text, frames, logos,
    barcodes) and distinguish them from handwriting (variable strokes,
    irregular geometry) and paper noise (near-paper-tone pixels).

    The returned :class:`PrintedLayer` carries a boolean ``mask`` of the
    detected printed strokes and a ``uint8`` ``tones`` map preserving the
    original grayscale at those pixels.

    Strategy: DocTR's pretrained detector is a near-zero-false-positive
    word-region locator that ignores cursive scribbles. We use it as the
    primary "is there printed text here?" gate, then intersect with a
    classical-CV "is this pixel substantially darker than paper?" test
    to convert padded word boxes into pixel-precise stroke masks.
    """

    DARKER_THAN_PAPER_BASE: ClassVar[int] = 60
    """Base grayscale gap (in uint8 units) below the paper's grayscale
    equivalent for a pixel to count as ink. Tuned to catch dark laser
    strokes while ignoring anti-aliased halo pixels at the stroke edge."""

    DARKER_THAN_PAPER_VAR_COEFF: ClassVar[float] = 2.0
    """Per-unit ``paper_variation`` widening of the threshold. Noisier
    substrates (yellow paper, ΔE ~1.9) need more headroom so substrate
    micro-variation does not cross the ink threshold."""

    MIN_TEXT_LINE_ASPECT_RATIO: ClassVar[float] = 1.5
    """Minimum width/height ratio of a DocTR-mask connected component to
    qualify as a printed-text strip. Padded word boxes from a single
    text line merge into a wide horizontal blob (aspect >> 1); isolated
    handwritten words form roughly square blobs (aspect ~ 1). This
    filter rejects DocTR's occasional false-positive hits on cursive
    text without sacrificing real printed lines."""

    MAX_STROKE_WIDTH_CV: ClassVar[float] = 0.25
    """Maximum coefficient of variation (std/mean) of stroke width within a
    connected component. Printed laser text has uniform strokes; pen ink has
    variable strokes due to pressure changes. Rejects horizontal-line
    handwriting that the aspect-ratio screen alone would let through."""

    MIN_STROKE_TO_LETTER_HEIGHT_RATIO: ClassVar[float] = 0.18
    """Minimum (2*median half-stroke-width) / (median letter-CC height) for
    a text-strip to count as printed. Laser fonts at our scan resolution
    sit around 0.20-0.32 (stroke is ~10-15% of cap height). Pen handwriting
    sits around 0.07-0.16 because pen marks make thin strokes relative to
    tall, often-connected letter shapes (ascender + body + descender all in
    one ink CC). Empirically discriminates handwriting strips that the
    plain stroke-width CV cannot: at 200 DPI the per-stroke uniformity of
    ballpoint ink rivals laser print, but the geometric ratio of stroke
    thickness to letter envelope still separates the two cleanly."""

    MIN_INK_CC_AREA_PX: ClassVar[int] = 30
    """Minimum area (pixels) for an ink connected component to participate
    in the stroke-width statistics. Smaller blobs (dust, single-pixel
    artifacts, accent marks) inflate the variance without being
    representative of stroke shape."""

    MIN_SPINE_PIXELS: ClassVar[int] = 10
    """Minimum number of spine pixels (samples drawn from the distance
    transform) needed to score a connected component. CCs below this are
    too small to estimate stroke uniformity from and are kept by default
    (small printed marks shouldn't be discarded just because they're small)."""

    MIN_SPINE_DISTANCE_FRACTION: ClassVar[float] = 0.5
    """Fraction of the per-CC max distance-transform value used to select
    spine pixels. ``dist >= 0.5 * max_dist`` is a cheap medial-axis proxy:
    pixels at least half as far from the boundary as the deepest interior
    pixel sit near the centerline of their stroke and their distance
    values approximate the local half-stroke-width."""

    def __init__(self, ink_detector: InkRegionDetector | None = None) -> None:
        """Build an extractor, optionally injecting a shared detector.

        The default constructs a fresh :class:`InkRegionDetector`; the
        underlying DocTR model is cached at the class level so repeated
        construction is cheap.
        """
        self._ink_detector: InkRegionDetector = (
            ink_detector or InkRegionDetector()
        )

    def extract(
        self,
        bgr: BgrImage,
        paper_color: BgrColor,
        paper_variation: float,
    ) -> PrintedLayer:
        """Extract the printed layer from ``bgr``.

        Args:
            bgr: BGR scan of the page (or a sub-crop). Shape ``(H, W, 3)``.
            paper_color: Median BGR color of the paper substrate, as
                returned by :func:`estimate_paper_tone`.
            paper_variation: ΔE_lab spread of the paper substrate
                population, as returned by :func:`estimate_paper_tone`.
                Used to calibrate the substrate-aware "darker than paper"
                threshold so the extractor handles white, cream, and
                yellow stocks without per-substrate tuning.

        Returns:
            A :class:`PrintedLayer` whose ``mask`` and ``tones`` arrays
            both have shape ``(H, W)`` matching the input. ``mask.dtype``
            is ``bool``; ``tones.dtype`` is ``uint8``.
        """
        # Primary signal: DocTR word-region detection. Boxes are already
        # padded by InkRegionDetector.PADDING_PX so stroke halos fall
        # inside the mask even though DocTR returns tight bboxes.
        doctr_mask: np.ndarray = self._ink_detector.detect(bgr)

        # First geometric screen: keep only wide-aspect DocTR blobs.
        # Cursive signatures DocTR occasionally fires on form roughly
        # square blobs and get culled here. Wide horizontal-handwriting
        # strips (e.g., a single line of "3FD Kraft 2 Back order")
        # survive this filter and need a second screen below.
        text_strip_mask: np.ndarray = self._keep_wide_text_strips(doctr_mask)

        # Tones source: original grayscale carries the actual ink darkness
        # at masked pixels; outside the mask the value is unspecified by
        # the contract but must remain valid uint8 (which grayscale is).
        tones: np.ndarray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Substrate-aware "darker than paper" screen. BT.601 weights
        # match cv2.COLOR_BGR2GRAY, which is the convention `tones` uses.
        paper_blue, paper_green, paper_red = paper_color
        paper_grayscale: int = int(round(
            0.114 * paper_blue
            + 0.587 * paper_green
            + 0.299 * paper_red
        ))
        threshold: float = (
            self.DARKER_THAN_PAPER_BASE
            + self.DARKER_THAN_PAPER_VAR_COEFF * paper_variation
        )
        ink_mask: np.ndarray = tones < (paper_grayscale - threshold)

        # Second geometric screen: stroke-width uniformity / geometry on
        # the surviving strips. Handwritten letters connect together into
        # tall ink CCs whose stroke thickness is small relative to letter
        # height; printed laser glyphs sit closer to a stroke-to-height
        # ratio of ~0.20+. This catches the horizontal-line handwriting
        # that fooled the aspect-ratio screen alone.
        uniform_stroke_strip_mask: np.ndarray = (
            self._keep_uniform_stroke_widths(ink_mask, text_strip_mask)
        )

        # Final mask: surviving text-strip geometry AND ink darkness.
        # Handwriting fails one of the geometry gates; blank paper fails
        # the ink gate.
        printed_mask: np.ndarray = uniform_stroke_strip_mask & ink_mask

        return PrintedLayer(mask=printed_mask, tones=tones)

    def _keep_wide_text_strips(self, doctr_mask: np.ndarray) -> np.ndarray:
        """Return ``doctr_mask`` with only the wide-aspect components kept.

        Padded DocTR boxes from a single line of printed words merge
        into a contiguous horizontal blob whose width is many times its
        height. Handwritten words DocTR mistakenly catches form roughly
        square blobs. Filtering by minimum aspect ratio keeps text lines
        and rejects stray handwriting hits.
        """
        n_labels: int
        labels: np.ndarray
        stats: np.ndarray
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            doctr_mask.astype(np.uint8), connectivity=8,
        )
        kept: np.ndarray = np.zeros_like(doctr_mask)
        # Label 0 is the background; iterate real components only.
        for label in range(1, n_labels):
            blob_width: int = int(stats[label, cv2.CC_STAT_WIDTH])
            blob_height: int = int(stats[label, cv2.CC_STAT_HEIGHT])
            aspect: float = blob_width / max(blob_height, 1)
            if aspect >= self.MIN_TEXT_LINE_ASPECT_RATIO:
                kept[labels == label] = True
        return kept

    def _keep_uniform_stroke_widths(
        self,
        ink_mask: np.ndarray,
        candidate_strip_mask: np.ndarray,
    ) -> np.ndarray:
        """Filter wide-aspect text strips by stroke-width geometry.

        For each surviving text-strip connected component, examines the
        ink connected components inside that strip and computes
        per-letter stroke-width statistics from the distance transform.
        A strip is kept only if its median stroke width (full width =
        ``2 * median half-stroke``) is at least
        :attr:`MIN_STROKE_TO_LETTER_HEIGHT_RATIO` of the median letter
        height — the geometric signature of laser-printed glyphs at our
        scan resolution. Pen handwriting fails this ratio because its
        strokes are thin relative to the connected letter shapes.

        Args:
            ink_mask: Boolean ndarray of "darker than paper" pixels for
                the whole input frame.
            candidate_strip_mask: Boolean ndarray of pixels belonging to
                wide-aspect text-strip connected components (output of
                :meth:`_keep_wide_text_strips`).

        Returns:
            Boolean ndarray of the same shape as the inputs, restricted
            to the surviving strips.
        """
        # Connected-component pass over the strip mask so every ink-CC
        # is scored in the context of one DocTR-bounded text line.
        n_strips: int
        strip_labels: np.ndarray
        strip_stats: np.ndarray
        n_strips, strip_labels, strip_stats, _ = (
            cv2.connectedComponentsWithStats(
                candidate_strip_mask.astype(np.uint8), connectivity=8,
            )
        )
        kept: np.ndarray = np.zeros_like(candidate_strip_mask)
        # Label 0 is the background; iterate real strip components only.
        for strip_id in range(1, n_strips):
            strip_mask: np.ndarray = (strip_labels == strip_id)
            if self._strip_has_uniform_strokes(
                strip_mask, ink_mask, strip_stats[strip_id],
            ):
                kept[strip_mask] = True
        return kept

    def _strip_has_uniform_strokes(
        self,
        strip_mask: np.ndarray,
        ink_mask: np.ndarray,
        strip_stats: np.ndarray,
    ) -> bool:
        """Return True if ``strip_mask`` looks like printed text by stroke geometry.

        Computes per-ink-CC stroke widths via the distance transform
        (medial-axis proxy: pixels with ``dist >= MIN_SPINE_DISTANCE_FRACTION
        * max_dist`` approximate the centerline). Aggregates across the
        ink CCs in the strip and compares median stroke width to median
        letter-CC height: printed text sits at a higher ratio than
        handwriting at our scan resolution.

        Strips with too few ink pixels to score are kept by default —
        small marks should not be culled simply for being small.
        """
        x: int = int(strip_stats[cv2.CC_STAT_LEFT])
        y: int = int(strip_stats[cv2.CC_STAT_TOP])
        w: int = int(strip_stats[cv2.CC_STAT_WIDTH])
        h: int = int(strip_stats[cv2.CC_STAT_HEIGHT])

        # Crop the ink mask to this strip's bbox AND restrict to ink that
        # falls inside the strip (avoids leaking ink from neighbouring
        # strips that happen to overlap the bbox).
        strip_ink_full: np.ndarray = (ink_mask & strip_mask)
        strip_ink: np.ndarray = strip_ink_full[y:y + h, x:x + w].astype(
            np.uint8
        )
        if int(strip_ink.sum()) < self.MIN_SPINE_PIXELS:
            # Too few ink pixels to form a meaningful stroke statistic;
            # default to keep so small printed marks survive.
            return True

        # Connected components of the ink itself — one per glyph (or
        # cursive group). Each CC is scored independently so that a strip
        # with one fat blob and one thin one is summarised by its median.
        n_ink: int
        ink_labels: np.ndarray
        ink_stats: np.ndarray
        n_ink, ink_labels, ink_stats, _ = cv2.connectedComponentsWithStats(
            strip_ink, connectivity=8,
        )
        per_cc_half_widths: list[float] = []
        per_cc_heights: list[int] = []
        for ink_id in range(1, n_ink):
            half_width, letter_height = self._score_ink_component(
                ink_labels, ink_stats, ink_id,
            )
            if half_width is None:
                continue
            per_cc_half_widths.append(half_width)
            per_cc_heights.append(letter_height)

        if not per_cc_half_widths:
            # No CC was large enough to score; default to keep.
            return True

        median_half_width: float = float(np.median(per_cc_half_widths))
        median_letter_height: float = float(np.median(per_cc_heights))
        if median_letter_height <= 0:
            return True
        # Full stroke width is 2x the half-stroke (distance-transform value).
        stroke_to_height_ratio: float = (
            2.0 * median_half_width
        ) / median_letter_height
        return stroke_to_height_ratio >= self.MIN_STROKE_TO_LETTER_HEIGHT_RATIO

    def _score_ink_component(
        self,
        ink_labels: np.ndarray,
        ink_stats: np.ndarray,
        ink_id: int,
    ) -> tuple[float | None, int]:
        """Compute (median half-stroke-width, bbox-height) for one ink CC.

        Returns ``(None, height)`` when the CC is too small to score
        (insufficient area or too few spine samples). The caller skips
        these but keeps the strip if NO CC scored — small printed marks
        should not be discarded just for being small.
        """
        area: int = int(ink_stats[ink_id, cv2.CC_STAT_AREA])
        cc_height: int = int(ink_stats[ink_id, cv2.CC_STAT_HEIGHT])
        if area < self.MIN_INK_CC_AREA_PX:
            return None, cc_height

        ix: int = int(ink_stats[ink_id, cv2.CC_STAT_LEFT])
        iy: int = int(ink_stats[ink_id, cv2.CC_STAT_TOP])
        iw: int = int(ink_stats[ink_id, cv2.CC_STAT_WIDTH])
        ih: int = int(ink_stats[ink_id, cv2.CC_STAT_HEIGHT])
        # Isolate this single CC inside its own bbox so the distance
        # transform sees only its own boundary (neighbour CCs would
        # otherwise change distances along the CC's edges).
        ink_cc: np.ndarray = (
            ink_labels[iy:iy + ih, ix:ix + iw] == ink_id
        ).astype(np.uint8)
        dist: np.ndarray = cv2.distanceTransform(ink_cc, cv2.DIST_L2, 5)
        ink_distances: np.ndarray = dist[ink_cc.astype(bool)]
        max_distance: float = float(ink_distances.max())
        if max_distance < 1.0:
            # Single-pixel-thick CC; the spine is essentially undefined.
            return None, cc_height
        spine_distances: np.ndarray = ink_distances[
            ink_distances >= self.MIN_SPINE_DISTANCE_FRACTION * max_distance
        ]
        if spine_distances.size < self.MIN_SPINE_PIXELS // 2:
            # Spine too sparse to estimate stroke width reliably.
            return None, cc_height
        return float(spine_distances.mean()), cc_height


class StoragePreparer:
    """Compose the storage pipeline: downsample → flatten → snap → clean → encode.

    Takes a (yellow-removed if applicable) BGR image and returns the bytes
    to upload to Odoo, plus the resulting MIME type.
    """

    SOURCE_DPI: ClassVar[int] = 300
    TARGET_DPI: ClassVar[int] = 200
    DOWNSAMPLE: ClassVar[float] = TARGET_DPI / SOURCE_DPI  # 2/3

    def __init__(
        self,
        flattener: BackgroundFlattener | None = None,
        snapper: BackgroundSnapper | None = None,
        rescuer: FaintInkRescuer | None = None,
        edge_cleaner: EdgeCleaner | None = None,
        edge_crispener: EdgeCrispener | None = None,
        quantizer: ForegroundQuantizer | None = None,
        ink_detector: InkRegionDetector | None = None,
        encoder: Encoder | None = None,
    ) -> None:
        self._flattener = flattener or BackgroundFlattener()
        self._snapper = snapper or BackgroundSnapper()
        self._rescuer = rescuer or FaintInkRescuer()
        self._edge_cleaner = edge_cleaner or EdgeCleaner()
        self._edge_crispener = edge_crispener or EdgeCrispener()
        self._quantizer = quantizer or ForegroundQuantizer()
        self._ink_detector = ink_detector or InkRegionDetector()
        self._encoder = encoder or Encoder()

    def prepare(self, cleaned: BgrImage | GrayImage) -> tuple[bytes, str]:
        # DocTR ink-region detection runs on the full-resolution cleaned
        # image (the detector benefits from the extra detail). The
        # resulting mask is downsampled with the image so it stays
        # aligned with the snapped output.
        if cleaned.ndim == 2:
            ink_input: BgrImage = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        else:
            ink_input = cleaned
        ink_mask_full: np.ndarray = self._ink_detector.detect(ink_input)
        new_w: int = int(cleaned.shape[1] * self.DOWNSAMPLE)
        new_h: int = int(cleaned.shape[0] * self.DOWNSAMPLE)
        scaled: BgrImage | GrayImage = cv2.resize(
            cleaned, (new_w, new_h), interpolation=cv2.INTER_AREA
        )
        ink_mask: np.ndarray = cv2.resize(
            ink_mask_full.astype(np.uint8), (new_w, new_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        flat: BgrImage | GrayImage = self._flattener.flatten(scaled)
        snapped: GrayImage
        fg_mask: GrayImage
        snapped, fg_mask = self._snapper.snap(flat)
        snapped, fg_mask = self._rescuer.rescue(snapped, fg_mask, flat)
        snapped = self._edge_cleaner.clean(snapped, fg_mask)
        # Final precision filter: anything outside the detected ink
        # regions is paper noise. Force-snap to white.
        snapped[~ink_mask] = 255
        # Compaction primitives (EdgeCrispener, ForegroundQuantizer) are
        # implemented but NOT wired into the production chain. Both were
        # tried and both damaged faint handwriting / printed text in
        # ways the office found unacceptable. They remain available as
        # building blocks if a future change wants them under different
        # parameters.
        return self._encoder.encode(snapped)


# -----------------------------------------------------------------------------
# Remaining classes (skeletons — implemented in subsequent TDD cycles).
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Region:
    """Percent-based crop window. ``(x1, y1, x2, y2)`` in 0..100."""

    x1_pct: int
    y1_pct: int
    x2_pct: int
    y2_pct: int

    def __post_init__(self) -> None:
        for v in (self.x1_pct, self.y1_pct, self.x2_pct, self.y2_pct):
            if not 0 <= v <= 100:
                raise ValueError(f"region percent out of range: {v}")
        if self.x2_pct <= self.x1_pct or self.y2_pct <= self.y1_pct:
            raise ValueError(f"region has zero or negative area: {self}")

    def crop(self, img: np.ndarray) -> np.ndarray:
        h: int
        w: int
        h, w = img.shape[:2]
        x1: int = w * self.x1_pct // 100
        y1: int = h * self.y1_pct // 100
        x2: int = w * self.x2_pct // 100
        y2: int = h * self.y2_pct // 100
        return img[y1:y2, x1:x2]

    @classmethod
    def from_list(cls, lst: list[int]) -> Self:
        return cls(*lst)


# -----------------------------------------------------------------------------
# Configuration loader.
# -----------------------------------------------------------------------------


class ConfigError(ValueError):
    """Raised when a required config key is missing or malformed."""


@dataclass(frozen=True, slots=True)
class ServerProfile:
    url: str
    database: str
    username: str
    password: str
    smtp_server: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    smtp_use_tls: bool
    verify_tls: bool = True


@dataclass(frozen=True, slots=True)
class DocumentTypeConfig:
    """YAML view of a single document-type entry."""

    name: str
    file_name_match: str
    mime_types: tuple[str, ...]
    ocr_regex: str
    search_regions: tuple[Region, ...]
    tesseract_config: str
    odoo_sequence: str
    odoo_object: str
    odoo_attachment_tag_id: int
    odoo_folder_id: int
    # Optional cascade fallbacks. Defaults are no-op (matches today's
    # single-attempt behavior). The cascade fires only if the primary OCR
    # misses, so the fast path is unaffected.
    fallback_search_regions: tuple[tuple[Region, ...], ...] = ()
    fallback_tesseract_configs: tuple[str, ...] = ()
    ocr_try_rotation: bool = False


class Config:
    """Loads YAML, picks the active server profile, exposes typed accessors."""

    def __init__(self, raw: dict[str, Any], server_name: str) -> None:
        self._raw: dict[str, Any] = raw
        servers: dict[str, dict[str, Any]] = raw.get("servers", {})
        if server_name not in servers:
            raise ConfigError(
                f"server '{server_name}' not in config (available: {list(servers)})"
            )
        s: dict[str, Any] = servers[server_name]
        try:
            self.server = ServerProfile(
                url=s["url"],
                database=s["database"],
                username=s["username"],
                password=s["password"],
                smtp_server=s["smtp-server"],
                smtp_port=int(s["smtp-port"]),
                smtp_user=s["smtp-user"],
                smtp_password=s["smtp-password"],
                smtp_use_tls=bool(s.get("smtp-use-tls", True)),
                verify_tls=bool(s.get("verify-tls", True)),
            )
        except KeyError as e:
            raise ConfigError(f"server '{server_name}' missing key {e}") from None
        self.server_name: str = server_name
        self.retry: int = int(raw.get("retry", 3))
        self.retry_sleep: float = float(raw.get("retry_sleep", 1.0))
        self.tesseract_bin: str = raw.get("tesseract-bin", "/usr/bin/tesseract")
        self.done_path: str = raw.get("done-path", "done")
        self.error_email: str = raw.get("error-email", "")
        self.error_mail_message: str = raw.get("error-mail-message", "")
        self.statistics_file: str = raw.get("statistics-file", "statistics.yaml")
        # Concurrency knobs. Defaults are tuned for steady-state scanning
        # (≈ 1 file every 5 s, rarely overlapping) on a small box. Two
        # workers cover the occasional scanner burst-of-two; each gets
        # two cv2 threads so a lone in-flight file uses 2 of the 4
        # production cores instead of 1. Bump both if your inbox is
        # genuinely bursty.
        self.workers: int = int(raw.get("workers", 2))
        self.cv2_threads: int = int(raw.get("cv2-threads", 2))
        self.documents: dict[str, DocumentTypeConfig] = {}
        for name, doc in (raw.get("documents") or {}).items():
            try:
                self.documents[name] = DocumentTypeConfig(
                    name=name,
                    file_name_match=doc["file-name-match"],
                    mime_types=tuple(doc["mime-types"]),
                    ocr_regex=doc["ocr_regex"],
                    search_regions=tuple(
                        Region.from_list(r) for r in doc["search_regions"]
                    ),
                    tesseract_config=doc.get("tesseract_config", "--psm 6 -l eng"),
                    odoo_sequence=doc["odoo_sequence"],
                    odoo_object=doc["odoo_object"],
                    odoo_attachment_tag_id=int(doc["odoo_attachment_tag_id"]),
                    odoo_folder_id=int(doc["odoo_folder_id"]),
                    fallback_search_regions=tuple(
                        tuple(Region.from_list(r) for r in region_set)
                        for region_set in doc.get("fallback_search_regions", [])
                    ),
                    fallback_tesseract_configs=tuple(
                        doc.get("fallback_tesseract_configs", [])
                    ),
                    ocr_try_rotation=bool(doc.get("ocr_try_rotation", False)),
                )
            except KeyError as e:
                raise ConfigError(
                    f"documents.{name} missing key {e}"
                ) from None

    @classmethod
    def load(cls, path: Path | str, server_name: str) -> Self:
        with open(path, encoding="utf-8") as f:
            raw: Any = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ConfigError(f"top-level YAML in {path} must be a mapping")
        return cls(cls._substitute_env(raw), server_name)

    _ENV_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")

    @classmethod
    def _substitute_env(cls, value: Any) -> Any:
        """Recursively replace ``${VAR}`` in string values with env lookups.

        Missing env vars become empty strings, matching shell semantics. Lets
        the YAML stay committable while secrets live in env / a runtime
        secrets manager.
        """
        match value:
            case str():
                return cls._ENV_PATTERN.sub(
                    lambda m: os.environ.get(m.group(1), ""), value
                )
            case dict():
                return {k: cls._substitute_env(v) for k, v in value.items()}
            case list():
                return [cls._substitute_env(v) for v in value]
            case _:
                return value


# -----------------------------------------------------------------------------
# OCR pipeline — preparer (in-memory binary), engine (Tesseract per region).
# -----------------------------------------------------------------------------


class OcrPreparer:
    """Produce the throw-away binary image used for OCR.

    Intentionally aggressive: the goal is glyph legibility, not preservation.
    Used by ``OcrEngine``; its output is never written to disk.
    """

    BLOCK_SIZE: ClassVar[int] = 31
    C: ClassVar[int] = 12

    def binarize(self, img: BgrImage | GrayImage) -> GrayImage:
        gray: GrayImage = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe: cv2.CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.fastNlMeansDenoising(gray, h=10)
        binary: GrayImage = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=self.BLOCK_SIZE,
            C=self.C,
        )
        return cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1
        )


@dataclass(frozen=True, slots=True)
class OcrMatch:
    """Result of a successful OCR extraction."""

    name: str
    region: Region


@dataclass(frozen=True, slots=True)
class OcrResult:
    """Cascade outcome.

    ``rotation_degrees`` is the angle (0 / 90 / 180 / 270) the cascade
    applied to the image to produce ``match``. Downstream consumers
    re-apply the same rotation to their input so the storage / archive
    representation comes out right-side-up.
    """

    match: OcrMatch
    rotation_degrees: int


class OcrEngine:
    """Run Tesseract on each region in priority order; first regex hit wins.

    The two-region rule lives here. The caller (the document type) picks
    the regions; this engine just tries them in order. ``SO…`` numbers are
    naturally rejected by an ``R?INV``-anchored regex.

    OCR output is normalized before matching: whitespace and common stray
    punctuation are stripped, so "INV,/2026 /05000" still matches.

    For documents that may arrive sideways or upside down or that need
    fallback regions/PSMs, use ``extract_with_fallbacks`` — it cascades
    through tier-by-tier and reports the rotation that worked.
    """

    UPSCALE_FACTOR: ClassVar[int] = 2
    NOISE_CHARS: ClassVar[re.Pattern[str]] = re.compile(r"[\s,.;:_|\\]")
    ROTATION_ANGLES: ClassVar[tuple[int, ...]] = (90, 180, 270)

    def __init__(self, tesseract_bin: str = "/usr/bin/tesseract") -> None:
        pytesseract.pytesseract.tesseract_cmd = tesseract_bin

    def extract(
        self,
        binary: GrayImage,
        regions: tuple[Region, ...],
        regex: re.Pattern[str],
        tesseract_config: str,
    ) -> OcrMatch | None:
        for region in regions:
            crop: GrayImage = region.crop(binary)
            if crop.size == 0:
                continue
            # Upscale small crops to give Tesseract more glyph detail.
            if min(crop.shape[:2]) < 200:
                crop = cv2.resize(
                    crop,
                    None,
                    fx=self.UPSCALE_FACTOR,
                    fy=self.UPSCALE_FACTOR,
                    interpolation=cv2.INTER_CUBIC,
                )
            text: str = pytesseract.image_to_string(crop, config=tesseract_config)
            normalized: str = self.NOISE_CHARS.sub("", text)
            match: re.Match[str] | None = regex.search(normalized)
            if match:
                return OcrMatch(name=match.group(0), region=region)
        return None

    def extract_with_fallbacks(
        self,
        binary: GrayImage,
        primary_regions: tuple[Region, ...],
        primary_config: str,
        fallback_regions: tuple[tuple[Region, ...], ...],
        fallback_configs: tuple[str, ...],
        try_rotation: bool,
        regex: re.Pattern[str],
    ) -> OcrResult | None:
        """Cascade until a regex hit lands, or return None.

        Tier 1: primary regions × primary config (the fast path).
        Tier 2: each fallback region set × primary config.
        Tier 3: each fallback PSM/config × primary regions.
        Tier 4: rotate 90/180/270 and re-run primary regions × primary config.

        First hit wins; later tiers are not consulted. Worst case for a
        hard miss with the Invoice config is ~7 Tesseract calls.
        """
        # Tier 1: primary
        match: OcrMatch | None = self.extract(
            binary, primary_regions, regex, primary_config
        )
        if match is not None:
            return OcrResult(match=match, rotation_degrees=0)

        # Tier 2: fallback regions
        for region_set in fallback_regions:
            match = self.extract(binary, region_set, regex, primary_config)
            if match is not None:
                return OcrResult(match=match, rotation_degrees=0)

        # Tier 3: fallback configs (e.g., PSM 11/12)
        for config in fallback_configs:
            match = self.extract(binary, primary_regions, regex, config)
            if match is not None:
                return OcrResult(match=match, rotation_degrees=0)

        # Tier 4: rotate the binary in 90° increments and retry primary.
        if try_rotation:
            for angle in self.ROTATION_ANGLES:
                rotated: GrayImage = np.rot90(binary, k=angle // 90)
                match = self.extract(
                    rotated, primary_regions, regex, primary_config
                )
                if match is not None:
                    return OcrResult(match=match, rotation_degrees=angle)

        return None


# -----------------------------------------------------------------------------
# Document type framework — Invoice today; Receipt etc. drop in tomorrow.
# -----------------------------------------------------------------------------


class DocumentType(ABC):
    """Per-type contract. Each subclass owns its preprocessing + OCR config."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def config(self) -> DocumentTypeConfig: ...

    @property
    @abstractmethod
    def regex(self) -> re.Pattern[str]: ...

    def preprocess(self, bgr: BgrImage) -> BgrImage:
        """Substrate-specific cleanup. Default = identity (no cleanup)."""
        return bgr

    def matches(self, path: Path, mime_type: str) -> bool:
        return (
            fnmatch(path.name, self.config.file_name_match)
            and mime_type in self.config.mime_types
        )


class Invoice(DocumentType):
    """Invoice on yellow paper.

    Yellow removal is part of THIS type's preprocessing — other future types
    on white or thermal paper skip it.
    """

    def __init__(
        self,
        config: DocumentTypeConfig,
        yellow_remover: YellowRemover | None = None,
    ) -> None:
        self._config: DocumentTypeConfig = config
        self._regex: re.Pattern[str] = re.compile(config.ocr_regex)
        self._yellow_remover: YellowRemover = yellow_remover or YellowRemover()

    @property
    @override
    def name(self) -> str:
        return self._config.name

    @property
    @override
    def config(self) -> DocumentTypeConfig:
        return self._config

    @property
    @override
    def regex(self) -> re.Pattern[str]:
        return self._regex

    @override
    def preprocess(self, bgr: BgrImage) -> BgrImage:
        return self._yellow_remover.remove(bgr)


class DocumentTypeRegistry:
    """Dispatch: filename + mime → DocumentType.

    The registry is built from a ``Config`` and resolves at runtime by walking
    its document types in declaration order. New types subclass
    ``DocumentType`` and register themselves here.
    """

    def __init__(self, types: list[DocumentType]) -> None:
        self._types: list[DocumentType] = types

    @classmethod
    def from_config(cls, config: Config) -> Self:
        types: list[DocumentType] = []
        for name, doc_config in config.documents.items():
            match name:
                case "Invoice":
                    types.append(Invoice(doc_config))
                case other:
                    raise ConfigError(
                        f"unknown document type '{other}' — extend "
                        "DocumentTypeRegistry to handle it"
                    )
        return cls(types)

    def classify(self, path: Path) -> DocumentType | None:
        mime_type: str = magic.from_file(str(path), mime=True)
        for dt in self._types:
            if dt.matches(path, mime_type):
                return dt
        return None


# -----------------------------------------------------------------------------
# Odoo client — httpx + JSON-RPC over (optionally) HTTPS. Never xmlrpc.
# -----------------------------------------------------------------------------


class OdooError(RuntimeError):
    """Server-side error returned by Odoo's JSON-RPC."""


class OdooClient:
    """JSON-RPC over HTTP(S) using httpx.

    Holds a single keep-alive ``httpx.Client``. Authenticates once on the
    first call and caches the resulting uid. If the client is closed at
    any point (because we hit a transient network error and recovered),
    the next call transparently rebuilds it — see ``_client``.
    """

    AUTH_TIMEOUT: ClassVar[float] = 30.0
    CALL_TIMEOUT: ClassVar[float] = 60.0

    def __init__(
        self,
        url: str,
        database: str,
        username: str,
        password: str,
        *,
        verify_tls: bool = True,
        retry: int = 3,
        retry_sleep: float = 1.0,
    ) -> None:
        self._url: str = url.rstrip("/")
        self._database: str = database
        self._username: str = username
        self._password: str = password
        self._verify_tls: bool = verify_tls
        self._retry: int = retry
        self._retry_sleep: float = retry_sleep
        self._uid: int | None = None
        self._http: httpx.Client | None = None

    @property
    def url(self) -> str:
        return self._url

    @property
    def database(self) -> str:
        return self._database

    def _client(self) -> httpx.Client:
        """Return a live httpx.Client, recreating on demand."""
        if self._http is None or self._http.is_closed:
            self._http = httpx.Client(
                base_url=self._url,
                timeout=self.CALL_TIMEOUT,
                verify=self._verify_tls,
                http2=self._url.startswith("https"),
                headers={"Content-Type": "application/json"},
            )
        return self._http

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None
            self._uid = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def authenticate(self) -> int:
        """Authenticate against ``service=common`` and cache the uid."""
        if self._uid is not None:
            return self._uid
        result: Any = self._jsonrpc(
            service="common",
            method="authenticate",
            args=[self._database, self._username, self._password, {}],
            timeout=self.AUTH_TIMEOUT,
        )
        if not result:
            raise OdooError(
                f"authentication failed for {self._username!r} on {self._database!r}"
            )
        self._uid = int(result)
        return self._uid

    def execute_kw(
        self,
        model: str,
        method: str,
        args: list[Any],
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        uid: int = self.authenticate()
        return self._jsonrpc(
            service="object",
            method="execute_kw",
            args=[
                self._database,
                uid,
                self._password,
                model,
                method,
                args,
                kwargs or {},
            ],
        )

    def search_read(
        self,
        model: str,
        domain: list[Any],
        fields: list[str],
        *,
        limit: int = 0,
    ) -> list[dict[str, Any]]:
        return self.execute_kw(
            model,
            "search_read",
            [domain, fields],
            {"limit": limit} if limit else {},
        )

    def find_record(self, model: str, name: str) -> int | None:
        rows: list[dict[str, Any]] = self.search_read(
            model, [["name", "=", name]], ["id"], limit=1
        )
        return int(rows[0]["id"]) if rows else None

    def attach(
        self,
        record_model: str,
        record_id: int,
        filename: str,
        mimetype: str,
        payload: bytes,
    ) -> int:
        b64: str = base64.b64encode(payload).decode("ascii")
        attachment_id: int = self.execute_kw(
            "ir.attachment",
            "create",
            [
                {
                    "name": filename,
                    "datas": b64,
                    "res_model": record_model,
                    "res_id": record_id,
                    "mimetype": mimetype,
                    "type": "binary",
                }
            ],
        )
        return int(attachment_id)

    def link_to_documents_app(
        self,
        attachment_id: int,
        folder_id: int,
        tag_id: int | None,
    ) -> int:
        vals: dict[str, Any] = {
            "attachment_id": attachment_id,
            "folder_id": folder_id,
        }
        if tag_id:
            vals["tag_ids"] = [(6, 0, [tag_id])]
        document_id: int = self.execute_kw(
            "documents.document",
            "create",
            [vals],
        )
        return int(document_id)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _jsonrpc(
        self,
        *,
        service: str,
        method: str,
        args: list[Any],
        timeout: float | None = None,
    ) -> Any:
        body: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": "call",
            "params": {"service": service, "method": method, "args": args},
        }
        last: Exception | None = None
        for attempt in range(self._retry + 1):
            try:
                client: httpx.Client = self._client()
                response: httpx.Response = client.post(
                    "/jsonrpc",
                    json=body,
                    timeout=timeout or self.CALL_TIMEOUT,
                )
                response.raise_for_status()
                payload: dict[str, Any] = response.json()
                if "error" in payload:
                    err: dict[str, Any] = payload["error"]
                    raise OdooError(
                        f"{err.get('message', 'odoo error')}: "
                        f"{err.get('data', {}).get('message', '')}"
                    )
                return payload.get("result")
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                last = e
                # Drop the (possibly half-broken) client; next attempt rebuilds.
                if self._http is not None:
                    self._http.close()
                    self._http = None
                if attempt < self._retry:
                    time.sleep(self._retry_sleep * (attempt + 1))
                    continue
        assert last is not None
        raise OdooError(f"transport error after {self._retry + 1} attempts: {last}") from last


# -----------------------------------------------------------------------------
# Archiver — moves the encoded storage payload into the done/ tree.
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParsedName:
    """Components of an extracted invoice name like ``INV/2026/05000``."""

    prefix: str  # INV | RINV | …
    year: int
    number: int

    @classmethod
    def parse(cls, name: str) -> Self:
        m: re.Match[str] | None = re.match(
            r"^(R?INV|REC)/(20\d{2})/(\d{4,5})$", name
        )
        if not m:
            raise ValueError(f"unrecognized document name: {name!r}")
        return cls(prefix=m.group(1), year=int(m.group(2)), number=int(m.group(3)))

    @property
    def number_range(self) -> int:
        return (self.number // 100) * 100


class Archiver:
    """File-system archive for successfully processed documents."""

    def __init__(self, root: Path, *, group_size: int = 100) -> None:
        self._root: Path = Path(root)
        self._group_size: int = group_size

    def archive(
        self,
        parsed: ParsedName,
        odoo_id: int,
        attachment_id: int,
        original_filename: str,
        payload: bytes,
        ext: str,
    ) -> Path:
        dest_dir: Path = (
            self._root / parsed.prefix / str(parsed.year) / f"{parsed.number_range:05d}"
        )
        dest_dir.mkdir(parents=True, exist_ok=True)
        stem: str = Path(original_filename).stem
        dest_name: str = (
            f"{parsed.prefix}-{parsed.year}-{parsed.number:05d}"
            f"_id-{odoo_id}_aid-{attachment_id}_{stem}.{ext.lstrip('.')}"
        )
        dest_path: Path = dest_dir / dest_name
        # Atomic write: write to a sibling tmp file and rename.
        tmp_path: Path = dest_path.with_name(f".{dest_path.name}.partial")
        tmp_path.write_bytes(payload)
        os.replace(tmp_path, dest_path)
        return dest_path

    def archive_unreadable(
        self,
        src_path: Path,
        payload: bytes | None = None,
        payload_ext: str = "",
    ) -> Path:
        """Route a failed-to-process file into the unreadable folder.

        With ``payload`` provided (cleaned/encoded bytes), write those to
        ``done/unreadable/<stem>.<payload_ext>`` and unlink the source so
        the watcher doesn't re-pick-up the file. Without payload, move
        the source into the folder unchanged.
        """
        dest_dir: Path = self._root / "unreadable"
        dest_dir.mkdir(parents=True, exist_ok=True)
        if payload is not None:
            dest_name: str = f"{src_path.stem}.{payload_ext.lstrip('.')}"
            dest_path: Path = dest_dir / dest_name
            dest_path.write_bytes(payload)
            try:
                src_path.unlink()
            except FileNotFoundError:
                pass
        else:
            dest_path = dest_dir / src_path.name
            # shutil.move handles cross-filesystem (the inbox bind mount
            # may be on a different fs than the done/ tree).
            shutil.move(str(src_path), str(dest_path))
        return dest_path


# -----------------------------------------------------------------------------
# ProcessedLedger — SQLite-backed duplicate gate.
# -----------------------------------------------------------------------------


class ProcessedLedger:
    """Records files we've already handled, so a re-drop is a no-op.

    Keyed by sha256 of the source bytes; survives restarts. Marks each
    record success or failure so we don't loop on a file that's
    permanently unreadable.
    """

    SCHEMA: ClassVar[str] = """
        CREATE TABLE IF NOT EXISTS processed (
            sha256 TEXT PRIMARY KEY,
            source_name TEXT NOT NULL,
            outcome TEXT NOT NULL,
            odoo_id INTEGER,
            attachment_id INTEGER,
            recorded_at REAL NOT NULL
        )
    """

    def __init__(self, path: Path) -> None:
        self._path: Path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock: threading.Lock = threading.Lock()
        self._conn: sqlite3.Connection = sqlite3.connect(
            self._path, check_same_thread=False
        )
        self._conn.execute(self.SCHEMA)
        self._conn.commit()

    @staticmethod
    def file_digest(path: Path) -> str:
        h: hashlib._Hash = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(64 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def has(self, digest: str) -> bool:
        with self._lock:
            row: tuple[Any, ...] | None = self._conn.execute(
                "SELECT 1 FROM processed WHERE sha256 = ?", (digest,)
            ).fetchone()
        return row is not None

    def record_success(
        self, digest: str, source_name: str, odoo_id: int, attachment_id: int
    ) -> None:
        self._upsert(digest, source_name, "success", odoo_id, attachment_id)

    def record_failure(self, digest: str, source_name: str) -> None:
        self._upsert(digest, source_name, "failure", None, None)

    def _upsert(
        self,
        digest: str,
        source_name: str,
        outcome: str,
        odoo_id: int | None,
        attachment_id: int | None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO processed (sha256, source_name, outcome, odoo_id, attachment_id, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(sha256) DO UPDATE SET
                    outcome = excluded.outcome,
                    odoo_id = excluded.odoo_id,
                    attachment_id = excluded.attachment_id,
                    recorded_at = excluded.recorded_at
                """,
                (digest, source_name, outcome, odoo_id, attachment_id, time.time()),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# -----------------------------------------------------------------------------
# Mailer — SMTP failure notifier.
# -----------------------------------------------------------------------------


class Mailer:
    """Send a failure email with the offending file attached."""

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        *,
        use_tls: bool = True,
        retry: int = 3,
        retry_sleep: float = 1.0,
    ) -> None:
        self._host: str = host
        self._port: int = port
        self._user: str = user
        self._password: str = password
        self._use_tls: bool = use_tls
        self._retry: int = retry
        self._retry_sleep: float = retry_sleep

    def send_failure(
        self,
        to_addr: str,
        subject: str,
        body: str,
        attachment: tuple[str, bytes, str] | None = None,
    ) -> None:
        msg: EmailMessage = EmailMessage()
        msg["From"] = self._user
        msg["To"] = to_addr
        msg["Subject"] = subject
        msg.set_content(body)
        if attachment is not None:
            name: str
            payload: bytes
            mime: str
            name, payload, mime = attachment
            maintype: str
            subtype: str
            maintype, _, subtype = mime.partition("/")
            msg.add_attachment(
                payload, maintype=maintype, subtype=subtype, filename=name
            )
        last: Exception | None = None
        for attempt in range(self._retry + 1):
            try:
                with smtplib.SMTP(self._host, self._port, timeout=30) as smtp:
                    if self._use_tls:
                        smtp.starttls()
                    if self._user and self._password:
                        smtp.login(self._user, self._password)
                    smtp.send_message(msg)
                return
            except (smtplib.SMTPException, OSError) as e:
                last = e
                if attempt < self._retry:
                    time.sleep(self._retry_sleep * (attempt + 1))
        assert last is not None
        raise RuntimeError(f"mail send failed after {self._retry + 1} attempts") from last


# -----------------------------------------------------------------------------
# StatsTracker — region-hit counter for tuning the search regions.
# -----------------------------------------------------------------------------


class StatsTracker:
    """YAML-backed counter of (doc_type, region) → hits.

    Read on startup, updated in-process, persisted on stop / explicit
    ``flush()``. Workers send their increments back to the main process
    via the ``record(...)`` API; the main process owns the file.
    """

    def __init__(self, path: Path) -> None:
        self._path: Path = Path(path)
        self._lock: threading.Lock = threading.Lock()
        self._counts: dict[str, dict[str, int]] = {}
        if self._path.exists():
            data: dict[str, dict[str, int]] = (
                yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
            )
            for doc_type, regions in data.items():
                self._counts[doc_type] = {str(k): int(v) for k, v in (regions or {}).items()}

    def record(self, doc_type: str, region: Region) -> None:
        key: str = f"{region.x1_pct},{region.y1_pct},{region.x2_pct},{region.y2_pct}"
        with self._lock:
            bucket: dict[str, int] = self._counts.setdefault(doc_type, {})
            bucket[key] = bucket.get(key, 0) + 1

    def flush(self) -> None:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(yaml.safe_dump(self._counts), encoding="utf-8")

    def snapshot(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {dt: dict(regions) for dt, regions in self._counts.items()}


# -----------------------------------------------------------------------------
# Pipeline — per-document state machine.
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProcessOutcome:
    """Result of a single Pipeline.process() call."""

    source: Path
    success: bool
    invoice_name: str | None
    odoo_id: int | None
    attachment_id: int | None
    archive_path: Path | None
    region: Region | None
    error: str | None


class Pipeline:
    """Glues classification → preprocess → OCR → Odoo → archive → ledger together.

    Stateless across calls — safe to run inside a worker. Each worker
    builds its own Pipeline once via ``for_worker(...)`` and reuses it
    for every file submitted.
    """

    def __init__(
        self,
        config: Config,
        registry: DocumentTypeRegistry,
        ocr_preparer: OcrPreparer,
        ocr_engine: OcrEngine,
        storage_preparer: StoragePreparer,
        odoo_client: OdooClient,
        archiver: Archiver,
        ledger: ProcessedLedger,
        mailer: Mailer | None,
        stats: StatsTracker,
    ) -> None:
        self._config: Config = config
        self._registry: DocumentTypeRegistry = registry
        self._ocr_preparer: OcrPreparer = ocr_preparer
        self._ocr_engine: OcrEngine = ocr_engine
        self._storage_preparer: StoragePreparer = storage_preparer
        self._odoo: OdooClient = odoo_client
        self._archiver: Archiver = archiver
        self._ledger: ProcessedLedger = ledger
        self._mailer: Mailer | None = mailer
        self._stats: StatsTracker = stats
        self._log: logging.Logger = logging.getLogger("scanrunner.pipeline")

    @classmethod
    def for_worker(cls, config_path: str, server_name: str, inbox: Path) -> Self:
        """Build a Pipeline inside a worker process. Reuse for every file."""
        config: Config = Config.load(config_path, server_name)
        # cv2 thread count is per-worker. Tuned alongside `workers` so
        # `workers * cv2_threads` stays at or below the box's core count.
        cv2.setNumThreads(max(1, config.cv2_threads))
        registry: DocumentTypeRegistry = DocumentTypeRegistry.from_config(config)
        archiver: Archiver = Archiver(inbox / config.done_path.lstrip("/"))
        ledger: ProcessedLedger = ProcessedLedger(inbox / ".processed.sqlite3")
        odoo: OdooClient = OdooClient(
            url=config.server.url,
            database=config.server.database,
            username=config.server.username,
            password=config.server.password,
            verify_tls=config.server.verify_tls,
            retry=config.retry,
            retry_sleep=config.retry_sleep,
        )
        mailer: Mailer | None = None
        if config.error_email:
            mailer = Mailer(
                host=config.server.smtp_server,
                port=config.server.smtp_port,
                user=config.server.smtp_user,
                password=config.server.smtp_password,
                use_tls=config.server.smtp_use_tls,
                retry=config.retry,
                retry_sleep=config.retry_sleep,
            )
        return cls(
            config=config,
            registry=registry,
            ocr_preparer=OcrPreparer(),
            ocr_engine=OcrEngine(config.tesseract_bin),
            storage_preparer=StoragePreparer(),
            odoo_client=odoo,
            archiver=archiver,
            ledger=ledger,
            mailer=mailer,
            stats=StatsTracker(inbox / config.statistics_file.lstrip("/")),
        )

    def process(self, source: Path, *, keep_original: bool = False) -> ProcessOutcome:
        digest: str = ProcessedLedger.file_digest(source)
        if self._ledger.has(digest):
            self._log.info("skip duplicate (already processed): %s", source.name)
            return ProcessOutcome(source, True, None, None, None, None, None, None)
        try:
            return self._process_inner(source, digest, keep_original)
        except Exception as e:
            self._log.exception("unhandled error processing %s", source)
            self._handle_failure(source, digest, error=str(e))
            return ProcessOutcome(
                source, False, None, None, None, None, None, str(e)
            )

    def _process_inner(
        self, source: Path, digest: str, keep_original: bool
    ) -> ProcessOutcome:
        # 1. Classify
        doc_type: DocumentType | None = self._registry.classify(source)
        if doc_type is None:
            self._log.warning("no DocumentType matches %s", source.name)
            self._handle_failure(source, digest, "no matching document type")
            return ProcessOutcome(
                source, False, None, None, None, None, None,
                "no matching document type",
            )

        # 2. Read + per-type preprocess
        bgr: BgrImage | None = cv2.imread(str(source))
        if bgr is None:
            self._handle_failure(source, digest, "cv2 could not read file")
            return ProcessOutcome(
                source, False, None, None, None, None, None, "unreadable image"
            )
        cleaned: BgrImage = doc_type.preprocess(bgr)

        # 3. OCR pass — cascade through fallbacks if the primary misses.
        binary: GrayImage = self._ocr_preparer.binarize(cleaned)
        result: OcrResult | None = self._ocr_engine.extract_with_fallbacks(
            binary,
            primary_regions=doc_type.config.search_regions,
            primary_config=doc_type.config.tesseract_config,
            fallback_regions=doc_type.config.fallback_search_regions,
            fallback_configs=doc_type.config.fallback_tesseract_configs,
            try_rotation=doc_type.config.ocr_try_rotation,
            regex=doc_type.regex,
        )
        if result is None:
            self._handle_failure(source, digest, "OCR did not match regex", cleaned=cleaned)
            return ProcessOutcome(
                source, False, None, None, None, None, None, "OCR miss"
            )
        ocr: OcrMatch = result.match
        if result.rotation_degrees:
            self._log.info(
                "OCR extracted %s from %s (rotated %d°)",
                ocr.name, source.name, result.rotation_degrees,
            )
            # Keep storage / archive in the orientation that OCR succeeded
            # at — the office reviewer never sees a sideways scan.
            cleaned = np.rot90(cleaned, k=result.rotation_degrees // 90)
        else:
            self._log.info("OCR extracted %s from %s", ocr.name, source.name)

        # 4. Odoo lookup
        odoo_id: int | None = self._odoo.find_record(
            doc_type.config.odoo_object, ocr.name
        )
        if odoo_id is None:
            self._handle_failure(
                source, digest, f"no Odoo record for {ocr.name}", cleaned=cleaned,
            )
            return ProcessOutcome(
                source, False, ocr.name, None, None, None, ocr.region,
                "Odoo record not found",
            )

        # 5. Storage prep — uses the rotation-corrected `cleaned` array.
        payload: bytes
        mime: str
        payload, mime = self._storage_preparer.prepare(cleaned)

        # 6. Odoo attach + link
        attachment_id: int = self._odoo.attach(
            doc_type.config.odoo_object,
            odoo_id,
            self._attachment_filename(ocr.name, source.name, mime),
            mime,
            payload,
        )
        self._odoo.link_to_documents_app(
            attachment_id=attachment_id,
            folder_id=doc_type.config.odoo_folder_id,
            tag_id=doc_type.config.odoo_attachment_tag_id,
        )

        # 7. Archive
        ext: str = "jpg" if mime == "image/jpeg" else "png"
        parsed: ParsedName = ParsedName.parse(ocr.name)
        archive_path: Path = self._archiver.archive(
            parsed, odoo_id, attachment_id, source.name, payload, ext
        )

        # 8. Ledger + stats + cleanup
        self._ledger.record_success(digest, source.name, odoo_id, attachment_id)
        self._stats.record(doc_type.name, ocr.region)
        if not keep_original:
            try:
                source.unlink()
            except FileNotFoundError:
                pass

        self._log.info(
            "OK %s id=%s aid=%s → %s", ocr.name, odoo_id, attachment_id, archive_path
        )
        return ProcessOutcome(
            source, True, ocr.name, odoo_id, attachment_id, archive_path, ocr.region, None
        )

    def _attachment_filename(self, ocr_name: str, original: str, mime: str) -> str:
        ext: str = "jpg" if mime == "image/jpeg" else "png"
        slug: str = ocr_name.replace("/", "-")
        stem: str = Path(original).stem
        return f"{slug}_{stem}.{ext}"

    def _handle_failure(
        self,
        source: Path,
        digest: str,
        error: str,
        *,
        cleaned: BgrImage | None = None,
    ) -> None:
        """Route a failed file into ``done/unreadable/`` and (optionally) email.

        If ``cleaned`` is supplied (we got far enough to preprocess the
        image — OCR miss, Odoo miss), we encode it once via
        ``StoragePreparer`` and use those bytes for BOTH the unreadable
        archive and the email attachment. The reviewer sees the same
        readable, yellow-stripped, rotation-corrected image the office
        would have got on success.

        If ``cleaned`` is None (no DocumentType matched, cv2 couldn't
        read the file, generic exception), we fall back to the raw
        original bytes with a magic-sniffed MIME.
        """
        self._log.error("FAIL %s: %s", source.name, error)

        payload: bytes | None = None
        mime: str = "application/octet-stream"
        ext: str = ""
        if cleaned is not None:
            payload, mime = self._storage_preparer.prepare(cleaned)
            ext = "jpg" if mime == "image/jpeg" else "png"

        # Snapshot raw bytes BEFORE the move, used as fallback.
        attach: tuple[str, bytes, str] | None = None
        if payload is not None:
            attach = (f"{source.stem}.{ext}", payload, mime)
        elif source.exists():
            try:
                sniffed: str = magic.from_file(str(source), mime=True) or "application/octet-stream"
            except Exception:
                sniffed = "application/octet-stream"
            attach = (source.name, source.read_bytes(), sniffed)

        try:
            self._archiver.archive_unreadable(source, payload, ext)
        except Exception:
            self._log.exception("could not move %s to unreadable/", source)
        self._ledger.record_failure(digest, source.name)

        if self._mailer is not None and self._config.error_email and attach is not None:
            try:
                self._mailer.send_failure(
                    self._config.error_email,
                    subject=f"[scanrunner] could not process {source.name}",
                    body=f"{self._config.error_mail_message}\n\n{error}",
                    attachment=attach,
                )
            except Exception:
                self._log.exception("could not send failure email for %s", source)


# -----------------------------------------------------------------------------
# Daemon — watchdog + ProcessPoolExecutor + signal-aware shutdown.
# -----------------------------------------------------------------------------


class WorkSubmitter:
    """Owns the ProcessPoolExecutor and dedupes in-flight submissions."""

    def __init__(
        self,
        config_path: str,
        server_name: str,
        inbox: Path,
        max_workers: int | None = None,
    ) -> None:
        self._config_path: str = config_path
        self._server_name: str = server_name
        self._inbox: Path = inbox
        # `max_workers` from the caller (Daemon reads `workers:` from
        # config.yaml). If left None, fall back to a single worker —
        # appropriate for the typical 1-file-every-5-seconds production
        # arrival rate. Override in YAML for genuinely bursty inboxes.
        self._max_workers: int = max_workers or 1
        # forkserver: predictable startup, no fork-after-thread foot-gun.
        ctx: Any = get_context("forkserver")
        self._pool: ProcessPoolExecutor = ProcessPoolExecutor(
            max_workers=self._max_workers, mp_context=ctx
        )
        self._inflight: set[str] = set()
        self._lock: threading.Lock = threading.Lock()
        self._log: logging.Logger = logging.getLogger("scanrunner.submitter")

    @property
    def max_workers(self) -> int:
        return self._max_workers

    def submit(self, source: Path) -> Future | None:
        path_key: str = str(source.resolve())
        with self._lock:
            if path_key in self._inflight:
                return None
            self._inflight.add(path_key)
        future: Future = self._pool.submit(
            _worker_process_path,
            path_key,
            self._config_path,
            self._server_name,
            str(self._inbox),
        )
        def _done(_f: Future, k: str = path_key) -> None:
            self._on_complete(k)
        future.add_done_callback(_done)
        return future

    def _on_complete(self, path_key: str) -> None:
        with self._lock:
            self._inflight.discard(path_key)

    def shutdown(self, *, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=not wait)


# Module-level worker entrypoint — picklable by reference.
_WORKER_PIPELINE: Pipeline | None = None


def _worker_process_path(
    source_str: str, config_path: str, server_name: str, inbox_str: str
) -> bool:
    """ProcessPool worker: lazy-init a Pipeline, then process the file."""
    global _WORKER_PIPELINE
    if _WORKER_PIPELINE is None:
        _WORKER_PIPELINE = Pipeline.for_worker(config_path, server_name, Path(inbox_str))
    outcome: ProcessOutcome = _WORKER_PIPELINE.process(Path(source_str))
    return outcome.success


class FileWatcher:
    """Watchdog Observer wired to ``WorkSubmitter`` via on_closed events."""

    SUFFIXES: ClassVar[frozenset[str]] = frozenset({".jpg", ".jpeg", ".png", ".pdf"})

    def __init__(self, inbox: Path, submitter: WorkSubmitter) -> None:
        self._inbox: Path = Path(inbox)
        self._submitter: WorkSubmitter = submitter
        # watchdog.observers.Observer is a polymorphic factory; the concrete
        # type is platform-specific (InotifyObserver on Linux).
        self._observer: Any = Observer()
        self._handler: _CloseHandler = _CloseHandler(self)
        self._log: logging.Logger = logging.getLogger("scanrunner.watcher")

    def start(self) -> None:
        self._observer.schedule(self._handler, str(self._inbox), recursive=False)
        self._observer.start()

    def stop(self) -> None:
        self._observer.stop()
        self._observer.join()

    def initial_sweep(self) -> int:
        count: int = 0
        for path in sorted(self._inbox.iterdir()):
            if path.is_file() and path.suffix.lower() in self.SUFFIXES:
                self._log.info("initial-sweep submit: %s", path.name)
                self._submitter.submit(path)
                count += 1
        return count

    def on_closed(self, src_path: Path) -> None:
        if src_path.suffix.lower() not in self.SUFFIXES:
            return
        self._log.info("file closed → submit: %s", src_path.name)
        self._submitter.submit(src_path)


class _CloseHandler(FileSystemEventHandler):
    def __init__(self, watcher: FileWatcher) -> None:
        super().__init__()
        self._watcher: FileWatcher = watcher

    @override
    def on_closed(self, event: FileClosedEvent) -> None:  # IN_CLOSE_WRITE
        if event.is_directory:
            return
        # watchdog types src_path as bytes | str; we only watch a str-pathed
        # inbox so it's always str in practice, but be explicit for mypy.
        src: str = event.src_path if isinstance(event.src_path, str) else event.src_path.decode()
        self._watcher.on_closed(Path(src))


class Daemon:
    """Top-level wiring: parse args, build watcher + submitter, wait for SIGTERM."""

    def __init__(
        self,
        config_path: str,
        server_name: str,
        inbox: Path,
        log_level: int = logging.INFO,
    ) -> None:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s %(processName)s %(levelname)s %(name)s: %(message)s",
        )
        # Suppress chatter from httpx / urllib3 wire-level loggers; keep our
        # scanrunner.* loggers at the requested level.
        for noisy in ("httpx", "httpcore", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        self._log: logging.Logger = logging.getLogger("scanrunner.daemon")
        self._inbox: Path = Path(inbox)
        # Pull worker count from the YAML so the same image can run on a
        # 4-core prod box and a multi-core dev workstation with sensible
        # behavior on each.
        config: Config = Config.load(config_path, server_name)
        self._submitter: WorkSubmitter = WorkSubmitter(
            config_path, server_name, self._inbox, max_workers=config.workers,
        )
        self._watcher: FileWatcher = FileWatcher(self._inbox, self._submitter)
        self._stop: threading.Event = threading.Event()
        signal.signal(signal.SIGTERM, self._signal)
        signal.signal(signal.SIGINT, self._signal)

    def _signal(self, *_: object) -> None:
        self._log.info("shutdown signal received")
        self._stop.set()

    def run(self) -> int:
        self._log.info(
            "starting watcher on %s with %d worker(s)",
            self._inbox,
            self._submitter.max_workers,
        )
        self._watcher.start()
        try:
            n: int = self._watcher.initial_sweep()
            self._log.info("initial sweep submitted %d file(s)", n)
            self._stop.wait()
        finally:
            self._log.info("stopping watcher and draining workers")
            self._watcher.stop()
            self._submitter.shutdown(wait=True)
        return 0

    @classmethod
    def cli(cls, argv: list[str]) -> int:
        parser: argparse.ArgumentParser = argparse.ArgumentParser(prog="docscanner")
        sub = parser.add_subparsers(dest="cmd", required=True)
        run: argparse.ArgumentParser = sub.add_parser(
            "daemon", help="run as a watch+process daemon"
        )
        run.add_argument("inbox", type=Path, help="directory to watch")
        run.add_argument("-c", "--config", default=os.environ.get("DS_CONFIG", "/etc/docscanner/config.yaml"))
        run.add_argument("-s", "--server", default=os.environ.get("DS_SERVER", "production"))
        run.add_argument("-v", "--verbose", action="store_true")
        once: argparse.ArgumentParser = sub.add_parser(
            "process", help="process one file (one-shot)"
        )
        once.add_argument("path", type=Path)
        once.add_argument("inbox", type=Path, help="archive root (used for done/, ledger)")
        once.add_argument("-c", "--config", default=os.environ.get("DS_CONFIG", "/etc/docscanner/config.yaml"))
        once.add_argument("-s", "--server", default=os.environ.get("DS_SERVER", "production"))
        once.add_argument("--keep", action="store_true", help="keep the source file after processing")
        once.add_argument("-v", "--verbose", action="store_true")
        ns: argparse.Namespace = parser.parse_args(argv)
        log_level: int = logging.DEBUG if ns.verbose else logging.INFO
        match ns.cmd:
            case "daemon":
                return cls(ns.config, ns.server, ns.inbox, log_level).run()
            case "process":
                logging.basicConfig(level=log_level, format="%(message)s")
                pipeline: Pipeline = Pipeline.for_worker(ns.config, ns.server, ns.inbox)
                outcome: ProcessOutcome = pipeline.process(ns.path, keep_original=ns.keep)
                return 0 if outcome.success else 1
            case other:
                raise ValueError(f"unknown subcommand {other}")


if __name__ == "__main__":
    raise SystemExit(Daemon.cli(sys.argv[1:]))
