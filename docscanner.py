#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "opencv-python-headless>=4.10",
#   "Pillow>=11.0",
#   "numpy>=2.0",
#   "python-magic>=0.4.27",
#   "PyYAML>=6.0",
#   "watchdog>=5.0",
#   "httpx[http2]>=0.27",
#   "python-doctr>=1.0",
#   "torch>=2.4",
#   "torchvision>=0.19",
# ]
#
# # Pin torch + torchvision to the CPU-only wheel index. The default PyPI
# # wheels carry a CUDA build that drags in nvidia_* runtime libs (cuBLAS,
# # cuDNN, cuFFT, NCCL, etc.) totalling ~3 GB. We run inference on CPU
# # only — DocTR + CRNN at ~5-15 s/page is fine — so the CUDA libs are
# # pure dead weight in the container image. `explicit = true` keeps the
# # CPU index from intercepting unrelated packages (numpy, doctr, etc.).
# [tool.uv.sources]
# torch = [{ index = "pytorch-cpu" }]
# torchvision = [{ index = "pytorch-cpu" }]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
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

    ORPHAN (v2 stage). Not wired into the v3 storage pipeline; the
    layered DocumentDecomposer handles uneven illumination implicitly
    via the substrate-aware ink threshold (paper_color +
    paper_variation from estimate_paper_tone). Retained as a tested
    primitive in case a future pipeline (e.g. a non-DocTR fallback or
    a different document type with its own preprocess chain) needs it.
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

    ORPHAN (v2 stage). Not wired into the v3 storage pipeline; the
    layered DocumentDecomposer composites onto an explicit white canvas
    and the per-layer extractors classify pixels by ink-vs-paper directly,
    so adaptive-threshold foreground discovery is no longer needed.
    Retained as a tested primitive for potential reuse in future
    pipelines.
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

    ORPHAN (v2 stage). Not wired into the v3 storage pipeline.
    PrintedLayerExtractor now invokes DocTR's full ocr_predictor
    (detection + recognition) directly so it can reject low-confidence
    "text" regions that turn out to be handwriting. This detector-only
    wrapper remains a tested primitive in case a future caller wants
    cheaper detection-only word boxes without paying for recognition.
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

    ORPHAN (v2 stage). Not wired into the v3 storage pipeline; the
    HandwrittenLayerExtractor catches faint pen ink directly via the
    substrate-aware "darker than paper" threshold (paper_grayscale -
    INK_DARKER_THAN_PAPER_BASE - INK_DARKER_THAN_PAPER_VAR_COEFF *
    paper_variation), without needing iterative neighborhood growing.
    Retained as a tested primitive — the dilate/anchor/rescue pattern
    could be useful for future faint-content recovery problems where
    a paper-tone baseline alone isn't enough.
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
    """Drop isolated foreground speckle (paper fiber, toner spray); keep strokes.

    ORPHAN (v2 stage). Not wired into the v3 storage pipeline;
    HandwrittenLayerExtractor performs equivalent CC-area filtering
    (MIN_INK_COMPONENT_AREA_PX = 12) directly inside its extract()
    method, and PrintedLayerExtractor screens via DocTR word boxes
    instead of CC area. Retained as a tested primitive in case a
    future caller needs standalone speckle removal.
    """

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

    ORPHAN (compaction primitive). Implemented as a v2 storage-size
    optimization but never wired into the production pipeline because
    it damaged faint handwriting and printed text in ways the office
    found unacceptable. Retained for potential future use under
    different parameter calibration. v3 has not attempted to revive it.
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
    small. The v3 encoder uses 8 levels — calibrated against the full
    good/ corpus to keep ≥90 % of pages under the 300 KB cap (max 324 KB,
    median 221 KB). 16 levels was rejected because it only fit 64 % of
    pages. Layered decomposition (PrintedLayer + HandwrittenLayer)
    happens *before* quantization, so faint pen ink is already preserved
    in the foreground mask — the v2 reason for orphaning this stage no
    longer applies.
    """

    DEFAULT_LEVELS: ClassVar[int] = 8
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
# Storage encoder — single calibrated setting (KISS, no runtime branches).
# -----------------------------------------------------------------------------
#
# Calibrated 2026-05-03 against 168 v3 composites from corpus/invoices/good/:
#   * downsample factor 2/3 (≈300 dpi → 200 dpi for storage)
#   * 8-level foreground quantization (ForegroundQuantizer.DEFAULT_LEVELS)
#   * PNG grayscale, optimize=True, compress_level=9
# Result: 91.1 % of pages ≤ 300 KB cap, max 324 KB, p50 221 KB.
#
# Sweep also tried JPEG q=60..85 (max 30..54 % fit), Q16+PNG (64 %),
# WebP q=70 (79 %). None hit the 90 % bar. Q8+PNG won decisively, so
# the encoder no longer branches per file — same format every time.


class Encoder:
    """Quantize foreground to 8 levels and encode as grayscale PNG.

    The (payload, mimetype) tuple is preserved so downstream callers can
    pick the file extension; mimetype is always ``"image/png"``.
    """

    SIZE_CAP_BYTES: ClassVar[int] = 300_000  # informational; not enforced

    def __init__(self, quantizer: ForegroundQuantizer | None = None) -> None:
        self._quantizer: ForegroundQuantizer = quantizer or ForegroundQuantizer()

    def encode(self, snapped: GrayImage) -> tuple[bytes, str]:
        """Return (payload, mimetype) for the storage-ready grayscale image."""
        quantized: GrayImage = self._quantizer.quantize(snapped)
        img: Image.Image = Image.fromarray(quantized, mode="L")
        buf: io.BytesIO = io.BytesIO()
        img.save(buf, format="PNG", optimize=True, compress_level=9)
        return buf.getvalue(), "image/png"


# --- shared DocTR OCR predictor ----------------------------------------------
# Both PrintedLayerExtractor (recognition-confidence screen) and OcrEngine
# (invoice-number extraction) need the same DocTR ocr_predictor. Loading the
# model is the expensive step (~63 MB pulled from S3 on first call, ~5 s
# on warm cache); the inference itself is comparatively cheap. Caching at
# module scope guarantees one load per worker process — across PrintedLayer
# extractions and OCR cascades alike.
_DOCTR_OCR_MODEL: Any | None = None


def _get_doctr_ocr_model() -> Any:
    """Lazily build (and cache) the DocTR OCR predictor.

    The import of :mod:`doctr.models` is deferred so module load does
    not drag torch in for callers that never touch the predictor.
    """
    global _DOCTR_OCR_MODEL
    if _DOCTR_OCR_MODEL is None:
        from doctr.models import ocr_predictor
        _DOCTR_OCR_MODEL = ocr_predictor(pretrained=True)
    return _DOCTR_OCR_MODEL


# --- shared substrate-aware ink threshold ------------------------------------
# Both PrintedLayerExtractor and HandwrittenLayerExtractor classify a pixel as
# ink when its grayscale value drops more than
#   INK_DARKER_THAN_PAPER_BASE + INK_DARKER_THAN_PAPER_VAR_COEFF * paper_variation
# below the paper's grayscale equivalent. Sharing the constants keeps the two
# extractors' "darker than paper" decisions calibrated identically; if one ever
# needs to tighten, both should move together.
INK_DARKER_THAN_PAPER_BASE: int = 60
INK_DARKER_THAN_PAPER_VAR_COEFF: float = 2.0


def _paper_color_to_grayscale(bgr_color: BgrColor) -> int:
    """Convert a BGR paper color to its grayscale equivalent (BT.601 weights).

    BT.601 matches ``cv2.COLOR_BGR2GRAY``, so a comparison between this
    return value and a ``cv2.cvtColor(..., COLOR_BGR2GRAY)`` array is
    apples-to-apples.
    """
    paper_blue, paper_green, paper_red = bgr_color
    return int(round(
        0.114 * paper_blue + 0.587 * paper_green + 0.299 * paper_red
    ))


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

    Strategy: DocTR's pretrained recognition model (``crnn_vgg16_bn``)
    scores each detected word with a per-word confidence. Words are
    grouped into lines by DocTR's layout pass; lines whose median word
    confidence falls below :attr:`MIN_LINE_RECOGNITION_CONFIDENCE` are
    rejected as non-printed (handwriting reads as gibberish to the
    recognizer and confidence collapses). Surviving word boxes are
    intersected with a substrate-aware "darker than paper" pixel test to
    convert padded boxes into pixel-precise stroke masks.
    """

    MIN_LINE_RECOGNITION_CONFIDENCE: ClassVar[float] = 0.75
    """Minimum median word recognition confidence required to classify a
    DocTR-detected line as printed text. Probe against this corpus
    showed printed lines at 0.984-0.996 median, handwritten at
    0.605-0.655. A 0.85 threshold cleanly separates the populations
    on white paper but rejects yellow-paper printed lines whose
    confidence drops because of substrate-induced recognition noise
    (e.g. "$836.07" reads at 0.824 on yellow). 0.75 keeps those
    printed amounts intact at the cost of letting block-letter
    handwriting whose recognized tokens skew high (e.g. "3FD Kraft
    2 Back order") leak through. The user-validated trade: accept
    the leak rather than lose any printed text."""

    MIN_TOKEN_LEN_FOR_CONFIDENCE: ClassVar[int] = 3
    """Minimum token length (characters) for a recognized word to count
    toward a line's confidence median. Short tokens — single digits,
    currency symbols, dashes — recognize at near-1.0 confidence even
    when handwritten, biasing the median upward on handwriting lines
    that happen to include numerals. Lines with NO long words fall
    back to all-words confidences so single-token printed amounts
    like "$836.07" still pass the threshold."""

    WORD_BOX_PADDING_PX: ClassVar[int] = 4
    """Pixels to pad each accepted word box. Captures anti-aliased stroke
    edges that fall just outside DocTR's tight box. Smaller than
    InkRegionDetector.PADDING_PX (18) because the recognition screen
    already filters out non-printed boxes upstream — we don't need extra
    slack to absorb noise."""

    def __init__(self) -> None:
        """Build a stateless extractor.

        The DocTR OCR model is loaded lazily via :func:`_get_doctr_ocr_model`
        and cached at module scope, so construction is free.
        """
        # Stateless; model loaded lazily via _get_doctr_ocr_model().

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
        # DocTR expects RGB; OpenCV gives us BGR.
        rgb: np.ndarray = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height: int
        width: int
        height, width = bgr.shape[:2]

        # Full OCR pass: detection + recognition. We need per-word
        # confidence to screen out handwriting that the detector would
        # otherwise pass through.
        result: Any = _get_doctr_ocr_model()([rgb])
        page: Any = result.pages[0]

        # Build a printed-line mask from the union of word boxes
        # belonging to lines whose median word confidence clears the
        # threshold. Handwriting collapses recognizer confidence (~0.6),
        # so its lines are rejected wholesale here.
        printed_line_mask: np.ndarray = self._build_printed_line_mask(
            page, height, width,
        )

        # Tones source: original grayscale carries the actual ink
        # darkness at masked pixels; outside the mask the value is
        # unspecified by the contract but must remain valid uint8
        # (which grayscale is by construction).
        tones: np.ndarray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Substrate-aware "darker than paper" screen. BT.601 weights
        # match cv2.COLOR_BGR2GRAY, which is the convention `tones` uses.
        ink_mask: np.ndarray = self._compute_ink_mask(
            tones, paper_color, paper_variation,
        )

        # Final mask: printed-line geometry AND ink darkness. Blank
        # paper fails the ink gate; handwriting fails the recognition
        # gate upstream.
        printed_mask: np.ndarray = printed_line_mask & ink_mask

        return PrintedLayer(mask=printed_mask, tones=tones)

    def _build_printed_line_mask(
        self,
        page: Any,
        height: int,
        width: int,
    ) -> np.ndarray:
        """Return a boolean mask of word boxes from confidently-recognized lines.

        Iterates DocTR's block/line/word hierarchy. For each line, takes
        the median per-word recognition confidence and accepts the line
        only if it clears :attr:`MIN_LINE_RECOGNITION_CONFIDENCE`. Each
        accepted word's normalized geometry is converted to pixel
        coordinates and stamped (with :attr:`WORD_BOX_PADDING_PX` of
        slack on each side) into the output mask.
        """
        line_mask: np.ndarray = np.zeros((height, width), dtype=bool)
        pad: int = self.WORD_BOX_PADDING_PX
        for block in page.blocks:
            for line in block.lines:
                words: list[Any] = list(line.words)
                if not words:
                    continue
                # Confidence median is computed only over multi-character
                # tokens. Single-char tokens like "$", "2", "-" recognize
                # at confidence ~1.0 regardless of authenticity, so they
                # drag the median up on handwriting lines that include
                # digits (e.g. "3FD Kratt 2 Back order" has DocTR scoring
                # 2 → 1.00 and 3FD → 0.99 even though it's pen ink).
                # Falling back to all-words confidences when every word
                # is short keeps single-word lines like "$836.07" intact.
                long_word_confidences: np.ndarray = np.array(
                    [
                        word.confidence
                        for word in words
                        if len(word.value) >= self.MIN_TOKEN_LEN_FOR_CONFIDENCE
                    ]
                )
                if long_word_confidences.size == 0:
                    long_word_confidences = np.array(
                        [word.confidence for word in words]
                    )
                if (
                    float(np.median(long_word_confidences))
                    < self.MIN_LINE_RECOGNITION_CONFIDENCE
                ):
                    # Reject this line as likely non-printed (handwriting,
                    # noise, or other content the recognizer can't read).
                    continue
                for word in words:
                    # word.geometry is ((x_min, y_min), (x_max, y_max))
                    # in normalized [0, 1] coordinates.
                    (x1_norm, y1_norm), (x2_norm, y2_norm) = word.geometry
                    x1: int = max(0, int(x1_norm * width) - pad)
                    y1: int = max(0, int(y1_norm * height) - pad)
                    x2: int = min(width, int(x2_norm * width) + pad)
                    y2: int = min(height, int(y2_norm * height) + pad)
                    line_mask[y1:y2, x1:x2] = True
        return line_mask

    def _compute_ink_mask(
        self,
        grayscale: np.ndarray,
        paper_color: BgrColor,
        paper_variation: float,
    ) -> np.ndarray:
        """Return a boolean mask of pixels substantially darker than paper.

        The threshold widens with ``paper_variation`` so noisier
        substrates (yellow stock, ΔE ~1.9) do not flood the mask with
        substrate micro-variation.
        """
        paper_grayscale: int = _paper_color_to_grayscale(paper_color)
        threshold: float = (
            INK_DARKER_THAN_PAPER_BASE
            + INK_DARKER_THAN_PAPER_VAR_COEFF * paper_variation
        )
        return grayscale < (paper_grayscale - threshold)


@dataclass(frozen=True, slots=True)
class HandwrittenLayer:
    """v3 carrier: handwritten ink (pen, pencil, marker, signature) lifted off a scanned page.

    Mirrors :class:`PrintedLayer` in shape so the downstream composer can
    render the two layers with the same primitives.

    Attributes:
        mask: Boolean ndarray of shape ``(H, W)``. ``True`` where a pixel
            belongs to handwritten ink, ``False`` elsewhere. Must never
            overlap with the printed layer's mask -- the printed layer
            takes precedence and the handwritten extractor is required
            to avoid claiming any pixel already in ``printed_mask``.
        tones: ``uint8`` ndarray of shape ``(H, W)``. Grayscale ink tones
            preserved from the original scan at masked positions; values
            outside the mask are unspecified but must be valid ``uint8``.
    """

    mask: np.ndarray
    tones: np.ndarray


class HandwrittenLayerExtractor:
    """Extract the handwritten-content layer from a scanned page (v3 phase 3).

    Sibling of :class:`PrintedLayerExtractor` in the layered v3 model.
    Both extractors share the same substrate-aware "darker than paper"
    ink test (see :data:`INK_DARKER_THAN_PAPER_BASE` /
    :data:`INK_DARKER_THAN_PAPER_VAR_COEFF`), but differ in which pixels
    they claim: printed runs first and stamps confidently-recognized
    word boxes; handwriting runs second and is forbidden from claiming
    any pixel already in ``printed_mask``. This precedence rule ensures
    every ink pixel has exactly one owner so the downstream composer
    never renders the same stroke twice.
    """

    MIN_INK_COMPONENT_AREA_PX: ClassVar[int] = 12
    """Minimum connected-component area for an ink blob to qualify
    as handwriting. Smaller blobs (<= 11 px) are paper micro-noise,
    speckle, or sub-stroke fragments. Real pen strokes -- even the
    dot of an 'i' -- exceed this at 200 DPI. Calibrated against the
    blank-paper coverage ceiling (0.1%) without compromising the
    handwriting coverage floor (0.5%)."""

    def __init__(self) -> None:
        """Build a stateless extractor; no model load required."""

    def extract(
        self,
        bgr: BgrImage,
        paper_color: BgrColor,
        paper_variation: float,
        printed_mask: np.ndarray,
    ) -> HandwrittenLayer:
        """Extract the handwritten layer from ``bgr``.

        Args:
            bgr: BGR scan of the page (or a sub-crop). Shape ``(H, W, 3)``.
            paper_color: Median BGR color of the paper substrate, from
                :func:`estimate_paper_tone`. Anchors the "darker than
                paper" decision across white, cream, and yellow stocks.
            paper_variation: ΔE_lab spread of the paper population, from
                :func:`estimate_paper_tone`. Widens the ink threshold on
                noisier substrates.
            printed_mask: Boolean ``(H, W)`` mask of pixels already
                claimed by the printed layer. The returned handwritten
                mask is guaranteed disjoint from this input.

        Returns:
            A :class:`HandwrittenLayer` whose ``mask`` (bool) and
            ``tones`` (uint8) arrays both share the input's ``(H, W)``.
        """
        tones: np.ndarray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Substrate-aware ink test, identical to PrintedLayerExtractor's.
        paper_grayscale: int = _paper_color_to_grayscale(paper_color)
        threshold: float = (
            INK_DARKER_THAN_PAPER_BASE
            + INK_DARKER_THAN_PAPER_VAR_COEFF * paper_variation
        )
        ink_mask: np.ndarray = tones < (paper_grayscale - threshold)

        # Precedence rule: handwriting only sees what printed didn't claim.
        candidate_mask: np.ndarray = ink_mask & ~printed_mask

        # Drop sub-area connected components -- paper micro-noise.
        # 8-connectivity matches FaintInkRescuer / EdgeCleaner conventions
        # elsewhere in this file.
        n_labels: int
        labels: np.ndarray
        stats: np.ndarray
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidate_mask.astype(np.uint8), connectivity=8,
        )
        # Vectorized CC area filter: keep[label_id] is True when that
        # component clears the area floor. Background (label 0) is always
        # rejected so it never bleeds into the final mask.
        areas: np.ndarray = stats[:, cv2.CC_STAT_AREA]
        keep: np.ndarray = areas >= self.MIN_INK_COMPONENT_AREA_PX
        keep[0] = False
        # `labels` is int; broadcasting `keep[labels]` produces the per-pixel
        # bool mask of "label index is among kept indices."
        mask: np.ndarray = keep[labels]

        return HandwrittenLayer(mask=mask, tones=tones)


@dataclass(frozen=True, slots=True)
class Document:
    """The v3 layered storage model for a single scanned page.

    Two semantic layers -- printed (laser toner) and handwritten (pen
    ink) -- composed over an implicit white-paper canvas. Paper is not
    an explicit layer: whatever the two layers don't claim is paper,
    rendered as white by :meth:`composite`. Frozen + slotted so the
    pipeline can pass Documents across stages without defensive copies.
    """

    shape: tuple[int, int]
    printed: PrintedLayer
    handwritten: HandwrittenLayer

    def composite(self) -> np.ndarray:
        """Render the layered Document back into a single grayscale image.

        Precedence rule: on any pixel claimed by both layers, printed
        wins. The decomposer's extractors already guarantee disjoint
        masks on real scans; the order below is defense-in-depth so a
        hand-built (e.g. test) Document with overlap still resolves
        deterministically.
        """
        # Implicit paper canvas: every pixel starts at paper-white. Any
        # pixel neither layer claims keeps this value in the output.
        out = np.full(self.shape, 255, dtype=np.uint8)
        # Handwritten first, printed second -- the second write wins on
        # overlap, which encodes the precedence rule above.
        out[self.handwritten.mask] = self.handwritten.tones[self.handwritten.mask]
        out[self.printed.mask] = self.printed.tones[self.printed.mask]
        return out


class DocumentDecomposer:
    """Turn a raw BGR scan into a layered :class:`Document`.

    Orchestrates :func:`estimate_paper_tone` ->
    :class:`PrintedLayerExtractor` -> :class:`HandwrittenLayerExtractor`
    and enforces the precedence rule that no single pixel may be claimed
    by both layers: handwriting only sees pixels printed didn't claim.
    """

    def __init__(
        self,
        printed_extractor: PrintedLayerExtractor | None = None,
        handwritten_extractor: HandwrittenLayerExtractor | None = None,
    ) -> None:
        """Build a decomposer with optional injected extractors.

        Args:
            printed_extractor: Override for the printed-layer extractor.
                Defaults to a fresh :class:`PrintedLayerExtractor`.
            handwritten_extractor: Override for the handwritten-layer
                extractor. Defaults to a fresh
                :class:`HandwrittenLayerExtractor`.
        """
        self._printed = printed_extractor or PrintedLayerExtractor()
        self._handwritten = handwritten_extractor or HandwrittenLayerExtractor()

    def decompose(self, bgr: BgrImage) -> Document:
        """Decompose ``bgr`` into a layered :class:`Document`.

        Args:
            bgr: BGR scan of a single page. Shape ``(H, W, 3)``.

        Returns:
            A :class:`Document` whose ``shape`` matches ``bgr.shape[:2]``
            and whose ``printed`` / ``handwritten`` layers are populated
            with disjoint masks (printed wins on overlap).
        """
        # Estimate the paper substrate first so both extractors can
        # calibrate their "darker than paper" thresholds against the same
        # reference -- white, cream, and yellow stocks all flow through
        # the same code path with no per-substrate branching here.
        paper_color, paper_variation = estimate_paper_tone(bgr)

        # Printed runs first; handwriting is told which pixels are
        # already claimed so the precedence rule lives in one place
        # (inside the handwritten extractor) rather than being repeated here.
        printed: PrintedLayer = self._printed.extract(
            bgr, paper_color, paper_variation,
        )
        handwritten: HandwrittenLayer = self._handwritten.extract(
            bgr, paper_color, paper_variation, printed.mask,
        )

        return Document(
            shape=bgr.shape[:2],
            printed=printed,
            handwritten=handwritten,
        )


class StoragePreparer:
    """v3 storage pipeline: decompose → composite → downsample → encode.

    Takes a raw BGR scan, runs the v3 layered decomposition (paper /
    printed / handwritten), composites the two ink layers onto a clean
    white canvas, downsamples to TARGET_DPI, and encodes for Odoo.

    The substrate-aware decomposer handles yellow paper natively via
    estimate_paper_tone; no pre-cleanup pass (YellowRemover etc.) is
    needed here. Callers should pass the ORIGINAL scan, not a
    yellow-removed version — pre-cleanup damages faint pen ink that
    the decomposer would otherwise preserve in the handwritten layer.
    """

    SOURCE_DPI: ClassVar[int] = 300
    TARGET_DPI: ClassVar[int] = 200
    DOWNSAMPLE: ClassVar[float] = TARGET_DPI / SOURCE_DPI  # 2/3

    MAX_LONG_SIDE_PX: ClassVar[int] = 1920
    """Cap the storage image's longer side at 1920 px so Odoo's default
    ``base.image_autoresize_max_px = 1920x1920`` is a no-op on us. If we
    exceed it Odoo silently downsamples-and-re-encodes at quality 80
    server-side, destroying the carefully-calibrated v3 fidelity AND
    breaking checksum-based idempotency (the stored bytes would no
    longer match what we sent). Doing the resize ourselves with
    INTER_AREA is higher quality than letting PIL guess on the Odoo
    side. A typical 2538×3324 raw scan at the 2/3 DPI factor is
    1692×2216 — already over 1920 on the long side, so this cap fires
    on most inputs."""

    def __init__(
        self,
        decomposer: DocumentDecomposer | None = None,
        encoder: Encoder | None = None,
    ) -> None:
        self._decomposer = decomposer or DocumentDecomposer()
        self._encoder = encoder or Encoder()

    def prepare(self, bgr: BgrImage) -> tuple[bytes, str]:
        # Decompose at full source resolution: DocTR's recognition is
        # more accurate on detailed input, and the substrate-aware ink
        # decisions benefit from access to original pixel values.
        document: Document = self._decomposer.decompose(bgr)
        composite: np.ndarray = document.composite()
        # Downsample to TARGET_DPI, then further if needed to fit
        # MAX_LONG_SIDE_PX. INTER_AREA is the right interpolation for
        # shrink ops — it averages source pixels rather than sampling,
        # preserving the printed / handwritten tones the layers
        # carried through.
        h, w = composite.shape[:2]
        factor: float = min(
            self.DOWNSAMPLE,
            self.MAX_LONG_SIDE_PX / max(h, w),
        )
        new_w: int = int(w * factor)
        new_h: int = int(h * factor)
        downsampled: np.ndarray = cv2.resize(
            composite, (new_w, new_h), interpolation=cv2.INTER_AREA,
        )
        return self._encoder.encode(downsampled)


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
    odoo_sequence: str
    odoo_object: str
    odoo_attachment_tag_id: int
    odoo_folder_id: int
    # Optional cascade fallbacks. Defaults are no-op (matches today's
    # single-attempt behavior). The cascade fires only if the primary OCR
    # misses, so the fast path is unaffected.
    fallback_search_regions: tuple[tuple[Region, ...], ...] = ()
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
                    odoo_sequence=doc["odoo_sequence"],
                    odoo_object=doc["odoo_object"],
                    odoo_attachment_tag_id=int(doc["odoo_attachment_tag_id"]),
                    odoo_folder_id=int(doc["odoo_folder_id"]),
                    fallback_search_regions=tuple(
                        tuple(Region.from_list(r) for r in region_set)
                        for region_set in doc.get("fallback_search_regions", [])
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
# OCR pipeline — DocTR detection + recognition over per-region BGR crops.
# -----------------------------------------------------------------------------


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
    """Run DocTR on each region in priority order; first regex hit wins.

    DocTR's ``ocr_predictor`` runs detection + recognition end-to-end.
    The engine concatenates per-word ``.value`` text within each region
    crop, normalizes whitespace and stray punctuation, and matches the
    configured regex. Two-region priority rule lives here: the caller
    (the document type) picks the regions; this engine tries them in
    order. ``SO…`` numbers are naturally rejected by an ``R?INV``-
    anchored regex.

    For documents that may arrive sideways or upside down or that need
    fallback regions, use ``extract_with_fallbacks`` — it cascades
    through tier-by-tier and reports the rotation that worked.

    Replaces the v3.4 Tesseract backend. DocTR's recognition is more
    robust on the unreadable-corpus tail (1-bit scans, faint scanner
    output, light skew) and avoids the per-PSM tuning the Tesseract
    cascade needed.
    """

    UPSCALE_THRESHOLD_PX: ClassVar[int] = 600
    """Crop side length below which DocTR loses glyph detail. Crops
    smaller than this get bicubic-upscaled to 2× before recognition."""

    UPSCALE_FACTOR: ClassVar[int] = 2

    NOISE_CHARS: ClassVar[re.Pattern[str]] = re.compile(r"[\s,.;:_|\\]")
    ROTATION_ANGLES: ClassVar[tuple[int, ...]] = (90, 180, 270)

    def extract(
        self,
        bgr: BgrImage,
        regions: tuple[Region, ...],
        regex: re.Pattern[str],
    ) -> OcrMatch | None:
        """Run DocTR on each ``regions`` crop in order; return the first hit."""
        for region in regions:
            crop: BgrImage = region.crop(bgr)
            if crop.size == 0:
                continue
            text: str = self._recognize(crop)
            normalized: str = self.NOISE_CHARS.sub("", text)
            match: re.Match[str] | None = regex.search(normalized)
            if match:
                return OcrMatch(name=match.group(0), region=region)
        return None

    def extract_with_fallbacks(
        self,
        bgr: BgrImage,
        primary_regions: tuple[Region, ...],
        fallback_regions: tuple[tuple[Region, ...], ...],
        try_rotation: bool,
        regex: re.Pattern[str],
    ) -> OcrResult | None:
        """Cascade until a regex hit lands, or return None.

        Tier 1: primary regions (the fast path).
        Tier 2: each fallback region set.
        Tier 3: rotate 90 / 180 / 270 and re-run primary regions.

        First hit wins; later tiers are not consulted. Worst case for a
        hard miss with the Invoice config is 1 + len(fallback_regions)
        + 3 DocTR calls.
        """
        # Tier 1: primary
        match: OcrMatch | None = self.extract(bgr, primary_regions, regex)
        if match is not None:
            return OcrResult(match=match, rotation_degrees=0)

        # Tier 2: fallback regions
        for region_set in fallback_regions:
            match = self.extract(bgr, region_set, regex)
            if match is not None:
                return OcrResult(match=match, rotation_degrees=0)

        # Tier 3: rotate the input in 90° increments and retry primary.
        if try_rotation:
            for angle in self.ROTATION_ANGLES:
                rotated: BgrImage = np.rot90(bgr, k=angle // 90)
                match = self.extract(rotated, primary_regions, regex)
                if match is not None:
                    return OcrResult(match=match, rotation_degrees=angle)

        return None

    def _recognize(self, bgr_crop: BgrImage) -> str:
        """Run DocTR on a single BGR crop, return concatenated word text."""
        # Small crops (e.g. the top-right header strip) lose glyph
        # detail at native resolution. Upscale before recognition; the
        # detection grid scales accordingly.
        if min(bgr_crop.shape[:2]) < self.UPSCALE_THRESHOLD_PX:
            bgr_crop = cv2.resize(
                bgr_crop, None,
                fx=self.UPSCALE_FACTOR, fy=self.UPSCALE_FACTOR,
                interpolation=cv2.INTER_CUBIC,
            )
        rgb: np.ndarray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        result: Any = _get_doctr_ocr_model()([rgb])
        page: Any = result.pages[0]
        words: list[str] = [
            word.value
            for block in page.blocks
            for line in block.lines
            for word in line.words
        ]
        return " ".join(words)


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

    def find_attachment_by_checksum(
        self,
        record_model: str,
        record_id: int,
        checksum: str,
    ) -> int | None:
        """Phase-2 idempotency primitive.

        Returns the id of an existing ``ir.attachment`` on this record
        whose sha1 checksum (Odoo's own ``checksum`` field) matches the
        bytes the caller is about to upload — or None. Lets
        Pipeline.process skip a redundant ``attach()`` call when a
        previous attempt uploaded but crashed before the local ledger
        recorded success.

        Scope is intentionally per-record so a checksum collision on a
        different invoice doesn't make us treat it as already-uploaded.
        """
        rows: list[dict[str, Any]] = self.search_read(
            "ir.attachment",
            [
                ["res_model", "=", record_model],
                ["res_id", "=", record_id],
                ["checksum", "=", checksum],
            ],
            ["id"],
            limit=1,
        )
        return int(rows[0]["id"]) if rows else None

    def verify_attachment(
        self,
        attachment_id: int,
        expected_res_model: str,
        expected_res_id: int,
        expected_name_contains: str,
    ) -> bool:
        """Phase-3 duplicate-handling primitive.

        Returns True iff the ``ir.attachment`` row with ``attachment_id``
        is still present AND attached to ``(expected_res_model,
        expected_res_id)`` AND its ``name`` field still contains
        ``expected_name_contains`` (typically the OCR'd invoice slug
        like ``"INV-2026-05001"``). Operator-edited names that lose
        the slug surface as False so we re-upload defensively.

        Transport errors are deliberately NOT swallowed — they
        propagate so ``Pipeline.process``'s catchall leaves the file
        in the inbox for retry rather than treating an Odoo outage as
        "attachment is gone".
        """
        rows: list[dict[str, Any]] = self.search_read(
            "ir.attachment",
            [["id", "=", attachment_id]],
            ["res_model", "res_id", "name"],
            limit=1,
        )
        if not rows:
            return False
        row: dict[str, Any] = rows[0]
        return (
            row.get("res_model") == expected_res_model
            and int(row.get("res_id") or 0) == expected_res_id
            and expected_name_contains in (row.get("name") or "")
        )

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

    def archive_duplicate(self, src_path: Path) -> Path:
        """Phase-3 duplicate handler.

        Moves ``src_path`` into ``done/duplicates/<original-name>``,
        intact (no v3 cleanup — the canonical clean copy already lives
        in ``done/INV/...`` from the first processing). On name
        collision, appends a numeric counter so we never overwrite a
        prior duplicate.

        Audit trail beats silent deletion for invoices: even a
        duplicate is a real document the operator may want to see.
        """
        dest_dir: Path = self._root / "duplicates"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path: Path = self._unique_dest(dest_dir, src_path.name)
        # shutil.move handles cross-fs (the inbox bind mount and done/
        # may be on different filesystems).
        shutil.move(str(src_path), str(dest_path))
        return dest_path

    @staticmethod
    def _unique_dest(dest_dir: Path, name: str) -> Path:
        """Return a non-existing path inside ``dest_dir`` based on
        ``name``. First attempt uses the bare name; subsequent
        collisions get a numeric counter inserted before the suffix
        (``foo.jpg`` → ``foo.1.jpg`` → ``foo.2.jpg``)."""
        candidate: Path = dest_dir / name
        if not candidate.exists():
            return candidate
        stem: str = candidate.stem
        suffix: str = candidate.suffix
        n: int = 1
        while True:
            candidate = dest_dir / f"{stem}.{n}{suffix}"
            if not candidate.exists():
                return candidate
            n += 1

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
            recorded_at REAL NOT NULL,
            res_model TEXT NOT NULL DEFAULT '',
            ocr_name TEXT NOT NULL DEFAULT '',
            archive_path TEXT NOT NULL DEFAULT ''
        )
    """

    # Phase-2 columns (added to support verified-duplicate handling) are
    # appended via ALTER TABLE on existing databases so a pre-Phase-2
    # ledger keeps loading without manual migration.
    _PHASE2_COLUMNS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("res_model",    "TEXT NOT NULL DEFAULT ''"),
        ("ocr_name",     "TEXT NOT NULL DEFAULT ''"),
        ("archive_path", "TEXT NOT NULL DEFAULT ''"),
    )

    def __init__(self, path: Path) -> None:
        self._path: Path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock: threading.Lock = threading.Lock()
        self._conn: sqlite3.Connection = sqlite3.connect(
            self._path, check_same_thread=False
        )
        self._conn.execute(self.SCHEMA)
        # Backfill columns introduced by Phase 2 onto pre-existing rows.
        existing_cols: set[str] = {
            row[1] for row in self._conn.execute(
                "PRAGMA table_info(processed)"
            ).fetchall()
        }
        for col_name, col_def in self._PHASE2_COLUMNS:
            if col_name not in existing_cols:
                self._conn.execute(
                    f"ALTER TABLE processed ADD COLUMN {col_name} {col_def}"
                )
        self._conn.commit()

    @staticmethod
    def file_digest(path: Path) -> str:
        h: hashlib._Hash = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(64 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def has(self, digest: str) -> bool:
        """True iff the file has been *successfully* processed before.

        Failure rows are kept for audit / stats but deliberately do NOT
        gate dedup: an infra glitch (Odoo down, OCR cache unwritable,
        etc.) must be retryable just by re-dropping the file. If the
        operator wants to surface a permanent unreadable, they leave it
        in ``done/unreadable/`` — they shouldn't have to perform SQL
        surgery on the ledger to retry.
        """
        with self._lock:
            row: tuple[Any, ...] | None = self._conn.execute(
                "SELECT 1 FROM processed WHERE sha256 = ? AND outcome = 'success'",
                (digest,),
            ).fetchone()
        return row is not None

    def record_success(
        self,
        digest: str,
        source_name: str,
        odoo_id: int,
        attachment_id: int,
        *,
        res_model: str,
        ocr_name: str,
        archive_path: Path,
    ) -> None:
        """Record a successful processing outcome.

        ``res_model``, ``ocr_name``, and ``archive_path`` are persisted so
        that Phase-3 duplicate verification can confirm an Odoo
        attachment is still attached to the right record without having
        to re-OCR the source file.
        """
        self._upsert(
            digest, source_name, "success", odoo_id, attachment_id,
            res_model=res_model, ocr_name=ocr_name,
            archive_path=str(archive_path),
        )

    def record_failure(self, digest: str, source_name: str) -> None:
        self._upsert(
            digest, source_name, "failure", None, None,
            res_model="", ocr_name="", archive_path="",
        )

    def get_success_row(self, digest: str) -> "ProcessedRow | None":
        """Read the success row for ``digest`` (or None for failure /
        missing). Returns the typed view Phase-3 verification needs."""
        with self._lock:
            row: tuple[Any, ...] | None = self._conn.execute(
                """
                SELECT odoo_id, attachment_id, res_model, ocr_name, archive_path
                FROM processed
                WHERE sha256 = ? AND outcome = 'success'
                """,
                (digest,),
            ).fetchone()
        if row is None:
            return None
        odoo_id, attachment_id, res_model, ocr_name, archive_path = row
        return ProcessedRow(
            odoo_id=int(odoo_id),
            attachment_id=int(attachment_id),
            res_model=res_model or "",
            ocr_name=ocr_name or "",
            archive_path=Path(archive_path) if archive_path else Path(),
        )

    def delete(self, digest: str) -> None:
        """Remove the row for ``digest``. Used by Phase-3 when Odoo
        verification reports the attachment is gone — the next
        ``Pipeline.process`` call must reprocess from scratch."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM processed WHERE sha256 = ?", (digest,),
            )
            self._conn.commit()

    def _upsert(
        self,
        digest: str,
        source_name: str,
        outcome: str,
        odoo_id: int | None,
        attachment_id: int | None,
        *,
        res_model: str,
        ocr_name: str,
        archive_path: str,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO processed (
                    sha256, source_name, outcome, odoo_id, attachment_id,
                    recorded_at, res_model, ocr_name, archive_path
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sha256) DO UPDATE SET
                    outcome = excluded.outcome,
                    odoo_id = excluded.odoo_id,
                    attachment_id = excluded.attachment_id,
                    recorded_at = excluded.recorded_at,
                    res_model = excluded.res_model,
                    ocr_name = excluded.ocr_name,
                    archive_path = excluded.archive_path
                """,
                (
                    digest, source_name, outcome, odoo_id, attachment_id,
                    time.time(), res_model, ocr_name, archive_path,
                ),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


@dataclass(frozen=True, slots=True)
class ProcessedRow:
    """Typed view of a ``ProcessedLedger`` success row, produced by
    :meth:`ProcessedLedger.get_success_row`. The ``res_model``,
    ``ocr_name``, and ``archive_path`` fields are empty / Path() for
    rows written before the Phase-2 schema bump."""

    odoo_id: int
    attachment_id: int
    res_model: str
    ocr_name: str
    archive_path: Path


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
        # Cap every thread pool the worker uses so `workers * cv2_threads`
        # stays at or below the box's core count.
        # cv2 reads its own pool size; torch ships with intra-op +
        # inter-op pools that default to one thread per physical core
        # (32 on a 64-core SMT host) and OpenBLAS / MKL / OpenMP under
        # torch each spin their own. Without these caps two workers
        # spawn >120 contending threads and stall under DocTR.
        # Lazy import: torch is only present in environments that have
        # the OCR backend installed; tests that don't touch the OCR
        # path can skip it.
        threads: int = max(1, config.cv2_threads)
        cv2.setNumThreads(threads)
        try:
            import torch  # noqa: PLC0415  — defer heavy import
            torch.set_num_threads(threads)
            torch.set_num_interop_threads(threads)
        except (ImportError, RuntimeError):
            # set_num_interop_threads raises RuntimeError if the
            # interop pool is already initialized (e.g. test re-runs in
            # the same process). Either way, the cap above on
            # set_num_threads still applies.
            pass
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
            ocr_engine=OcrEngine(),
            storage_preparer=StoragePreparer(),
            odoo_client=odoo,
            archiver=archiver,
            ledger=ledger,
            mailer=mailer,
            stats=StatsTracker(inbox / config.statistics_file.lstrip("/")),
        )

    def process(self, source: Path, *, keep_original: bool = False) -> ProcessOutcome:
        # The ledger lookup AND the file_digest call live inside the
        # try block: an unreadable source (PermissionError, ENOENT, …)
        # would otherwise propagate out of the worker callable into the
        # WorkSubmitter's future, which nobody awaits, and the failure
        # would vanish silently. Catch everything here so workers always
        # produce a structured outcome.
        digest: str = ""
        try:
            digest = ProcessedLedger.file_digest(source)
            if self._ledger.has(digest):
                # Phase-3 verified-duplicate handling: confirm the
                # Odoo attachment we recorded is still attached to the
                # right invoice before retiring the duplicate. If
                # Odoo lost it, blow away the stale ledger row and
                # reprocess so we never silently treat a missing
                # upload as already-done.
                row: ProcessedRow | None = self._ledger.get_success_row(digest)
                if row is not None and self._odoo.verify_attachment(
                    attachment_id=row.attachment_id,
                    expected_res_model=row.res_model,
                    expected_res_id=row.odoo_id,
                    expected_name_contains=row.ocr_name.replace("/", "-"),
                ):
                    dup_path: Path = self._archiver.archive_duplicate(source)
                    self._log.info(
                        "duplicate of %s (aid=%s) → %s",
                        row.ocr_name, row.attachment_id, dup_path,
                    )
                    return ProcessOutcome(
                        source, True, row.ocr_name, row.odoo_id,
                        row.attachment_id, dup_path, None, None,
                    )
                # Verify failed (or row is None for a pre-Phase-2 ledger
                # entry without enough info to check): clear the stale
                # ledger row and fall through to reprocessing. Phase-2
                # idempotency at upload time prevents a duplicate
                # attachment in the rare case Odoo actually does still
                # have it.
                self._log.warning(
                    "ledger says %s was processed but Odoo can't confirm — "
                    "reprocessing", source.name,
                )
                self._ledger.delete(digest)
            return self._process_inner(source, digest, keep_original)
        except Exception as e:
            # Catchall = infra failure (network, cache permission, …).
            # The file's own readability / classifiability / OCR success
            # all run inside _process_inner, which calls _handle_failure
            # itself for those file-specific cases (and routes to
            # done/unreadable/). What lands here is "the daemon couldn't
            # try" — leave the file in the inbox so the next
            # initial_sweep on container restart retries it after the
            # operator fixes the underlying issue. No ledger row, no
            # unreadable move, no email storm.
            self._log.exception(
                "infra failure processing %s — leaving in inbox for retry",
                source,
            )
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
        # DocTR works on color BGR directly; no separate binarize step.
        result: OcrResult | None = self._ocr_engine.extract_with_fallbacks(
            cleaned,
            primary_regions=doc_type.config.search_regions,
            fallback_regions=doc_type.config.fallback_search_regions,
            try_rotation=doc_type.config.ocr_try_rotation,
            regex=doc_type.regex,
        )
        if result is None:
            self._handle_failure(source, digest, "OCR did not match regex", bgr=bgr)
            return ProcessOutcome(
                source, False, None, None, None, None, None, "OCR miss"
            )
        ocr: OcrMatch = result.match
        if result.rotation_degrees:
            # Rotation is interesting — log it. The downstream OK line
            # already names the file and the extracted invoice number,
            # so there's no need for a redundant "OCR extracted" line in
            # the no-rotation case.
            self._log.info(
                "rotated %s by %d° to OCR successfully",
                source.name, result.rotation_degrees,
            )
            # Keep storage / archive in the orientation that OCR succeeded
            # at — the office reviewer never sees a sideways scan. Rotate
            # the ORIGINAL bgr (not the OCR-preprocessed cleaned), since
            # v3 storage operates on the original via the layered model.
            bgr = np.rot90(bgr, k=result.rotation_degrees // 90)

        # 4. Odoo lookup
        odoo_id: int | None = self._odoo.find_record(
            doc_type.config.odoo_object, ocr.name
        )
        if odoo_id is None:
            self._handle_failure(
                source, digest, f"no Odoo record for {ocr.name}", bgr=bgr,
            )
            return ProcessOutcome(
                source, False, ocr.name, None, None, None, ocr.region,
                "Odoo record not found",
            )

        # 5. Storage prep — v3 layered decomposer runs against the
        # rotation-corrected ORIGINAL bgr (not the OCR-cleaned version).
        # The substrate-aware extractors handle yellow paper natively;
        # pre-cleanup would damage faint pen ink that the handwriting
        # extractor would otherwise preserve.
        payload: bytes
        mime: str
        payload, mime = self._storage_preparer.prepare(bgr)

        # 6. Odoo attach + link.
        # Idempotency check: if a previous attempt uploaded this exact
        # payload and crashed before record_success, Odoo already has
        # the attachment under the same checksum. Reuse the existing
        # aid instead of creating a duplicate.
        payload_sha1: str = hashlib.sha1(payload).hexdigest()
        existing_aid: int | None = self._odoo.find_attachment_by_checksum(
            doc_type.config.odoo_object, odoo_id, payload_sha1,
        )
        attachment_id: int
        if existing_aid is not None:
            self._log.info(
                "Odoo already has attachment %s for %s (checksum %s) — reusing",
                existing_aid, source.name, payload_sha1[:12],
            )
            attachment_id = existing_aid
        else:
            attachment_id = self._odoo.attach(
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
        self._ledger.record_success(
            digest, source.name, odoo_id, attachment_id,
            res_model=doc_type.config.odoo_object,
            ocr_name=ocr.name,
            archive_path=archive_path,
        )
        self._stats.record(doc_type.name, ocr.region)
        if not keep_original:
            try:
                source.unlink()
            except FileNotFoundError:
                pass
            except OSError as e:
                # Source unlink failed AFTER successful Odoo upload + archive
                # — file is preserved (Odoo aid + archive), only inbox cleanup
                # blocked. NEVER let this become a "failure" outcome: that
                # would trigger the catchall, leave the file in the inbox,
                # and the next sweep would re-upload, creating a duplicate
                # in Odoo. Email the operator with the cleaned PNG attached
                # and the precise paths so they can rm the source by hand.
                # If the email itself fails, log loudly and continue —
                # the success outcome is still correct, the file is still
                # preserved, and the inbox accumulating is the operator's
                # eventual signal that something needs attention.
                self._log.warning(
                    "could not remove processed source %s: %s — file is preserved "
                    "(aid=%s, archive=%s), emailing operator for manual cleanup",
                    source, e, attachment_id, archive_path,
                )
                self._send_cleanup_failure_email(
                    source=source, ocr_name=ocr.name, odoo_id=odoo_id,
                    attachment_id=attachment_id, archive_path=archive_path,
                    payload=payload, mime=mime, error=str(e),
                )

        self._log.info(
            "OK %s id=%s aid=%s → %s", ocr.name, odoo_id, attachment_id, archive_path
        )
        return ProcessOutcome(
            source, True, ocr.name, odoo_id, attachment_id, archive_path, ocr.region, None
        )

    def _send_cleanup_failure_email(
        self, *, source: Path, ocr_name: str, odoo_id: int,
        attachment_id: int, archive_path: Path,
        payload: bytes, mime: str, error: str,
    ) -> None:
        """Operator-facing notification when a source can't be removed
        from the inbox after a successful upload + archive. Reuses the
        Mailer's send_failure entry point so the body, subject, and
        attachment all flow through the same tested code path."""
        if self._mailer is None or not self._config.error_email:
            self._log.warning(
                "no error-email configured; cleanup-failure for %s is logged only",
                source.name,
            )
            return
        ext: str = "jpg" if mime == "image/jpeg" else "png"
        attach_name: str = f"{ocr_name.replace('/', '-')}_{source.stem}.{ext}"
        body: str = (
            f"{self._config.error_mail_message}\n\n"
            f"{ocr_name} was successfully processed and is preserved:\n"
            f"  Odoo attachment id : {attachment_id}\n"
            f"  Odoo record id     : {odoo_id}\n"
            f"  Archive path       : {archive_path}\n\n"
            f"However, the daemon could not remove the original source from the\n"
            f"inbox — please rm it by hand:\n  {source}\n\n"
            f"Underlying error: {error}\n"
        )
        try:
            self._mailer.send_failure(
                self._config.error_email,
                subject=f"[scanrunner] processed {ocr_name} but could not remove source",
                body=body,
                attachment=(attach_name, payload, mime),
            )
            self._log.info(
                "cleanup-failure email sent for %s to %s",
                source.name, self._config.error_email,
            )
        except Exception:
            self._log.exception(
                "cleanup-failure email send failed for %s — file preserved, "
                "inbox cleanup still pending operator action",
                source,
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
        bgr: BgrImage | None = None,
    ) -> None:
        """Route a failed file into ``done/unreadable/`` and (optionally) email.

        If ``bgr`` is supplied (we got far enough into the pipeline that
        the original image is in memory — OCR miss, Odoo miss), we encode
        it once via ``StoragePreparer`` (which runs the full v3 layered
        decomposition) and use those bytes for BOTH the unreadable
        archive and the email attachment. The reviewer sees the same
        clean, rotation-corrected image the office would have got on
        success.

        If ``bgr`` is None (no DocumentType matched, cv2 couldn't read
        the file, generic exception), we fall back to the raw original
        bytes with a magic-sniffed MIME.
        """
        self._log.error("FAIL %s: %s", source.name, error)

        payload: bytes | None = None
        mime: str = "application/octet-stream"
        ext: str = ""
        if bgr is not None:
            payload, mime = self._storage_preparer.prepare(bgr)
            ext = "jpg" if mime == "image/jpeg" else "png"

        # Snapshot raw bytes BEFORE any move, used as fallback for the
        # email attachment. Source is still in the inbox at this point.
        attach: tuple[str, bytes, str] | None = None
        if payload is not None:
            attach = (f"{source.stem}.{ext}", payload, mime)
        elif source.exists():
            try:
                sniffed: str = magic.from_file(str(source), mime=True) or "application/octet-stream"
            except Exception:
                sniffed = "application/octet-stream"
            try:
                attach = (source.name, source.read_bytes(), sniffed)
            except OSError:
                attach = None

        # Email-then-move ordering: notification is the precondition
        # for any destructive bookkeeping. If the email fails (or can't
        # be attempted because there's no mailer / no error_email
        # configured), leave the source in the inbox and skip the
        # ledger record so the next initial_sweep on container restart
        # retries it once the operator fixes the SMTP outage. The
        # failure folder accumulating orphan files nobody knows about
        # is a worse outcome than the inbox getting visibly stuck.
        notified: bool = False
        if self._mailer is not None and self._config.error_email and attach is not None:
            try:
                self._mailer.send_failure(
                    self._config.error_email,
                    subject=f"[scanrunner] could not process {source.name}",
                    body=f"{self._config.error_mail_message}\n\n{error}",
                    attachment=attach,
                )
                self._log.info(
                    "failure email sent for %s to %s",
                    source.name, self._config.error_email,
                )
                notified = True
            except Exception:
                self._log.exception(
                    "failure email send failed for %s — leaving in inbox for retry",
                    source,
                )
                return
        elif self._mailer is None or not self._config.error_email:
            # No mailer configured — operator opted into silent failures;
            # proceed with move + ledger.
            notified = True

        if not notified:
            return

        try:
            self._archiver.archive_unreadable(source, payload, ext)
        except Exception:
            self._log.exception("could not move %s to unreadable/", source)
        self._ledger.record_failure(digest, source.name)


# -----------------------------------------------------------------------------
# Daemon — watchdog + ProcessPoolExecutor + signal-aware shutdown.
# -----------------------------------------------------------------------------


_WORKER_LOG_FORMAT: str = (
    "%(asctime)s %(processName)s %(levelname)s %(name)s: %(message)s"
)


def _worker_init_logging() -> None:
    """ProcessPoolExecutor initializer: configure stderr logging.

    forkserver workers start with a fresh interpreter and Python's
    default WARNING-only root config, so per-file outcome lines emitted
    by ``scanrunner.pipeline`` (the most useful operational signal)
    would otherwise be silently dropped from container logs. Wire each
    worker to the same stderr / format the daemon uses, and silence
    the same wire-level chatter (httpx / httpcore / urllib3).

    Idempotent: ``logging.basicConfig`` is a no-op if root already has
    handlers, which it normally won't in a freshly-started worker.
    """
    logging.basicConfig(level=logging.INFO, format=_WORKER_LOG_FORMAT)
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


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
        # Each worker runs _worker_init_logging on startup so its INFO
        # logs reach stderr (and from there the container log driver).
        ctx: Any = get_context("forkserver")
        self._pool: ProcessPoolExecutor = ProcessPoolExecutor(
            max_workers=self._max_workers,
            mp_context=ctx,
            initializer=_worker_init_logging,
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
                # Per-file noise — the daemon emits a single summary
                # ("initial sweep submitted N file(s)") right after this
                # method returns. Dropping per-file submit logs to DEBUG
                # so production INFO logs stay tractable.
                self._log.debug("initial-sweep submit: %s", path.name)
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
        logging.basicConfig(level=log_level, format=_WORKER_LOG_FORMAT)
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
        sub.add_parser(
            "warm-models",
            help=(
                "Force-download and load the DocTR OCR predictor into the "
                "configured DOCTR_CACHE_DIR. Used at container build time so "
                "the first invoice in a freshly-started worker doesn't pay "
                "the ~150 MB / 10–30 s model-download cliff."
            ),
        )
        ns: argparse.Namespace = parser.parse_args(argv)
        log_level: int = logging.DEBUG if getattr(ns, "verbose", False) else logging.INFO
        match ns.cmd:
            case "daemon":
                return cls(ns.config, ns.server, ns.inbox, log_level).run()
            case "process":
                logging.basicConfig(level=log_level, format="%(message)s")
                pipeline: Pipeline = Pipeline.for_worker(ns.config, ns.server, ns.inbox)
                outcome: ProcessOutcome = pipeline.process(ns.path, keep_original=ns.keep)
                return 0 if outcome.success else 1
            case "warm-models":
                logging.basicConfig(level=log_level, format=_WORKER_LOG_FORMAT)
                _get_doctr_ocr_model()
                logging.getLogger("scanrunner.warm").info(
                    "DocTR OCR predictor loaded; weights cached"
                )
                return 0
            case other:
                raise ValueError(f"unknown subcommand {other}")


if __name__ == "__main__":
    raise SystemExit(Daemon.cli(sys.argv[1:]))
