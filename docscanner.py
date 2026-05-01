#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import base64
import logging
import os
import re
import smtplib
import ssl
import sys
import typing
import xmlrpc.client
from email.message import EmailMessage
from pathlib import Path
from smtplib import SMTP
from time import sleep
from typing import Any

try:
    import psutil
except ImportError:
    print("The psutil module is not installed.", sys.stderr)
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("The opencv-python module is not installed.", sys.stderr)
    sys.exit(1)

try:
    import pytesseract
except ImportError:
    print("The pytesseract module is not installed.", sys.stderr)
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("The PyYAML module is not installed.", sys.stderr)
    sys.exit(1)

try:
    import magic
except ImportError:
    print("The python-magic module is not installed.", sys.stderr)
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("The Pillow module is not installed.", sys.stderr)
    sys.exit(1)


class DocumentImage:

    def __init__(self, config: dict, file: object):
        """
        Class for all document images being processed by OCR
        :param config configuration from YAML file
        :type config: dict
        :param file: The file to be processed
        :type file: object
        """

        # if we're passed a string, convert to Path
        self.file: Path = file if type(file) == Path else Path(file)
        # Keep the original input path for later cleanup/move decisions
        self.original_file: Path = Path(self.file)

        if not self.file.exists():
            raise FileNotFoundError

        self.mime_type: str = ""
        self._name: str = ""
        self._document_type = ""
        self.odoo_id: int = 0
        self.odoo_attachment_id: int = 0
        self.odoo_document_id: int = 0
        self.is_emailed: bool = False
        self._error_mail_message: str = ""
        self._odoo_sequence: str = ""
        self._threshold_region_ignore: int = 0
        self.regex: re.Pattern = re.compile("")
        self._regions_list: list[list[int]] = []
        self.ocr_top_half_only: bool = False
        self.tesseract_config: str = "--psm 6 -l eng"
        self.search_regions: list[list[int]] = []
        # Output controls
        self.storage_png_compress_level: int = 6  # used for Odoo/Email B&W PNG
        self.keep_intermediate_ocr_image: bool = False
        self.ocr_whitelist: str | None = None
        # Odoo B&W rendering controls
        self.odoo_storage_bw_method: str = "adaptive_gaussian"  # adaptive_mean|adaptive_gaussian|background_subtract
        self.odoo_bw_use_clahe: bool = True
        self.odoo_bw_denoise_h: int = 10
        self.odoo_bw_dilate_kernel: int = 7
        self.odoo_bw_median_ksize: int = 21
        self.odoo_bw_threshold: int = 210
        self.odoo_bw_open_kernel: int = 2
        self.odoo_bw_open_iterations: int = 1
        self.odoo_bw_post_dilate_kernel: int = 0  # 0 to disable
        # Path for produced Odoo/Email output
        self.odoo_storage_path: Path | None = None

        self.config = config
        self.logger = config['logger']

        if self.document_type:
            doc_cfg = config['documents'][self.document_type]
            self._odoo_sequence = doc_cfg.get('odoo_sequence', '')
            self._threshold_region_ignore = doc_cfg.get('threshold_region_ignore', 80)
            self.regex = re.compile(doc_cfg['ocr_regex'])
            # Backwards compatibility: support old 'regions' while preferring 'search_regions'
            self.search_regions = doc_cfg.get('search_regions', doc_cfg.get('regions', []))
            self.ocr_top_half_only = bool(doc_cfg.get('ocr_top_half_only', False))
            self.tesseract_config = doc_cfg.get('tesseract_config', self.tesseract_config)
            # output options
            self.storage_png_compress_level = int(
                doc_cfg.get('storage_png_compress_level', self.storage_png_compress_level))
            self.keep_intermediate_ocr_image = bool(
                doc_cfg.get('keep_intermediate_ocr_image', self.keep_intermediate_ocr_image))
            self.ocr_whitelist = doc_cfg.get('ocr_whitelist', self.ocr_whitelist)
            # Odoo B&W rendering options
            self.odoo_storage_bw_method = doc_cfg.get('odoo_storage_bw_method', self.odoo_storage_bw_method)
            self.odoo_bw_use_clahe = bool(doc_cfg.get('odoo_bw_use_clahe', self.odoo_bw_use_clahe))
            self.odoo_bw_denoise_h = int(doc_cfg.get('odoo_bw_denoise_h', self.odoo_bw_denoise_h))
            self.odoo_bw_dilate_kernel = int(doc_cfg.get('odoo_bw_dilate_kernel', self.odoo_bw_dilate_kernel))
            self.odoo_bw_median_ksize = int(doc_cfg.get('odoo_bw_median_ksize', self.odoo_bw_median_ksize))
            self.odoo_bw_threshold = int(doc_cfg.get('odoo_bw_threshold', self.odoo_bw_threshold))
            self.odoo_bw_open_kernel = int(doc_cfg.get('odoo_bw_open_kernel', self.odoo_bw_open_kernel))
            self.odoo_bw_open_iterations = int(doc_cfg.get('odoo_bw_open_iterations', self.odoo_bw_open_iterations))
            self.odoo_bw_post_dilate_kernel = int(
                doc_cfg.get('odoo_bw_post_dilate_kernel', self.odoo_bw_post_dilate_kernel))

    @property
    def filename(self) -> str:
        """
        Returns the name of the file
        :return: file name
        :rtype: str
        """
        return str(self.file)

    @filename.setter
    def filename(self, filename: str) -> None:
        """
        Sets the file name
        :param filename:
        :type filename: str
        :return:
        :rtype: None
        """
        self.file = Path(filename)

    @property
    def odoo_sequence(self) -> str:
        """
        The sequence string Odoo uses to preface document numbers of this type
        :return: Odoo document sequence
        :rtype: str
        """

        return self._odoo_sequence

    @odoo_sequence.setter
    def odoo_sequence(self, value: str) -> None:
        """
        This method is not implemented
        :param value: Ignored
        :type value: str
        :return:
        :rtype: NotImplementedError
        """

        raise NotImplementedError("This field can not be set. Please modify the config.yaml instead.")

    @property
    def error_mail_message(self) -> str:
        """
        The error message to be sent when an error occurs during OCR processing.
        :return: Error message
        :rtype: str
        """
        if self._error_mail_message:
            return self._error_mail_message

        self._error_mail_message = self.config['error-mail-message']
        return self._error_mail_message

    @error_mail_message.setter
    def error_mail_message(self, message: str) -> None:
        self._error_mail_message = message

    def _read(self) -> str:
        """
        Run OCR over configured regions to extract the document number.
        Also produces two output images:
        - Odoo/Email: high-quality B&W PNG suitable for printing
        - Local archive: color image (e.g., WEBP) preserving original look
        """
        document_str: str = ''

        # Load original (color)
        color = cv2.imread(self.filename)
        if color is None:
            return document_str

        # Prepare OCR image in-memory (do not overwrite file)
        ocr_img = self._preprocess_for_ocr(color)

        # Produce Odoo (B&W PNG) output
        try:
            self.odoo_storage_path = self._save_odoo_storage_bw_image(color)
        except Exception as e:
            self.logger.warning(f"Failed to save Odoo B&W image: {e}")
            self.odoo_storage_path = None

        # Prepare OCR base (top-half if configured)
        ocr_base = ocr_img.copy()
        h, w = ocr_base.shape[:2]
        if self.ocr_top_half_only:
            ocr_base = ocr_base[0:int(h * 0.5), :]

        # Iterate configured percent-based regions; if none provided, fall back to whole top area
        regions = self.search_regions or [[60, 0, 100, 55]]

        for idx, rect in enumerate(regions, start=1):
            x1p, y1p, x2p, y2p = rect
            oh, ow = ocr_base.shape[:2]
            x1 = max(0, min(ow - 1, int(ow * (x1p / 100.0))))
            y1 = max(0, min(oh - 1, int(oh * (y1p / 100.0))))
            x2 = max(0, min(ow, int(ow * (x2p / 100.0))))
            y2 = max(0, min(oh, int(oh * (y2p / 100.0))))
            if x2 <= x1 or y2 <= y1:
                continue
            roi = ocr_base[y1:y2, x1:x2]

            # If ROI is tiny, skip
            rh, rw = roi.shape[:2]
            if rh < self.threshold_region_ignore or rw < self.threshold_region_ignore:
                continue

            # Slightly upscale small ROIs to help OCR
            scale = 1.0
            if max(rh, rw) < 300:
                scale = 1.5
            if scale != 1.0:
                roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            # OCR the region
            try:
                tess_cfg = self.tesseract_config
                if self.ocr_whitelist:
                    tess_cfg = f"{tess_cfg} -c tessedit_char_whitelist={self.ocr_whitelist}"
                txt = pytesseract.image_to_string(roi, config=tess_cfg) or ''
                t = txt.replace('\n', ' ')
                self.logger.debug(f'Reading {self.filename} region% {rect} result: {t}')
                m: re.Match | None = self.regex.search(t)
                if m:
                    document_str = m.group(1)
                    if 'statistics' in self.config:
                        count: int = self.config['statistics'][self.document_type].setdefault(str(rect), 0) + 1
                        self.config['statistics'][self.document_type][str(rect)] = count
                        self.logger.debug(f'Region% {rect} found {document_str} in document string: {t}')
                    return document_str
            except Exception as e:
                self.logger.debug(f"OCR error for region {rect}: {e}")
                continue

        return document_str

    def _preprocess_fullpage_bw(self, color_bgr) -> Any:
        """
        Legacy: aggressive binarization used previously for both OCR and storage.
        Kept for reference. New code uses `_preprocess_for_ocr` for OCR only.
        """
        lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        gray = cv2.fastNlMeansDenoising(gray, h=7, templateWindowSize=7, searchWindowSize=21)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        bw = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 12)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        clean = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
        return clean

    def _save_odoo_storage_bw_image(self, color_bgr) -> Path:
        """
        Produce a printable, denoised black & white PNG at original dimensions and return its path.
        Selectable methods via config:
        - adaptive_mean / adaptive_gaussian
        - background_subtract (legacy-like)
        - background_otsu (robust illumination normalization + Otsu)
        Includes guardrails to auto-correct polarity and fall back if result is unusable.
        """
        # Optional CLAHE on L channel to normalize illumination
        if self.odoo_bw_use_clahe:
            lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            base_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        else:
            base_bgr = color_bgr

        gray = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)

        def background_otsu(img_gray: Any) -> Any:
            # Fast denoise
            gdn = cv2.fastNlMeansDenoising(img_gray, h=max(0, int(self.odoo_bw_denoise_h)), templateWindowSize=7,
                                           searchWindowSize=21)
            # Estimate smooth background via large median blur on a lightly dilated image
            dil_k = max(1, int(self.odoo_bw_dilate_kernel))
            dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_k, dil_k))
            dil = cv2.dilate(gdn, dil_kernel)
            med_k = int(self.odoo_bw_median_ksize)
            if med_k % 2 == 0:
                med_k += 1
            bg = cv2.medianBlur(dil, med_k)
            # Divide (or subtract) to flatten illumination, then blur slightly
            # Avoid divide-by-zero by offsetting background
            bg = cv2.max(bg, 1)
            norm = cv2.divide(gdn, bg, scale=255)
            norm_blur = cv2.GaussianBlur(norm, (3, 3), 0)
            # Otsu threshold
            _, bw_local = cv2.threshold(norm_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return bw_local

        method = (self.odoo_storage_bw_method or "adaptive_gaussian").lower()

        # Produce candidate bw according to method
        if method == "background_subtract":
            h = max(0, int(self.odoo_bw_denoise_h))
            dilate_k = max(1, int(self.odoo_bw_dilate_kernel))
            median_k = int(self.odoo_bw_median_ksize)
            if median_k % 2 == 0:
                median_k += 1
            thresh_val = int(self.odoo_bw_threshold)
            denoised = cv2.fastNlMeansDenoising(gray, h=h, templateWindowSize=7, searchWindowSize=21)
            dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
            dilated = cv2.dilate(denoised, dil_kernel)
            bg = cv2.medianBlur(dilated, median_k)
            diff = 255 - cv2.absdiff(denoised, bg)
            _, bw = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
        elif method == "adaptive_mean" or method == "adaptive_gaussian":
            gray_dn = cv2.fastNlMeansDenoising(gray, h=7, templateWindowSize=7, searchWindowSize=21)
            blur = cv2.GaussianBlur(gray_dn, (3, 3), 0)
            adapt_method = cv2.ADAPTIVE_THRESH_MEAN_C if method == "adaptive_mean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            C = 10 if method == "adaptive_mean" else 12
            bw = cv2.adaptiveThreshold(blur, 255, adapt_method, cv2.THRESH_BINARY, 31, C)
        elif method == "background_otsu":
            bw = background_otsu(gray)
        else:
            # Fallback to adaptive_gaussian if unknown
            gray_dn = cv2.fastNlMeansDenoising(gray, h=7, templateWindowSize=7, searchWindowSize=21)
            blur = cv2.GaussianBlur(gray_dn, (3, 3), 0)
            bw = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 12)

        # Morphological cleanups
        if self.odoo_bw_open_kernel and self.odoo_bw_open_kernel > 0:
            ok = int(self.odoo_bw_open_kernel)
            open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ok, ok))
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, open_kernel, iterations=int(self.odoo_bw_open_iterations))
        if self.odoo_bw_post_dilate_kernel and self.odoo_bw_post_dilate_kernel > 0:
            pk = int(self.odoo_bw_post_dilate_kernel)
            post_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (pk, pk))
            bw = cv2.dilate(bw, post_kernel, iterations=1)

        # Guardrails: ensure correct polarity (white background) and usable ink ratio
        hgt, wdt = bw.shape[:2]
        # Infer polarity by sampling 10px frame borders: expect white background
        frame = 10 if min(hgt, wdt) >= 40 else 2
        top = bw[0:frame, :]
        bottom = bw[-frame:, :]
        left = bw[:, 0:frame]
        right = bw[:, -frame:]
        border_black = ((top == 0).sum() + (bottom == 0).sum() + (left == 0).sum() + (right == 0).sum()) / (
        (top.size + bottom.size + left.size + right.size))
        if border_black > 0.25:
            bw = cv2.bitwise_not(bw)
        # Compute black pixel ratio
        black_ratio = float((bw == 0).sum()) / float(bw.size)
        # If too sparse or too dense, try a safer fallback (adaptive_gaussian)
        if black_ratio < 0.003 or black_ratio > 0.75:
            try:
                gray_dn = cv2.fastNlMeansDenoising(gray, h=7, templateWindowSize=7, searchWindowSize=21)
                blur = cv2.GaussianBlur(gray_dn, (3, 3), 0)
                bw_fb = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 12)
                # re-run polarity check
                top = bw_fb[0:frame, :];
                bottom = bw_fb[-frame:, :];
                left = bw_fb[:, 0:frame];
                right = bw_fb[:, -frame:]
                border_black_fb = ((top == 0).sum() + (bottom == 0).sum() + (left == 0).sum() + (right == 0).sum()) / (
                (top.size + bottom.size + left.size + right.size))
                if border_black_fb > 0.25:
                    bw_fb = cv2.bitwise_not(bw_fb)
                bw = bw_fb
            except Exception as _:
                pass

        # Convert to 1-bit PIL and save as PNG
        pil_img = Image.fromarray(bw)
        pil_1 = pil_img.point(lambda p: 255 if p > 127 else 0, mode='1')
        out_path = self.file.with_suffix('.png')
        compress_level = int(self.storage_png_compress_level)
        pil_1.save(out_path, format='PNG', optimize=True, compress_level=compress_level)
        return out_path

    def _preprocess_for_ocr(self, color_bgr) -> Any:
        """
        OCR-focused preprocessing that reduces yellow background artifacts and produces a clean binary image.
        Steps:
        - LAB CLAHE on L channel for contrast
        - Convert to grayscale
        - Denoise (fastNlMeans) and slight blur
        - Adaptive threshold (MEAN or GAUSSIAN)
        - Morphological open and light dilation to thicken strokes
        Returns uint8 0/255 single-channel image.
        """
        lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        gray = cv2.fastNlMeansDenoising(gray, h=7, templateWindowSize=7, searchWindowSize=21)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        bw = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 31, 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        clean = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
        # Light dilation to connect broken characters
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        clean = cv2.dilate(clean, kernel2, iterations=1)
        return clean

    def _save_processed_over_original(self, bw_img) -> None:
        """
        Deprecated: previously saved OCR-binarized image over original.
        Left to avoid breaking external references but not used anymore.
        """
        if bw_img.ndim == 3:
            bw_img = cv2.cvtColor(bw_img, cv2.COLOR_BGR2GRAY)
        pil_img = Image.fromarray(bw_img)
        pil_1 = pil_img.point(lambda p: 255 if p > 127 else 0, mode='1')
        out_path = self.file.with_suffix('.png')
        pil_1.save(out_path, format='PNG', optimize=True)
        self.file = out_path


    @property
    def document_type(self) -> str:
        """
        Using the file name, and mime type from the config file, this method determines the
        correct document type to be used for further processing of the file
        :return: the document type as defined in the configuration file
        :rtype: str
        """

        if self._document_type:
            return self._document_type

        self.mime_type = magic.from_file(self.filename, mime=True)

        for document, values in self.config['documents'].items():

            if self.file.match(values['file-name-match']):
                if self.mime_type in values['mime-types']:
                    self.logger.debug(f"File: {self.filename} mime-type: {self.mime_type} document-type: {document}")
                    self._document_type = document
        return self._document_type

    @document_type.setter
    def document_type(self, document_type: str) -> None:
        """
        Sets the document type. This would not normally be used.
        :param document_type:
        :type document_type: str
        :return: None
        :rtype: None
        """
        self._document_type = document_type

    @property
    def name(self) -> str:
        """
        The title of the document. E.g. the invoice number, picking number, etc. This method is a lazy load, if the
        value is not set, it will call the _read() method to get the value, then store it in the object.
        :return: the document's title
        :rtype: str
        """

        while self._name == "" and self.threshold_region_ignore >= self.config['documents'][self.document_type][
            'threshold_region_ignore_min']:
            name = self._read()

            # If we still don't have a name, increase sensitivity and try again
            if name:
                self._name = self.odoo_sequence + name
            else:
                self.threshold_region_ignore -= self.config['documents'][self.document_type][
                    'threshold_region_ignore_decrement']
                self.logger.debug(
                    f"{self.filename} can not be parsed. Changing OCR sensitivity {self.threshold_region_ignore + self.config['documents'][self.document_type]['threshold_region_ignore_decrement']} -> {self.threshold_region_ignore}.")

        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """
        Not implement
        :param value:
        :type value: str
        :return:
        :rtype:
        """
        raise NotImplementedError("This field can not be set. Try reset() to clear it.")

    def reset(self) -> None:
        """
        Resets the objects name property back to an empty string. Calling the objects
        name getter will reread the file.
        :return: None
        :rtype: None
        """

        self._name = ""

    @property
    def threshold_region_ignore(self) -> int:
        """
        For OpenCV
        :return: current value of the DocumentImage
        :rtype: int
        """
        return self._threshold_region_ignore

    @threshold_region_ignore.setter
    def threshold_region_ignore(self, threshold_region_ignore: int) -> None:
        """
        Sets the value for OpenCV processing of DocumentImage
        :param threshold_region_ignore:
        :type threshold_region_ignore: int
        :return:
        :rtype: None
        """
        self._threshold_region_ignore = threshold_region_ignore
        self.reset()

    def _mark_region(self):
        """
        This method finds and defines regions in the image file using opencv2. Once the regions are identified, we can
        feed them to tesseract for OCR.

        :return: None
        :rtype: None
        """

        image = cv2.imread(self.filename)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 30)

        # Dilate to combine adjacent text contours
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        dilate = cv2.dilate(thresh, kernel, iterations=4)

        # Find contours, highlight text areas, and extract ROIs
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        line_items_coordinates: list[list[tuple[Any, Any]]] = []
        for c in cnts:
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)

            if w < self.threshold_region_ignore or h < self.threshold_region_ignore:
                continue

            image = cv2.rectangle(image, (x, y), (x + w, y + h), color=(255, 0, 255), thickness=3)
            line_items_coordinates.append([(x, y), (x + w, y + h)])

        return image, line_items_coordinates

    def _read_text(self, image, line_items_coordinates, index) -> str:
        # get co-ordinates to crop the image
        c = line_items_coordinates[index]

        # cropping image img = image[y0:y1, x0:x1]
        img = image[c[0][1]:c[1][1], c[0][0]:c[1][0]]

        # convert the image to black and white for better OCR
        ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)

        # pytesseract image to string to get results
        text = str(pytesseract.image_to_string(thresh1, config='--psm 6'))
        return text


class OdooConnector:

    def __init__(self, configuration: dict) -> None:
        """
        Connection Handler to communicate with Odoo
        :param configuration:
        :type configuration:
        """
        self.config: dict = configuration
        self.url: str = self.config['url']
        self.db: str = self.config['database']
        self.username: str = self.config['username']
        self.password: str = self.config['password']
        self.logger: logging.Logger = configuration['logger']

        self._uid = 0

    def _get_uid(self):

        try:
            self._uid = self.config['uid']

        except KeyError:

            with xmlrpc.client.ServerProxy(f"{self.url}/xmlrpc/2/common", allow_none=True, verbose=self.config['debug'],
                                           context=ssl._create_unverified_context()) as common:
                self._uid = common.authenticate(self.db, self.username, self.password, {})

                self.config['uid'] = self._uid

        return self._uid

    @property
    def uid(self) -> int:
        """
        Odoo's user id for the username
        :return: user id
        :rtype: int
        """
        return self._uid or self._get_uid()

    @uid.setter
    def uid(self, user_id: int) -> None:
        """
        Set the Odoo uid for the connection
        :param user_id: Odoo User ID
        :type user_id: int
        :return:
        :rtype: None
        """
        self._uid = user_id

    def odoo_document_id(self, document: DocumentImage) -> int:
        """
        Gets the documents Odoo ID by searching for the documents name in Odoo
        :param document:
        :type document: DocumentImage
        :return: document's ID
        :rtype: int
        """
        retry: int = 0
        odoo_id: int = 0

        while odoo_id == 0 and retry < self.config['retry']:

            # If we do not have a document name, we can't search Odoo so bail out
            if not document.name:
                self.logger.warning(f"Unable to read document name from {document.filename}. SKIPPING")
                return odoo_id
            try:
                with xmlrpc.client.ServerProxy(f'{self.url}/xmlrpc/2/object', allow_none=True,
                                               context=ssl._create_unverified_context()) as models:
                    res = models.execute_kw(self.db, self.uid, self.password,
                                            self.config['documents'][document.document_type]['odoo_object'],
                                            'search_read', [[['name', '=', document.name]]], {'fields': ['id', 'name']})
                    # If we get an id, set it in the document
                    if type(res) == list and res[0]['name'] == document.name:
                        odoo_id = res[0]['id']
                        self.logger.debug(
                            f'File: {document.filename} Name: {document.name} has an Odoo ID of {odoo_id}')
                        document.odoo_id = odoo_id
                    else:
                        break

            except IndexError as e:
                message = f"This document {document.name} can not be found in Odoo. Please attach it manually."
                document._error_mail_message = message
                self.logger.warning(message)
                self.logger.exception(e)
                break

            except Exception as e:
                odoo_id = 0
                retry += 1
                self.logger.exception(
                    f"There was a problem getting the document id from Odoo. Retry {retry}/{self.config['retry']}")
                sleep(self.config['retry_sleep'])

        return odoo_id

    def save_document(self, document: DocumentImage) -> int:
        """
        Saves the document to Odoo by creating an attachment
        :param document:
        :type document: DocumentImage
        :return: The attachment ID
        :rtype: int
        """
        retry: int = 0
        while retry < self.config['retry']:
            try:
                # Make sure we have a document id from Odoo, if not, get one
                if document.odoo_id or self.odoo_document_id(document):
                    with xmlrpc.client.ServerProxy(f'{self.url}/xmlrpc/2/object', allow_none=True,
                                                   context=ssl._create_unverified_context()) as models:
                        # Choose the Odoo B&W PNG if available; otherwise fall back to original
                        upload_path = document.odoo_storage_path if document.odoo_storage_path else document.file
                        with upload_path.open('rb') as f:
                            data = base64.b64encode(f.read())
                            values = {
                                'name': document.name.replace('/', '-') + '_' + Path(upload_path).name.replace('/',
                                                                                                               '-'),
                                'res_id': document.odoo_id,
                                'res_model': self.config['documents'][document.document_type]['odoo_object'],
                                'attachment_tag_id': self.config['documents'][document.document_type][
                                    'odoo_attachment_tag_id'],
                                'datas': data.decode('ascii')}
                            document.odoo_attachment_id = models.execute_kw(self.db, self.uid, self.password,
                                                                            'ir.attachment', 'create', [values, ])
                            # From ir.attachment, we create an Odoo document
                            doc_values = {
                                'attachment_id': document.odoo_attachment_id,
                                'folder_id': self.config['documents'][document.document_type]['odoo_folder_id'],
                                'active': True,
                            }
                            document.odoo_document_id = models.execute_kw(self.db, self.uid, self.password,
                                                                          'documents.document', 'create',
                                                                          [doc_values, ])

                            return document.odoo_document_id
                else:
                    self.logger.error(
                        f'Save failed: document {document.name} from file: {document.filename} can not be saved in Odoo.')
                    break
            except Exception as e:
                retry += 1
                self.logger.warning(e)
                self.logger.exception(
                    f"There was a problem saving the attachment in Odoo. Retry {retry}/{self.config['retry']}")
                sleep(self.config['retry_sleep'])

        return 0


class FileManager:

    def __init__(self, config: dict) -> None:
        """
        This class processes filenames given to its process_files() method

        :param config: configuration data from the YAML config file returned by get_configuration()
        :type config: dict
        """

        self.config = config
        self.logger: logging.Logger = config['logger']

    def _get_paths_from_string(self, path_string: str) -> typing.List[Path]:
        """
        Takes a string and figures out how to make Path objects from it. Handles directories as well as file globs.
        :param path_string: directory, file or file glob as string
        :return: List[Path]
        """
        path = Path(path_string)
        paths: typing.List[Path] = []

        try:
            # Glob returns a generator. If the generator throws a ValueError, we don't have a file glob
            paths.extend([f for f in path.parent.glob(path.name) if f.is_file()])

            # if that worked, we know it's an iterable, and therefore a valid file glob
            return paths

        except ValueError:
            self.logger.debug(f"{path_string} is not a file glob")

        if path.is_dir():
            self.logger.debug(f"{path_string} is a directory")
            paths = [file for file in path.iterdir() if file.is_file()]
        elif path.is_file():
            self.logger.debug(f"{path_string} is a regular file")
            paths.append(path)
        else:
            self.logger.warning(f"{path_string} is not a directory, regular file or file glob. Ignoring.")

        return paths

    def _is_generated_bw_png(self, p: Path) -> bool:
        """
        Heuristic to detect our own generated 1-bit B&W PNGs left next to a source JPEG.
        Skip these to prevent duplicate processing.
        Conditions:
        - file suffix is .png
        - image mode is '1' (1-bit) when opened via PIL
        - a sibling JPEG/JPG with the same stem exists in the same directory
        """
        try:
            if p.suffix.lower() != '.png':
                return False
            # Quick sibling check first to avoid opening large files unnecessarily
            has_jpeg_sibling = p.with_suffix('.jpg').exists() or p.with_suffix('.jpeg').exists()
            if not has_jpeg_sibling:
                return False
            with Image.open(p) as im:
                return im.mode == '1'
        except Exception:
            return False

    def document_generator(self, paths: typing.List[str]) -> typing.Generator[DocumentImage, None, None]:
        """
        Returns a generator that yields a DocumentImage object for (hopefully) each file
        or file glob passed

        :param paths: a list of files or file globs to process
        :type paths: list[str]
        :return: generator for DocumentImage(s)
        :rtype: typing.Generator[DocumentImage]
        """

        files_list: typing.List[Path] = []

        for path in paths:
            files_list.extend(self._get_paths_from_string(path))

        # Filter out our own generated B&W PNGs to avoid duplicates
        filtered = []
        for f in files_list:
            if self._is_generated_bw_png(f):
                self.logger.debug(f"Skipping generated B&W PNG {f} to avoid duplicate processing")
                continue
            filtered.append(f)

        for document in filtered:
            try:
                yield DocumentImage(self.config, document)
            except Exception as e:
                self.logger.exception(e)
                self.logger.warning(f"Unable to parse file {document}. IGNORING.")
                continue

    def done(self, document: DocumentImage) -> str:
        """
        Relocates file that has been processed to storage directory
        For successful reads, we store the local color archival image (WEBP by default).
        For unreadable-but-emailed cases, we store under 'unreadable'.
        :param document:
        :type document: DocumentImage
        :return: New file location
        :rtype: str
        """

        done_top_dir: Path = Path(f"{document.file.parent}/{self.config['done-path']}")

        # Route to unreadable if emailing occurred OR Odoo save failed (even if OCR succeeded)
        if document.is_emailed or not document.odoo_id or not document.odoo_attachment_id:
            done_path: Path = done_top_dir.joinpath("unreadable")
            done_path.mkdir(exist_ok=True, parents=True)
            # store the B&W PNG (odoo_storage_path) if available, else original
            src_path = document.odoo_storage_path if document.odoo_storage_path else document.file
            target = done_path.joinpath(Path(src_path).name)
            if Path(src_path) != target:
                document.file = Path(src_path).replace(target)
            else:
                document.file = target
            self.logger.warning(f"Moved unreadable file -> {document.filename}")
            # Remove the original input file unless --keep was specified, and only if it still exists
            try:
                if not self.config.get('keep_original', False):
                    if document.original_file.exists() and document.original_file.resolve() != document.file.resolve():
                        document.original_file.unlink(missing_ok=True)
                        self.logger.debug(f"Deleted original file {document.original_file}")
            except Exception as e:
                self.logger.warning(f"Could not delete original file {document.original_file}: {e}")
            return document.filename

        doc_type: str
        doc_year: str
        doc_number: str

        # If we can't split/unpack name, we should leave the method
        try:
            doc_type, doc_year, doc_number = document.name.split('/')
        except:
            self.logger.warning(f"No appropriate document name for {document.filename}. Can not be safely moved.")
            return ""

        # Group documents by 100 to ease speed file access
        doc_grouping: int = int(doc_number) // 100

        done_path = done_top_dir.joinpath(f"{doc_type}/{doc_year}/{doc_grouping:02}00")

        # Make the directory if it doesn't exist, including parent directories
        done_path.mkdir(exist_ok=True, parents=True)

        if not document.odoo_id or not document.odoo_attachment_id:
            self.logger.warning(f"Document {document.name} file:{document.filename} is not saved to Odoo")

        # Move the original input file to archive location (no reformatting)
        src_path = document.original_file
        new_file_name = f"{document.name.replace('/', '-')}_id-{document.odoo_id}_aid-{document.odoo_attachment_id}_{Path(src_path).name}"

        self.logger.debug(f"Targeting {new_file_name} for file {document.filename}")

        document.file = Path(src_path).replace(done_path.joinpath(new_file_name))

        self.logger.info(f"Moved {document.name} -> {document.filename}")

        # Remove the original input file unless --keep was specified, and only if it still exists
        try:
            if not self.config.get('keep_original', False):
                if document.original_file.exists() and document.original_file.resolve() != document.file.resolve():
                    document.original_file.unlink(missing_ok=True)
                    self.logger.debug(f"Deleted original file {document.original_file}")
        except Exception as e:
            self.logger.warning(f"Could not delete original file {document.original_file}: {e}")

        # Also delete any generated Odoo B&W PNG left beside the source to prevent reprocessing
        try:
            if document.odoo_storage_path and Path(document.odoo_storage_path).exists():
                od_png = Path(document.odoo_storage_path)
                # Only delete if it still sits in the original source directory
                if od_png.parent.resolve() == document.original_file.parent.resolve():
                    od_png.unlink(missing_ok=True)
                    self.logger.debug(f"Deleted transient Odoo PNG {od_png}")
                # Clear the reference to avoid accidental reuse
                document.odoo_storage_path = None
        except Exception as e:
            self.logger.warning(f"Could not delete transient Odoo PNG {document.odoo_storage_path}: {e}")

        return document.filename

    # def _environ_or_required(key):


#     """Helper to ensure args are set or an ENV variable is present"""
#     if os.environ.get(key):
#         return {'default': os.environ.get(key)}
#     else:
#         return {'required': True}
class MailSender:

    def __init__(self, configuration: dict) -> None:
        """
        Class to email Documents that can not be parsed

        :param configuration: The configuration, loaded from the YAML file
        :type configuration: dict
        """

        self.config: dict = configuration
        self.logger = configuration['logger']

    def mail_document(self, document: DocumentImage) -> None:
        """
        Mails a DocumentImage to the email specified in the config file
        Attachment must be the printable B&W PNG (same as Odoo upload).
        :param document:
        :type document: DocumentImage
        :return:
        :rtype: None
        """
        retry: int = 0
        while retry < self.config['retry']:
            try:

                # Create the message
                msg = EmailMessage()
                msg['Subject'] = f'Document failed to scan: {document.filename}'
                msg['To'] = self.config['error-email']
                msg['From'] = f"Document Scanner <{self.config['smtp-user']}>"
                msg.preamble = "A MIME aware email client is required to view this email properly.\n"

                msg.set_content(document._error_mail_message)

                # Choose the B&W PNG (odoo_storage_path) if available, else fallback to original
                attach_path = document.odoo_storage_path if document.odoo_storage_path else document.file
                # Determine MIME type from the selected file
                mime_type: str = magic.from_file(str(attach_path), mime=True)
                mime_maintype, mime_subtype = mime_type.split('/', 1)
                with Path(attach_path).open('rb') as fp:
                    msg.add_attachment(fp.read(), maintype=mime_maintype, subtype=mime_subtype,
                                       filename=str(attach_path))

                # Email the message
                with SMTP(host=self.config['smtp-server'], port=self.config['smtp-port']) as smtp:
                    if self.config['smtp-use-tls']:
                        smtp.starttls()
                    smtp.ehlo_or_helo_if_needed()
                    smtp.login(self.config['smtp-user'], self.config['smtp-password'])
                    smtp.send_message(msg)

                    self.logger.info(f"Emailed failed document {attach_path} to {self.config['error-email']}")
                    document.is_emailed = True

                    # If we've gotten this far, the mail sent, and we can short circuit the retry loop
                    return

            except smtplib.SMTPException as e:
                self.logger.error(e)
                retry += 1
                sleep(self.config['retry_sleep'])


def _parse_args():
    # Get configuration from environmental variables or command line
    parser = argparse.ArgumentParser(description="Script to read scanned documents and send them to Odoo")

    try:
        parser.add_argument('-s', '--server', dest='server', default=os.environ.get("DS_SERVER", 'production'),
                            help="The server configuration to use from the config file")
        parser.add_argument('-c', '--config', dest='config_file',
                            default=os.environ.get("DS_CONFIG", "/etc/docscanner.conf"),
                            help="The path to the YAML configuration file. Defaults to /etc/docscanner.conf")
        parser.add_argument('-v', '--verbose', dest='debug', action='store_true', help="enable verbose output")
        parser.add_argument('--stats', action='store_true', help="store region statistics in file")
        parser.add_argument('--keep', action='store_true',
                            help="Preserve original input files in their original location")
        parser.add_argument('file', type=str, nargs='+',
                            help="The file, files or directories to process. Can be more than one. (required)")

        return parser.parse_args()

    except Exception as e:
        logger = logging.getLogger()
        logger.exception("Exception while parsing command line arguments.")
        sys.exit(1)


def get_configuration(config_file_name: str, server: str = "development", debug: bool = False,
                      stats: bool = False) -> dict:
    """
    The configuration read from a YAML file and various other housekeeping such as setting up logging.

    :param debug: Turns on debug logging
    :type debug: bool
    :param config_file: the configuration file, either as pathlib.Path or str
    :type config_file: object
    :param server: the server configuration to use from the [servers] section
    :type server: str
    :return: dictionary of configuration settings
    :rtype: dict
    """

    config_file: Path = Path(config_file_name)

    with config_file.open() as f:
        # global config
        config: dict = yaml.safe_load(f)

        # add some convenience keys, so we don't have to read the server config everywhere
        config['server']: str = server

        for config_key, config_value in config['servers'][server].items():
            config[config_key] = config_value

        # Keep track of debug
        config['debug'] = debug

        # Set up statistics, if needed
        if stats:
            stats_file = Path(config['statistics-file'])
            statistics: dict
            if stats_file.exists():
                statistics = yaml.safe_load(stats_file.read_text()) or {}

                for doc_type in config['documents']:
                    if not statistics.setdefault(doc_type, 0):
                        statistics[doc_type] = {}
            else:
                statistics = {doc_type: {} for doc_type in config['documents']}
                with stats_file.open('w') as f:
                    yaml.safe_dump(statistics, f)
            config['statistics'] = statistics

        # Get root logger
        logger = logging.getLogger()
        # create console handler with a higher log level
        console_handler = logging.StreamHandler()
        if debug:
            logger.setLevel(logging.DEBUG)
            console_handler.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
            console_handler.setLevel(logging.INFO)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        # add the handlers to logger

        logger.addHandler(console_handler)
        config['logger'] = logger

        return config


def main():
    args = _parse_args()
    config = get_configuration(args.config_file, args.server, args.debug, args.stats)
    # Propagate --keep flag into configuration for downstream logic
    config['keep_original'] = bool(getattr(args, 'keep', False))

    logger = config['logger']

    # get path to tesseract from config
    pytesseract.pytesseract.tesseract_cmd = config['tesseract-bin']

    # Improve OCR by increasing threads to max cpus minus one
    try:
        os.environ['OMP_THREAD_LIMIT'] = str((len(psutil.Process().cpu_affinity()) - 1) or 1)

    except AttributeError:
        os.environ['OMP_THREAD_LIMIT'] = str((psutil.cpu_count() - 1) or 1)

    # get our manager objects
    file_manager: FileManager = FileManager(config)
    odoo: OdooConnector = OdooConnector(config)
    mailer: MailSender = MailSender(config)

    # Get the documents
    documents: typing.Generator[DocumentImage, None, None] = file_manager.document_generator(args.file)

    for document in documents:
        if document.document_type:
            if odoo.save_document(document):
                logger.info(f"Saved {document.name} ID: {document.odoo_id} document:{document.odoo_document_id} to odoo server: {args.server}")
            else:
                logger.error(f"Unable to process file: {document.filename}. Mailing to {config['error-email']}")
                mailer.mail_document(document)

            file_manager.done(document)
    # Save statistics before we exit
    if args.stats:
        with open(config['statistics-file'], 'w') as f:
            yaml.safe_dump(config['statistics'], f)


if __name__ == "__main__":
    main()
