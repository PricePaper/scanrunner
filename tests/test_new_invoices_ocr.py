#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from unittest import TestCase

import docscanner
from docscanner import DocumentImage


class TestNewInvoicesOCR(TestCase):

    def setUp(self) -> None:
        self.config = docscanner.get_configuration("./test_config.yaml", "development", debug=True, stats=True)
        self.samples_dir = Path("./new_invoices").resolve()
        # Only first pages are required to contain invoice number
        self.first_pages = sorted(self.samples_dir.glob("*.jpg"))

    def test_can_read_invoice_number_top_right(self):
        # Ensure we have sample files
        self.assertTrue(len(self.first_pages) > 0, "No sample first pages found in new_invoices")
        ok = 0
        total = 0
        for img in self.first_pages:
            total += 1
            doc = DocumentImage(self.config, img)
            name = doc.name
            if name and self.config['documents'][doc.document_type]['ocr_regex']:
                ok += 1
        # Expect a very high hit rate with new pipeline
        self.assertGreaterEqual(ok / max(total, 1), 0.99, f"Hit rate below expectation: {ok}/{total}")

    def test_odoo_storage_is_bw_png_printable(self):
        # pick one sample
        if not self.first_pages:
            self.skipTest("No sample images")
        img = self.first_pages[0]
        from PIL import Image as PILImage
        orig_path = Path(img)
        with PILImage.open(orig_path) as im:
            orig_w, orig_h = im.size
        doc = DocumentImage(self.config, img)
        _ = doc.name
        self.assertIsNotNone(doc.odoo_storage_path, "Odoo B&W image was not created")
        bw_path = Path(doc.odoo_storage_path)
        self.assertEqual(bw_path.suffix.lower(), ".png")
        with PILImage.open(bw_path) as out_im:
            out_w, out_h = out_im.size
            self.assertEqual((out_w, out_h), (orig_w, orig_h))
            # Should be 1-bit mode
            self.assertEqual(out_im.mode, "1")
