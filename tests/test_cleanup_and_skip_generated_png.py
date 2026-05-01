#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from unittest import TestCase

from PIL import Image as PILImage

import docscanner
from docscanner import DocumentImage, FileManager


class TestCleanupAndSkipGeneratedPNG(TestCase):
    def setUp(self) -> None:
        self.config = docscanner.get_configuration("./test_config.yaml", "development", debug=True, stats=False)
        # Keep fixtures intact; we explicitly control cleanup of artifacts we create
        self.config['keep_original'] = True
        self.samples_dir = Path("./new_invoices").resolve()
        self.sample = next(self.samples_dir.glob("*.jpg"), None)
        if self.sample is None:
            self.skipTest("No sample images in new_invoices to run cleanup/skip tests")
        self.fm = FileManager(self.config)

    def tearDown(self) -> None:
        # Remove any transient files we might have created during tests (only *.png with same stem)
        if self.sample is not None:
            png = self.sample.with_suffix('.png')
            try:
                png.unlink(missing_ok=True)
            except Exception:
                pass

    def test_success_cleanup_removes_odoo_png(self):
        # Generate outputs by triggering name computation
        doc = DocumentImage(self.config, self.sample)
        _ = doc.name
        # Ensure a B&W PNG exists next to the source
        bw_path = Path(doc.odoo_storage_path) if doc.odoo_storage_path else None
        self.assertIsNotNone(bw_path, "Expected Odoo B&W PNG to be created")
        self.assertTrue(bw_path.exists(), "Expected Odoo B&W PNG file to exist")

        # Simulate successful Odoo save (IDs non-zero) and archive
        doc.odoo_id = 123
        doc.odoo_attachment_id = 456
        result_path = Path(self.fm.done(doc))
        self.assertTrue(result_path.exists(), "Archived file was not created")

        # After success, the transient B&W PNG should be deleted from the source folder
        self.assertFalse(bw_path.exists(), "Transient B&W PNG should be removed to avoid duplicate processing")

        # Cleanup archived file to keep repo tidy
        try:
            result_path.unlink(missing_ok=True)
        except Exception:
            pass

    def test_document_generator_skips_generated_bw_png(self):
        # Create a fake 1-bit PNG next to the sample (same stem) to simulate leftover B&W file
        png_path = self.sample.with_suffix('.png')
        # Create a tiny 1-bit image but ensure it's saved at the same size for realism
        with PILImage.open(self.sample) as im:
            bw = im.convert('1')
            bw.save(png_path, format='PNG', optimize=True)
        self.assertTrue(png_path.exists(), "Failed to create test 1-bit PNG")

        # Now run the document generator against the directory; it should skip the generated PNG
        inputs = [str(self.samples_dir)]
        yielded = list(self.fm.document_generator(inputs))
        yielded_paths = [Path(d.filename) for d in yielded]
        # Ensure none of the yielded paths are the generated PNG
        self.assertNotIn(png_path, yielded_paths, "Generator should skip our own generated 1-bit PNGs")

        # Clean up the fake PNG
        try:
            png_path.unlink(missing_ok=True)
        except Exception:
            pass

    def test_unreadable_moves_odoo_png_and_no_leftover(self):
        doc = DocumentImage(self.config, self.sample)
        _ = doc.name
        bw_path = Path(doc.odoo_storage_path) if doc.odoo_storage_path else None
        self.assertIsNotNone(bw_path, "Expected Odoo B&W PNG to be created")
        self.assertTrue(bw_path.exists(), "Expected Odoo B&W PNG file to exist")

        # Simulate failure: no Odoo IDs and email sent
        doc.odoo_id = 0
        doc.odoo_attachment_id = 0
        doc.is_emailed = True
        unreadable_target = Path(self.fm.done(doc))
        self.assertIn("/done/unreadable", str(unreadable_target))
        self.assertTrue(unreadable_target.exists(), "Unreadable archive target does not exist")
        # The original B&W PNG should no longer exist in the source folder (it was moved)
        self.assertFalse(bw_path.exists(), "B&W PNG should have been moved out of the source folder")

        # Cleanup moved file
        try:
            unreadable_target.unlink(missing_ok=True)
        except Exception:
            pass
