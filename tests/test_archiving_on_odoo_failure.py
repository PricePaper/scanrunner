#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from unittest import TestCase

import docscanner
from docscanner import DocumentImage, FileManager


class TestArchivingOnOdooFailure(TestCase):
    def setUp(self) -> None:
        # Use test configuration and prevent deletion of fixtures
        self.config = docscanner.get_configuration("./test_config.yaml", "development", debug=True, stats=False)
        self.config['keep_original'] = True
        # Pick a real sample image from new_invoices
        self.samples_dir = Path("./new_invoices").resolve()
        self.sample = next(self.samples_dir.glob("*.jpg"), None)
        if self.sample is None:
            self.skipTest("No sample images in new_invoices to run archiving tests")
        self.fm = FileManager(self.config)

    def test_routes_to_unreadable_when_odoo_save_failed_even_with_name(self):
        doc = DocumentImage(self.config, self.sample)
        # Trigger processing to generate outputs and OCR name if possible
        _ = doc.name  # may or may not succeed; we will set a name if missing
        if not doc._name:
            # Provide a plausible name to simulate successful OCR
            doc._name = "INV/2025/13001"
        # Simulate Odoo failure: IDs remain zero and no email was sent yet
        doc.odoo_id = 0
        doc.odoo_attachment_id = 0
        doc.is_emailed = False

        result_path = Path(self.fm.done(doc))
        # Expect it to be routed to the unreadable folder, storing the B&W PNG if available
        self.assertIn("/done/unreadable", str(result_path))
        self.assertTrue(result_path.exists(), "Unreadable archive target does not exist")
        # Clean up the moved file to keep the repo tidy
        try:
            result_path.unlink(missing_ok=True)
        except Exception:
            pass

    def test_routes_to_unreadable_when_email_was_sent(self):
        doc = DocumentImage(self.config, self.sample)
        _ = doc.name
        if not doc._name:
            doc._name = "INV/2025/13002"
        # Simulate that a failure email was already sent
        doc.is_emailed = True
        # Odoo ids may or may not be present; email flag alone should force unreadable
        result_path = Path(self.fm.done(doc))
        self.assertIn("/done/unreadable", str(result_path))
        self.assertTrue(result_path.exists(), "Unreadable archive target does not exist")
        # Clean up
        try:
            result_path.unlink(missing_ok=True)
        except Exception:
            pass
