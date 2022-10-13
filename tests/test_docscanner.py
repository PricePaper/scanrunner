#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from unittest import TestCase

from docscanner import *


class TestInvoice(TestCase):

    def setUp(self) -> None:
        self.test_file = "INV-2022-12412_20221006_Customer_Invoice_011671.jpg"
        self.test_invoice_name = "INV/2022/12412"

    def test_construction(self):
        assert Invoice(self.test_file)
        self.invoice = Invoice(self.test_file)
        self.assertEqual(self.invoice.file, Path(self.test_file))
        self.assertEqual(self.invoice.filename, self.test_file)
        self.assertEqual(self.invoice.threshold_region_ignore, 80)
        self.assertTrue(self.invoice.regex.search(self.test_invoice_name))

    def test__mark_region(self):
        pass

    def test__read_text(self):
        invoice = Invoice(self.test_file)
        image, line_items_coordinates = invoice._mark_region()
        self.assertEqual(invoice._read_text(image, line_items_coordinates, -6), "/2022/12412")

    def test__read(self):
        pass

    def test_name(self):
        pass

    def test_name_setter(self):
        pass

    def test_reset(self):
        pass

    def test_threshold_region_ignore(self):
        pass
