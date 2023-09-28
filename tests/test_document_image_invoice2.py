#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from unittest import TestCase

from docscanner import *


class TestInvoice2(TestCase):

    def setUp(self) -> None:
        self.config_file: Path = Path("./test_config.yaml")
        self.config_file_name = str(self.config_file)
        self.server = "development"

        self.test_invoice_file = "documents/INV-2022-10515-2.jpg"
        self.test_invoice_name = "INV/2022/10515"

    def test_construction_with_invoice(self) -> None:
        """
        Testing the construction of a DocumentImage with an invoice file
        :return: None
        :rtype: None
        """
        assert DocumentImage(self.test_invoice_file)
        self.invoice = DocumentImage(self.test_invoice_file)
        self.assertEqual(self.invoice.file, Path(self.test_invoice_file))
        self.assertEqual(self.invoice.filename, self.test_invoice_file)
        self.assertEqual(self.invoice.document_type, "Invoice")
        self.assertEqual(self.invoice.threshold_region_ignore, 80)
        self.assertTrue(self.invoice.regex.search(self.test_invoice_name))
        self.assertListEqual(self.invoice._regions_list, [[1, 2, 3, 4], [5, 6, 7, 8]])
        print(self.shortDescription())

    def test__mark_region(self):
        pass

    def test__read_text(self) -> None:
        """
        Testing the private method DocumentImage._read_text
        :return: None
        :rtype: None
        """
        print(self.shortDescription())
        invoice: DocumentImage = DocumentImage(self.test_invoice_file)
        image, line_items_coordinates = invoice._mark_region()

        # Read the 6th region from the top and compare to what it should be
        self.assertEqual(invoice._read_text(image, line_items_coordinates, -6), "Draft Invoice INV/2022/12412\n")


    def test__read(self):
        """
        Testing the private method DocumentImage._read
        :return: None
        :rtype: None
        """
        print(self.shortDescription())
        invoice = DocumentImage(self.test_invoice_file)
        self.assertEqual(self.test_invoice_name.lstrip('INV'),invoice._read())


    def test_name(self):
        """
       Testing the public accessor DocumentImage.name
       :return: None
       :rtype: None
       """
        print(self.shortDescription())
        invoice = DocumentImage(self.test_invoice_file)
        self.assertEqual(self.test_invoice_name, invoice.name)

    def test_name_setter(self):
        """
        Testing the public setter DocumentImage.name. Should throw an exception
        :return: None
        :rtype: None
        """
        print(self.shortDescription())
        invoice = DocumentImage(self.test_invoice_file)
        with self.assertRaises(NotImplementedError):
            invoice.name = "Test"

    def test_reset(self):
        """
         Testing the public setter DocumentImage.name. Should throw an exception
         :return: None
         :rtype: None
         """
        print(self.shortDescription())
        invoice = DocumentImage(self.test_invoice_file)
        self.assertEqual(invoice.name, self.test_invoice_name)
        invoice.reset()
        self.assertEqual(invoice._name, "")

        # Calling .name again re-reads the document image
        self.assertEqual(invoice.name, self.test_invoice_name)



    def test_odoo_sequence(self):
        invoice = DocumentImage(self.test_invoice_file)
        self.assertEqual("INV", invoice.odoo_sequence)
