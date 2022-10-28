#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from unittest import TestCase

import docscanner
from docscanner import *

docscanner.console_handler.setLevel(logging.DEBUG)


class TestGetConfiguration(TestCase):

    def setUp(self) -> None:
        self.config_file: Path = Path("./test_config.yaml")
        self.config_file_name = str(self.config_file)
        self.server = "development"

    def test_function_get_configuration(self) -> None:
        config = get_configuration(self.config_file_name, self.server)

        self.assertEqual("ppt-apps15-20220920", config['db'])
        self.assertEqual("/usr/bin/tesseract", config['tesseract-bin'])
        self.assertEqual("account.move", config['documents']['Invoice']['odoo_object'])


class TestOdooConnector(TestCase):
    def setUp(self) -> None:
        self.config = docscanner.get_configuration("./test_config.yaml", "development")
        self.test_invoice_file = "1-Customer_Invoice-INV-2022-11528.jpg"
        self.test_invoice_name = "INV/2022/11528"

    def test_constructor(self):
        conn = OdooConnector(self.config)

    def test_get_uid(self):
        conn = OdooConnector(self.config)
        self.assertTrue(conn.uid, 39)

    def test_odoo_document_id(self):
        conn = OdooConnector(self.config)
        document = DocumentImage(self.config, self.test_invoice_file)

        self.assertEqual(conn.odoo_document_id(document), 267485)
        self.assertEqual(document.odoo_id, 267485)

    def test_save_document(self):
        conn = OdooConnector(self.config)
        document = DocumentImage(self.config, self.test_invoice_file)

        doc_id = conn.save_document(document)

        self.assertGreater(doc_id, 0, "Odoo did not return an ID, doc did not save.")


class TestFileManager(TestCase):

    def setUp(self) -> None:
        self.config = docscanner.get_configuration("./test_config.yaml", "development")

    def test_construction(self):
        self.assertIsInstance(FileManager(self.config), FileManager)

    def test_document_generator_files(self):
        mgr = FileManager(self.config)

        files: [str] = ["./1-Customer_Invoice-INV-2022-11528.jpg",
                        "2-Customer_Invoice-INV-2022-10515-2.jpg"]

        documents = mgr.document_generator(files)


class TestInvoice(TestCase):

    def setUp(self) -> None:
        self.test_invoice_file = "1-Customer_Invoice-INV-2022-11528.jpg"
        self.test_invoice_name = "INV/2022/11528"
        self.odoo_id = 100

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
        self.assertEqual(self.test_invoice_name.lstrip('INV'), invoice._read())

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

    def test_odoo_id(self):
        invoice = DocumentImage(self.test_invoice_file)
        self.assertEqual(self.odoo_id, invoice.odoo_id)
