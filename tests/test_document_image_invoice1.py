#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from unittest import TestCase

import docscanner
from docscanner import *


class TestGetConfiguration(TestCase):

    def setUp(self) -> None:
        self.config_file: Path = Path("./test_config.yaml")
        self.config_file_name = str(self.config_file)
        self.server = "development"

    def test_function_get_configuration(self) -> None:
        config = get_configuration(self.config_file_name, self.server)

        self.assertEqual("ppt-apps15-20221028", config['db'])
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
        self.test_invoice_file = "1-Customer_Invoice-INV-2022-11528.jpg"
        self.test_invoice_name = "INV/2022/11528"
        self.config = docscanner.get_configuration("./test_config.yaml", "development", True)

    def test_construction(self):
        self.assertIsInstance(FileManager(self.config), FileManager)

    def test_get_paths_from_string(self):

        fmgr: FileManager = FileManager(self.config)

        arg1: str = "."
        arg2: str = "./1-Customer_Invoice-INV-2022-11528.jpg"
        arg3: str = "*.jpg"
        arg4: str = "*.foo"

        self.assertEqual([f for f in Path(arg1).iterdir() if f.is_file()], fmgr._get_paths_from_string(arg1))
        self.assertEqual([Path(arg2), ], fmgr._get_paths_from_string(arg2))
        self.assertEqual([f for f in Path(arg3).parent.glob("*.jpg") if f.is_file()], fmgr._get_paths_from_string(arg3))

        # Test logger.warning
        fmgr._get_paths_from_string(arg4)

    def test_document_generator_files(self):
        mgr = FileManager(self.config)

        files: [str] = ["./1-Customer_Invoice-INV-2022-11528.jpg",
                        "2-Customer_Invoice-INV-2022-10515-2.jpg"]

        names: [str] = ["INV/2022/11528",
                        "INV/2022/10515"]

        documents = mgr.document_generator(files)

        i: int = 0
        for doc in documents:
            self.assertEqual(doc.file, Path(files[i]))
            self.assertEqual(doc.name, names[i])
            i += 1

    def test_document_generator_glob(self):
        mgr = FileManager(self.config)

        glob: [str] = ["*.png", "*.jpg"]

        names: [str] = ["INV/2022/11528", "INV/2022/11528", "INV/2022/10515"]

        documents = mgr.document_generator(glob)

        i: int = 0
        for doc in documents:
            self.assertEqual(doc.name, names[i])
            i += 1

    def test_done(self):
        """
        Testing moving a Document to its done location
        :return: None
        :rtype: None
        """
        mgr = FileManager(self.config)

        document = DocumentImage(self.config, self.test_invoice_file)
        original_file = document.file

        print(f"{self.shortDescription()} original file:{document.filename}")

        mgr.done(document)

        self.assertTrue(document.file.exists())
        print(f"{self.shortDescription()} done file:{document.filename}")

        # reset file move
        document.file.replace(original_file)

class TestMailer(TestCase):
    def setUp(self) -> None:
        self.bad_test_invoice_file1 = "bad_Customer_Invoice1.jpg"
        self.bad_test_invoice_file2 = "bad_Customer_Invoice2.jpg"
        self.config = docscanner.get_configuration("./test_config.yaml", "development", True)

        self.bad_doc1 = DocumentImage(self.config, self.bad_test_invoice_file1)
        self.bad_doc2 = DocumentImage(self.config, self.bad_test_invoice_file2)

    def test_constructor(self):
        mail = MailSender(self.config)

        #Make sure we can read the config
        self.assertEqual(mail.config['smtp-server'], "smtp.gmail.com")
        self.assertEqual(mail.config['error-email'], "ean@pricepaper.com")

    def test_send_documents(self):

        # package up the bad docs as a list
        bad_docs: list = [self.bad_doc1, self.bad_doc2]

        mail = MailSender(self.config)

        mail.mail_documents(bad_docs)


class TestInvoice(TestCase):

    def setUp(self) -> None:
        self.config = docscanner.get_configuration("./test_config.yaml", "development")
        self.test_invoice_file = "1-Customer_Invoice-INV-2022-11528.jpg"
        self.test_invoice_name = "INV/2022/11528"
        self.odoo_id = 100

    def test_construction_with_invoice(self) -> None:
        """
        Testing the construction of a DocumentImage with an invoice file
        :return: None
        :rtype: None
        """
        assert DocumentImage(self.config, self.test_invoice_file)
        self.invoice = DocumentImage(self.config, self.test_invoice_file)
        self.assertEqual(self.invoice.file, Path(self.test_invoice_file))
        self.assertEqual(self.invoice.filename, self.test_invoice_file)
        self.assertEqual(self.invoice.document_type, "Invoice")
        self.assertEqual(self.invoice.threshold_region_ignore, 80)
        self.assertTrue(self.invoice.regex.search(self.test_invoice_name))
        self.assertListEqual(self.invoice._regions_list, [[1, 2, 3, 4], [5, 6, 7, 8, 9]])
        print(self.shortDescription())

    def test_get_document_type(self):
        document = DocumentImage(self.config, self.test_invoice_file)

        doc_type: str = document.document_type

        self.assertEqual(doc_type, "Invoice")

    def test__mark_region(self):
        pass

    def test__read_text(self) -> None:
        """
        Testing the private method DocumentImage._read_text
        :return: None
        :rtype: None
        """
        print(self.shortDescription())
        invoice: DocumentImage = DocumentImage(self.config, self.test_invoice_file)
        image, line_items_coordinates = invoice._mark_region()

        # Read the 6th region from the top and compare to what it should be
        self.assertEqual(invoice._read_text(image, line_items_coordinates, -7),
                         f"Draft Invoice {self.test_invoice_name}\n")

    def test__read(self):
        """
        Testing the private method DocumentImage._read
        :return: None
        :rtype: None
        """
        print(self.shortDescription())
        invoice = DocumentImage(self.config, self.test_invoice_file)
        self.assertEqual(self.test_invoice_name.lstrip('INV'), invoice._read())

    def test_name(self):
        """
       Testing the public accessor DocumentImage.name
       :return: None
       :rtype: None
       """
        print(self.shortDescription())
        invoice = DocumentImage(self.config, self.test_invoice_file)
        self.assertEqual(self.test_invoice_name, invoice.name)

    def test_name_setter(self):
        """
        Testing the public setter DocumentImage.name. Should throw an exception
        :return: None
        :rtype: None
        """
        print(self.shortDescription())
        invoice = DocumentImage(self.config, self.test_invoice_file)
        with self.assertRaises(NotImplementedError):
            invoice.name = "Test"

    def test_reset(self):
        """
         Testing the public setter DocumentImage.name. Should throw an exception
         :return: None
         :rtype: None
         """
        print(self.shortDescription())
        invoice = DocumentImage(self.config, self.test_invoice_file)
        self.assertEqual(invoice.name, self.test_invoice_name)
        invoice.reset()
        self.assertEqual(invoice._name, "")

        # Calling .name again re-reads the document image
        self.assertEqual(invoice.name, self.test_invoice_name)

    def test_odoo_sequence(self):
        invoice = DocumentImage(self.config, self.test_invoice_file)
        self.assertEqual("INV", invoice.odoo_sequence)

    def test_odoo_id(self):
        invoice = DocumentImage(self.config, self.test_invoice_file)
        self.assertEqual(invoice.odoo_id, 0)
