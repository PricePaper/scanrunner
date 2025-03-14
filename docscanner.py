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

        if not self.file.exists():
            raise FileNotFoundError

        self.mime_type: str = ""
        self._name: str = ""
        self._document_type = ""
        self.odoo_id: int = 0
        self.odoo_attachment_id: int = 0
        self.odoo_document_id: int = 0
        self.is_emailed: bool = False
        self._odoo_sequence: str = ""
        self._threshold_region_ignore: int = 0
        self.regex: re.Pattern = re.compile("")
        self._regions_list: list[list[int]] = []

        self.config = config
        self.logger = config['logger']

        if self.document_type:
            self._odoo_sequence = config['documents'][self.document_type]['odoo_sequence']
            self._threshold_region_ignore = config['documents'][self.document_type]['threshold_region_ignore']
            self.regex = re.compile(config['documents'][self.document_type]['ocr_regex'])
            self._regions_list = config['documents'][self.document_type]['regions']

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

    def _read(self) -> str:
        """
        This method must be implemented by the child class as each document type has a different format.
        :return: None
        :rtype: None
        """

        document_str: str = ''
        image, line_items_coordinates = self._mark_region()

        # the invoice number usually lives in regions -1 to -3
        for regions in self._regions_list:
            for i in regions:
                try:
                    t: str = self._read_text(image, line_items_coordinates, -i).replace('\n', ' ')
                    self.logger.debug(f'Reading {self.filename} region: {i} result: {t}')
                    m: re.Match = self.regex.search(t)
                    document_str = m.group(1)
                    if 'statistics' in self.config:
                        count: int = self.config['statistics'][self.document_type].setdefault(i, 0) + 1
                        self.config['statistics'][self.document_type][i] = count
                        self.logger.debug(f'Region: {i} found {document_str} in document string: {t}')

                    return document_str

                except Exception:
                    continue

        return document_str

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
            except Exception as e:
                odoo_id = 0
                retry += 1
                self.logger.warning(e)
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
                        # Open file and send to Odoo to create ir.attachment
                        with document.file.open('rb') as f:
                            data = base64.b64encode(f.read())
                            values = {
                                'name': document.name.replace('/', '-') + '_' + document.filename.replace('/', '-'),
                                'res_id': document.odoo_id,
                                'res_model': self.config['documents'][document.document_type]['odoo_object'],
                                'attachment_tag_id': self.config['documents'][document.document_type]['odoo_attachment_tag_id'],
                                'datas': data.decode('ascii')}
                            document.odoo_attachment_id = models.execute_kw(self.db, self.uid, self.password,
                                                                            'ir.attachment', 'create', [values, ])
                            # From ir.attachment, we create an Odoo document
                            doc_values = {
                                'attachment_id': document.odoo_attachment_id,
                                'folder_id': self.config['documents'][document.document_type]['odoo_folder_id'],
                                'active': True,
                            }
                            document.odoo_document_id = models.execute_kw(self.db, self.uid, self.password, 'documents.document', 'create', [doc_values, ])

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

        for document in files_list:
            try:
                yield DocumentImage(self.config, document)
            except Exception as e:
               self.logger.warning(f"Unable to parse file {document}. IGNORING.")
               continue

    def done(self, document: DocumentImage) -> str:
        """
        Relocates file that has been processed to storage directory
        :param document:
        :type document: DocumentImage
        :return: New file location
        :rtype: str
        """

        done_top_dir: Path = Path(f"{document.file.parent}/{self.config['done-path']}")

        # If this document wasn't read, but has been emailed, short circuit
        if not document.name and document.is_emailed:
            done_path: Path = done_top_dir.joinpath("unreadable")
            done_path.mkdir(exist_ok=True, parents=True)
            document.file = document.file.replace(done_path.joinpath(str(document.file)))
            self.logger.warning(f"Moved unreadable file -> {document.filename}")
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

        new_file_name = f"{document.name.replace('/', '-')}_id-{document.odoo_id}_aid-{document.odoo_attachment_id}_{document.filename}"

        self.logger.debug(f"Targeting {new_file_name} for file {document.filename}")

        document.file = document.file.replace(done_path.joinpath(new_file_name))

        self.logger.info(f"Moved {document.name} -> {document.filename}")

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

                msg.set_content(self.config['error-mail-message'])

                # Add the files
                mime_maintype: str
                mime_subtype: str
                mime_type: str = document.mime_type if document.mime_type is not None or "" else "application/octet-stream"
                mime_maintype, mime_subtype = mime_type.split('/', 1)
                with document.file.open('rb') as fp:
                    msg.add_attachment(fp.read(), maintype=mime_maintype, subtype=mime_subtype,
                                       filename=document.filename)

                # Email the message
                with SMTP(host=self.config['smtp-server'], port=self.config['smtp-port']) as smtp:
                    if self.config['smtp-use-tls']:
                        smtp.starttls()
                    smtp.ehlo_or_helo_if_needed()
                    smtp.login(self.config['smtp-user'], self.config['smtp-password'])
                    smtp.send_message(msg)

                    self.logger.info(f"Emailed failed document {document.filename} to {self.config['error-email']}")
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
