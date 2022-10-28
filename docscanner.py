#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import base64
import logging
import os
import re
import ssl
import sys
import xmlrpc.client
from pathlib import Path
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

# Set up logging
logger = logging.getLogger()

# create console handler with a higher log level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
# add the handlers to logger
logger.addHandler(console_handler)


class DocumentImage:

    def __init__(self, config: dict, file: object):
        """
        Class for all document images being processed by OCR
        :param file: The file to be processed
        :type file: Path or string that will be converted to a Path object
        """

        # if we're passed a string, convert to Path
        self.file: Path = file if type(file) == Path else Path(file)
        self._name: str = ""
        self.document_type = None
        self.odoo_id: int = 0
        self.config = config

        for fileglob in config['document_type'].keys():
            if self.file.match(fileglob):
                self.document_type = config['document_type'][fileglob]

        # self.document_type:str = [config['file_type'][fileglob] for
        #                   fileglob in config['document_type']
        #                   if self.file.match(config['document_type'][fileglob])
        #                   ][0]

        self._odoo_sequence: str = config['documents'][self.document_type]['odoo_sequence']
        self._threshold_region_ignore: int = config['documents'][self.document_type]['threshold_region_ignore']
        self.regex: re.Pattern = re.compile(config['documents'][self.document_type]['ocr_regex'])
        self._regions_list: list[list[int]] = config['documents'][self.document_type]['regions']

    @property
    def filename(self):
        return str(self.file)

    @filename.setter
    def filename(self, filename: str):
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
    def odoo_sequence(self, value):

        raise NotImplementedError("This field can not be set. Please modify the config.yaml instead.")

    def _read(self) -> str:
        """
        This method must be implemented by the child class as each document type has a different format.
        :return: None
        :rtype: None
        """

        invoice: str = ''
        image, line_items_coordinates = self._mark_region()

        # the invoice number usually lives in regions -1 to -3
        for regions in self._regions_list:
            for i in regions:
                try:
                    t: str = self._read_text(image, line_items_coordinates, -i).replace('\n', ' ')
                    logger.debug(f'Region: {i} Invoice: {invoice}')
                    m = self.regex.search(t)
                    invoice = m.group(1)

                except IndexError:
                    break
                except Exception:
                    pass

        return invoice

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
                logger.info(
                    f"{self.filename} can not be parsed. Changing OCR sensitivity {self.threshold_region_ignore + self.config['documents'][self.document_type]['threshold_region_ignore_decrement']} -> {self.threshold_region_ignore}."
                )

        return self._name

    @name.setter
    def name(self, value):
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
    def threshold_region_ignore(self):
        return self._threshold_region_ignore

    @threshold_region_ignore.setter
    def threshold_region_ignore(self, threshold_region_ignore):
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
        self.db: str = self.config['db']
        self.username: str = self.config['username']
        self.password: str = self.config['password']

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
        self._uid = user_id

    def odoo_document_id(self, document: DocumentImage) -> int:
        retry: int = 0
        odoo_id: int = 0

        while odoo_id == 0 and retry < self.config['retry']:
            try:
                with xmlrpc.client.ServerProxy(f'{self.url}/xmlrpc/2/object', allow_none=True,
                                               context=ssl._create_unverified_context()) as models:
                    res = models.execute_kw(self.db, self.uid, self.password,
                                            self.config['documents'][document.document_type]['odoo_object'],
                                            'search_read', [[['name', '=', document.name]]],
                                            {'fields': ['id', 'name']}
                                            )
                    # If we get an id, set it in the document
                    if res and res[0]['name'] == document.name:
                        odoo_id = res[0]['id']
                        logger.debug(f'File: {document.filename} Name: {document.name} has an Odoo ID of {odoo_id}')
                        document.odoo_id = odoo_id
                    else:
                        break
            except Exception as e:
                logger.error(e)
                odoo_id = 0
                retry += 1
                sleep(self.config['retry_sleep'])

        return odoo_id

    def save_document(self, document: DocumentImage) -> int:
        retry: int = 0
        while retry < self.config['retry']:
            try:
                # Make sure we have a document id from Odoo, if not, get one
                if document.odoo_id or self.odoo_document_id(document):
                    with xmlrpc.client.ServerProxy(f'{self.url}/xmlrpc/2/object', allow_none=True,
                                                   context=ssl._create_unverified_context()) as models:
                        with document.file.open('rb') as f:
                            data = base64.b64encode(f.read())
                            values = {
                                'name': document.name.replace('/', '-') + '_' + document.filename.replace('/', '-'),
                                'res_id': document.odoo_id,
                                'res_model': self.config['documents'][document.document_type]['odoo_object'],
                                'datas': data.decode('ascii')
                            }
                            res = models.execute_kw(self.db, self.uid, self.password, 'ir.attachment', 'create', [
                                values, ])

                            return res
                else:
                    logger.error(
                        f'Save failed: document {document.name} from file: {document.filename} can not be found in Odoo.')
                    break

            except Exception as e:
                logger.error(e)
                retry += 1
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

    def document_generator(self, files: [str]) -> DocumentImage:
        """
        Returns a generator that yields a DocumentImage object for (hopefully) each file
        or file glob passed

        :param files: a list of files or file globs to process
        :type files: list[str]
        :return: generator for DocumentImage(s)
        :rtype: DocumentImage
        """
        pass


# def _environ_or_required(key):
#     """Helper to ensure args are set or an ENV variable is present"""
#     if os.environ.get(key):
#         return {'default': os.environ.get(key)}
#     else:
#         return {'required': True}

def _parse_args():
    # Get configuration from environmental variables or command line
    parser = argparse.ArgumentParser(description="Script to read scanned documents and send them to Odoo")

    try:
        parser.add_argument('-s', '--server', dest='server', default=os.environ.get("DS_SERVER",
                                                                                    'production'),
                            help="The server configuration to use from the config file")
        parser.add_argument('-c', '--config', dest='config_file', default=os.environ.get("DS_CONFIG",
                                                                                         "/etc/docscanner.conf"),
                            help="The path to the YAML configuration file. Defaults to /etc/docscanner.conf")
        parser.add_argument('-v', '--verbose', action='store_true', help="enable verbose output")
        parser.add_argument('file', type=str, nargs='+',
                            help="The file, files or directories to process (required)")

        return parser.parse_args()

    except Exception as e:
        logger.error(e)
        sys.exit(1)


def get_configuration(config_file: object, server: str = "development") -> dict:
    """
    The configuration read from a YAML file
    :param config_file: the configuration file, either as pathlib.Path or str
    :type config_file: object
    :param server: the server configuration to use from the [servers] section
    :type server: str
    :return: dictionary of configuration settings
    :rtype: dict
    """

    config_file: Path = config_file if type(config_file) == Path else Path(config_file)
    server: str = server

    with open(config_file) as f:
        # global config
        config: dict = yaml.safe_load(f)

        # get login info from the config config_file
        config['url'] = config['servers'][server]['url']
        config['db'] = config['servers'][server]['database']
        config['username'] = config['servers'][server]['username']
        config['password'] = config['servers'][server]['password']

        config['debug'] = False

        return config


def main():
    args = _parse_args()
    config = get_configuration(args.config_file, args.server)
    config.debug = args.debug

    # get path to tesseract from config
    pytesseract.pytesseract.tesseract_cmd = config['tesseract-bin']
    path = Path(sys.argv[1])
    done_path = Path(str(path) + '/done')
    done_path.mkdir(exist_ok=True)

    # Improve OCR by increasing threads to max cpus minus one
    try:
        os.environ['OMP_THREAD_LIMIT'] = str((len(psutil.Process().cpu_affinity()) - 1) or 1)

    except AttributeError:
        os.environ['OMP_THREAD_LIMIT'] = str((psutil.cpu_count() - 1) or 1)

    # for filename in path.glob('*.png'):
    for filename in path.glob('*.jpg'):
        res = 0
        invoice: DocumentImage = DocumentImage(filename)
        if invoice.odoo_id and invoice.name:
            try:
                res = invoice.save()
            except Exception as e:
                logger.error(e)

            if res:
                logger.info(f"Moving {invoice.filename} to {done_path}")
                filename.rename(
                    done_path.joinpath(
                        f"{invoice.name.replace('/', '-')}_{invoice.odoo_id}_{invoice.filename.replace('/', '-')}"))
        else:
            if invoice.name:
                # filename.replace(invoice.name.replace('/', '-') + '_' + invoice.filename.replace('/', '-'))
                logger.warning(f"{invoice.name or invoice.filename} can not be processed at this time.")


if __name__ == "__main__":
    main()
