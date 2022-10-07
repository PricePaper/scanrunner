#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import base64
import os
import re
import ssl
import sys
import xmlrpc.client
from pathlib import Path

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

with open('/config.yaml') as f:
    config = yaml.safe_load(f)

# Set default server config to test server
server = "odoo-dev"

# get login info from the config file
url = config[server]['url']
db = config[server]['database']
username = config[server]['username']
password = config[server]['password']

# get path to tesseract from config
pytesseract.pytesseract.tesseract_cmd = config['tesseract-bin'] 

# get uid of Odoo user
with xmlrpc.client.ServerProxy(f"{url}/xmlrpc/2/common", allow_none=True,
                               context=ssl._create_unverified_context()) as common:
    uid = common.authenticate(db, username, password, {})
    # Cleanup
    del common


def mark_region(image_path):
    image = cv2.imread(image_path)

    # define threshold of regions to ignore
    THRESHOLD_REGION_IGNORE = 80

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except AssertionError:
        gray = image

    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 30)

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    line_items_coordinates = []
    for c in cnts:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)

        if w < THRESHOLD_REGION_IGNORE or h < THRESHOLD_REGION_IGNORE:
            continue

        image = cv2.rectangle(image, (x, y), (x + w, y + h), color=(255, 0, 255), thickness=3)
        line_items_coordinates.append([(x, y), (x + w, y + h)])

    return image, line_items_coordinates


def read_text(image, line_items_coordinates, index):
    # get co-ordinates to crop the image
    c = line_items_coordinates[index]

    # cropping image img = image[y0:y1, x0:x1]
    img = image[c[0][1]:c[1][1], c[0][0]:c[1][0]]

    # convert the image to black and white for better OCR
    ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)

    # pytesseract image to string to get results
    text = str(pytesseract.image_to_string(thresh1, config='--psm 6'))
    return text


def read_invoice(filename: str, invoice_rexp: re.Pattern) -> str:
    invoice: str = ''
    image, line_items_coordinates = mark_region(filename)

    # the invoice number usually lives in regions -1 to -3
    region: int = 0
    for i in (1, 2, 3, 4):
        try:
            t: str = read_text(image, line_items_coordinates, -i).replace('\n', ' ')

            # invoice, inv_date, partner = invoice_rexp.search(t).groups()
            # print(f'Region: {i} Invoice: {invoice} date: {inv_date} partner-code: {partner}')
            m = invoice_rexp.search(t)
            invoice = m.group(1)
            region = i

        except IndexError:
            region = 0
            break
        except Exception as e:
            region = 0
            pass

    # Check to see if we got anything useful, or try again in regions -5 through -7
    if region == 0:
        for i in (5, 6, 7, 8):
            try:
                t = read_text(image, line_items_coordinates, -i).replace('\n', ' ')
                # invoice, inv_date, partner = invoice_rexp.search(t).groups()
                # print(f'Region: {i} Invoice: {invoice} date: {inv_date} partner-code: {partner}')
                m = invoice_rexp.search(t)
                invoice = m.group(1)
                region = i
            except IndexError:
                region = 0
                break
            except Exception as e:
                region = 0
                pass

    return f"INV{invoice}"


def attach_invoice(invoice: str, filename: str):
    with xmlrpc.client.ServerProxy(f'{url}/xmlrpc/2/object', allow_none=True,
                                   context=ssl._create_unverified_context()) as models:
        res = models.execute_kw(db, uid, password, 'account.move', 'search_read',
                                [[['name', '=', invoice]]],
                                {'fields': ['id', 'name']}
                                )

        if res:
            odoo_invoice = res[0]
            with open(filename, 'rb') as f:
                data = base64.b64encode(f.read())
                values = {
                    'name': invoice.replace('/', '-') + '_' + filename,
                    'res_id': odoo_invoice.get('id'),
                    'res_model': 'account.move',
                    'datas': data.decode('ascii')
                }
                res = models.execute_kw(db, uid, password, 'ir.attachment', 'create', [values, ])

                return res
        else:
            return None


def main():
    path = Path(sys.argv[1])
    done_path = Path(str(path) + '/done')
    done_path.mkdir(exist_ok=True)

    # Improve OCR by increasing threads to max cpus minus one
    try:
        os.environ['OMP_THREAD_LIMIT'] = str((len(psutil.Process().cpu_affinity()) - 1) or 1)

    except AttributeError:
        os.environ['OMP_THREAD_LIMIT'] = str((psutil.cpu_count() - 1) or 1)

    # regular expression to extract the invoice number from the text
    # invoice_rexp = re.compile(r'(R?INV\/20[0-9]{2}\/[0-9]{4,7}).+Date:.*([01][0-0]\/[0-3][0-9]\/20[0-9]{2}).*Partner.*Code:.+([A-Z0-9]{6})')
    invoice_rexp = re.compile(r'(/20[0-9]{2}/[0-9]{4,7})')

    #for filename in path.glob('*.png'):
    for filename in path.glob('*.jpg'):
        invoice: str = read_invoice(str(filename), invoice_rexp)
        if invoice:
            res = attach_invoice(invoice, str(filename))
            if res:
                filename.rename(done_path.joinpath(invoice.replace('/', '-') + '_' + filename.name))


if __name__ == "__main__":
    main()
