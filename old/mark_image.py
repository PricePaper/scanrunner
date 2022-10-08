#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# use this command to install open cv2
# pip install opencv-python

import sys
import re
import pytesseract


invoice_rexp = re.compile(r'(R?INV\/20[0-9]{2}\/[0-9]{4,7}).+Date:.+([01][0-0]\/[0-3][0-9]\/20[0-9]{2}).+Partner Code:.+([A-Z0-9]{6})')

pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'


#def mark_region(image_path):
#    image = cv2.imread(image_path)
#
#    # define threshold of regions to ignore
#    THRESHOLD_REGION_IGNORE = 80
#
#    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#    blur = cv2.GaussianBlur(gray, (9, 9), 0)
#    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 30)
#
#    # Dilate to combine adjacent text contours
#    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
#    dilate = cv2.dilate(thresh, kernel, iterations=4)
#
#    # Find contours, highlight text areas, and extract ROIs
#    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#
#    line_items_coordinates = []
#    for c in cnts:
#        area = cv2.contourArea(c)
#        x, y, w, h = cv2.boundingRect(c)
#
#        if w < THRESHOLD_REGION_IGNORE or h < THRESHOLD_REGION_IGNORE:
#            continue
#
#        image = cv2.rectangle(image, (x, y), (x + w, y + h), color=(255, 0, 255), thickness=3)
#        line_items_coordinates.append([(x, y), (x + w, y + h)])
#
#    return image, line_items_coordinates

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

def main():
    FILENAME = sys.argv[1]
    image, line_items_coordinates = mark_region(FILENAME)
    a_i = Image.fromarray(image)
    a_i.save(FILENAME, "JPEG")

    # index 25 is invoice number
    t = read_text(image,line_items_coordinates, 5).replace('\n', ' ')
    #print(t)
    invoice, inv_date, partner = invoice_rexp.search(t).groups()
    print(f'Invoice: {invoice} date: {inv_date} partner-code: {partner}')

if __name__ == "__main__":
    main()
