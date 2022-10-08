import sys

from pdf2image import convert_from_path

pdfs = sys.argv[1]
pages = convert_from_path(pdfs, 350)

i = 1
for page in pages:
    image_name = pdfs + "-Page_" + str(i) + ".png"
    page.save(image_name, "PNG")
    print(f"{pdfs} -> {image_name} ")
    i = i+1