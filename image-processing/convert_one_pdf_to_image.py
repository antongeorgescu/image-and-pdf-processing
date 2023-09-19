# import module
from pdf2image import convert_from_path
import os

PDF_FILE_PATH = f'{os.getcwd()}\\data-samples\\pdf'
POPPLER_PATH = 'C:\\Users\\ag4488\OneDrive - Finastra\\Visual Studio 2022\\Projects\\Poppler\\poppler-23.08.0\\Library\\bin'

PDF_FILE = 'ARB Voting - CFSA Financial Literacy Training Module.pdf'

IMAGE_FILES_PATH = f'{os.getcwd()}\\image-files'

# read the PDF file to be converted
images = convert_from_path(
    poppler_path=POPPLER_PATH,
    pdf_path=os.path.join(PDF_FILE_PATH,PDF_FILE))

# Save pages as images in the pdf
img_file = PDF_FILE.replace('.pdf','.png')
images[0].save(f'{IMAGE_FILES_PATH}\\PNG\\{img_file}', 'PNG')

img_file = PDF_FILE.replace('.pdf','.jpg')
images[0].save(f'{IMAGE_FILES_PATH}\\JPEG\\{img_file}', 'JPEG')

