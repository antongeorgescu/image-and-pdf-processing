# import module
from pdf2image import convert_from_path
import os

NET_PDF_FILES_PATH = 'C:\\Users\\ag4488\OneDrive - Finastra\\Visual Studio 2022\\Projects\\create-pdf-in-csharp\\pdf_from_excel\\pdf_from_excel\\bin\\Debug\\net6.0\\files'
POPPLER_PATH = 'C:\\Users\\ag4488\OneDrive - Finastra\\Visual Studio 2022\\Projects\\Poppler\\poppler-23.08.0\\Library\\bin'
IMAGE_FILES_PATH = f'{os.getcwd()}\\image-files'

# read the PDF filers to be converted
for filename in os.listdir(NET_PDF_FILES_PATH):
    if os.path.isfile(os.path.join(NET_PDF_FILES_PATH,filename)):
        images = convert_from_path(
	        poppler_path=POPPLER_PATH,
			pdf_path=os.path.join(NET_PDF_FILES_PATH,filename))
        # Save pages as images in the pdf
        jpg_file = filename.replace('.pdf','.jpg')
        images[0].save(f'{IMAGE_FILES_PATH}\\JPEG\\{jpg_file}', 'JPEG')



