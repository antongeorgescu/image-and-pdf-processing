# Import the libraries
import os
import pytesseract
import pandas as pd

# Define the PNG file name
IMG_EXT = 'png'
IMG_FILE = 'ARB Voting - CFSA Financial Literacy Training Module'
IMG_FILE_PATH = f'{os.getcwd()}\\image-files\\{IMG_EXT.upper()}\\{IMG_FILE}.{IMG_EXT}'
EXCEL_FILE_PATH = f'{os.getcwd()}\\extracts\\{IMG_FILE}.xlsx'

pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\ag4488\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'

# Read the PNG file as an image
image = pytesseract.image_to_string(IMG_FILE_PATH)

# Convert the image to a dataframe
df = pd.read_csv(image, sep="\t")

# Write the dataframe to an Excel file
df.to_excel(EXCEL_FILE_PATH, index=False)

# Print a message to indicate the script is done
print("The script has extracted an Excel spreadsheet from", IMG_FILE_PATH)
