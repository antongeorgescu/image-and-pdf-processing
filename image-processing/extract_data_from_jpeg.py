# Import the os module
import os
import camelot
from img2table.document import Image
from PIL import Image as PILImage
import cv2

IMG_EXT = 'png'
IMG_FILE = 'test' #'0BCQUNID7I'
IMG_FOLDER_PATH = f'{os.getcwd()}\\image-files\\{IMG_EXT.upper()}'


def create_file_inventory():
    # Define the folder path
    invfolderpath = f'{os.getcwd()}\\image-files\\inventory'

    # Define the output file name
    invfiles = ["train.txt","test.txt"]

    # Open the output file in write mode
    for invfile in invfiles:
        with open(f'{invfolderpath}\\{invfile}', "w+") as ifile:
            # Loop through the files in the folder
            for jfile in os.listdir(IMG_FOLDER_PATH):
                filename = jfile.replace(IMG_FOLDER_PATH,'').replace(f'.{IMG_EXT}','')
                # Write the file name to the output file, followed by a newline
                ifile.write(f'{filename}\n')

            # Close the output file
            ifile.close()

    # Print a message to indicate the script is done
    print("The script has finished writing the file names to", ", ".join(invfiles))

def extract_pdf_table(file):
    tables = camelot.read_pdf(file)
    print("Total tables extracted:", tables.n)

# Check if the file is executed as the main script
if __name__ == "__main__":
    create_file_inventory()

    image = Image(f'{IMG_FOLDER_PATH}\\{IMG_FILE}.{IMG_EXT}')
    # Table identification
    imgage_tables = image.extract_tables()

    # Display extracted tables
    table_img = cv2.imread(f'{IMG_FOLDER_PATH}\\{IMG_FILE}.{IMG_EXT}')

    for table in imgage_tables:
        for row in table.content.values():
            for cell in row:
                cv2.rectangle(table_img, (cell.bbox.x1, cell.bbox.y1), (cell.bbox.x2, cell.bbox.y2), (255, 0, 0), 2)
                
    PILImage.fromarray(table_img).save(f'{os.getcwd()}\\extracts\\{IMG_FILE}.{IMG_EXT}')