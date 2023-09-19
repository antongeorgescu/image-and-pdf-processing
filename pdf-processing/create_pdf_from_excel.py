# Import the required modules
import win32com.client # For creating a COM object
import os # For getting the file names in a directory

# Create a COM object for Excel application
excel = win32com.client.Dispatch("Excel.Application")

# Get the path of the directory where the Excel files are located
excel_dir_path = f'{os.getcwd()}\\source-data\\excel' 
pdf_dir_path = f'{os.getcwd()}\\source-data\\pdf' 

# Loop through the files in the directory
for file in os.listdir(excel_dir_path):
    # Check if the file is an Excel file
    if file.endswith(".xlsx"):
        # Get the full path of the file
        file_path = os.path.join(excel_dir_path, file)
        # Get the file name without extension
        file_name = os.path.splitext(file)[0]
        # Open the Excel file
        workbook = excel.Workbooks.Open(file_path)
        # Save the Excel file as a PDF file with the same name
        workbook.ExportAsFixedFormat(0, pdf_dir_path + "\\" + file_name + ".pdf")
        # Close the Excel file
        workbook.Close()

# Quit the Excel application
excel.Quit()
