import os
import pydicom as dicom
import cv2   

dicom_dir = r'C:\Users\kevin\OneDrive\Documents\Hem dator\D7047E-Project\XRAY DICOM'
jpeg_dir = r'C:\Users\kevin\OneDrive\Documents\Hem dator\D7047E-Project\XRAY JPEG'

# kolla om det finns en map som jpeg till jpeg filerna
if not os.path.exists(jpeg_dir):
    os.makedirs(jpeg_dir)

# loopa igenom alla dicom filer
for file in os.listdir(dicom_dir):
    if file.endswith('.dcm'):
        
        ds = dicom.dcmread(os.path.join(dicom_dir, file))       #läs dicom filen
        pixel_array_numpy = ds.pixel_array                      #konventera till pixel array

        image_path = os.path.join(jpeg_dir, file.replace('.dcm', '.jpg')) #spara som en jpeg fil i den nya mappen
        cv2.imwrite(image_path, pixel_array_numpy)
