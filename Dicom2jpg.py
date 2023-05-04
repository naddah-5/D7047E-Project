import os
import pydicom as dicom
import cv2   

# specify your input and output directories
input_dir = r'C:\Users\kevin\OneDrive\Documents\Hem dator\D7047E-Project\XRAY DICOM'
output_dir = r'C:\Users\kevin\OneDrive\Documents\Hem dator\D7047E-Project\XRAY JPEG'

# create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# loop over all DICOM files in the input directory
for file in os.listdir(input_dir):
    if file.endswith('.dcm'):
        # read the DICOM file
        ds = dicom.dcmread(os.path.join(input_dir, file))

        # convert the pixel array to a numpy array
        pixel_array_numpy = ds.pixel_array

        # save the numpy array as a JPEG image in the output directory
        image_path = os.path.join(output_dir, file.replace('.dcm', '.jpg'))
        cv2.imwrite(image_path, pixel_array_numpy)
