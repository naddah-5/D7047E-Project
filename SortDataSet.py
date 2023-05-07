import os
import shutil
import pandas as pd
import pydicom as dicom
import cv2

data_dir = 'C:/Users/kevin/OneDrive/Documents/Hem dator/D7047E-Project/rsna-pneumonia-detection-challenge/stage_2_train_images'
train_csv = 'C:/Users/kevin/OneDrive/Documents/Hem dator/D7047E-Project/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv'
pneumonia_dir = 'C:/Users/kevin/OneDrive/Documents/Hem dator/D7047E-Project/RSNA_DATASET/PNEUMONIA'
normal_dir = 'C:/Users/kevin/OneDrive/Documents/Hem dator/D7047E-Project/RSNA_DATASET/NORMAL'


df_train = pd.read_csv(train_csv, skiprows=[0], header=None) # read the csv file
df_train.columns = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'label']


image_filenames = df_train['filename'].values # get the image file names form the csv file
labels = df_train['label'].values


sorted_indices = sorted(range(len(image_filenames)), key=lambda k: image_filenames[k]) # sort the files bases on the label of the image
image_filenames = image_filenames[sorted_indices]
labels = labels[sorted_indices]


# loop through the sorted image filenames and labels
for i, filename in enumerate(image_filenames):
    # load DICOM image
    image_path = os.path.join(data_dir, filename + '.dcm')
    ds = dicom.dcmread(image_path)
    pixel_array_numpy = ds.pixel_array
    
    # convert to JPEG and save to appropriate directory based on label
    if labels[i] == 1:
        save_path = os.path.join(pneumonia_dir, filename + '.jpg')
    else:
        save_path = os.path.join(normal_dir, filename + '.jpg')
    
    cv2.imwrite(save_path, pixel_array_numpy)
