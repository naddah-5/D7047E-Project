import os
import random
import shutil

# Set the input path for your image dataset
input_path = 'C:/Users/kevin/OneDrive/Documents/Hem dator/D7047E-Project/RSNA_DATASET/DATASET_RSNA/RSNA_DATASET'

# Set the percentage of images for each set
train_pct = 0.6
val_pct = 0.2
test_pct = 0.1
fake_test_pct = 0.1

# Iterate over the class folders
for class_folder in ['Normal (6000)', 'Pneumonia (6000)']:
    # Get a list of image file names in the folder
    image_files = os.listdir(os.path.join(input_path, class_folder))
    # Shuffle the list of image file names
    random.shuffle(image_files)
    # Calculate the number of images for each set
    num_images = len(image_files)
    num_train = int(train_pct * num_images)
    num_val = int(val_pct * num_images)
    num_test = int(test_pct * num_images)
    num_fake_test = int(fake_test_pct * num_images)
    # Create subfolders for each set and class
    os.makedirs(os.path.join('training_set', class_folder))
    os.makedirs(os.path.join('validation_set', class_folder))
    os.makedirs(os.path.join('test_set', class_folder))
    os.makedirs(os.path.join('fake_test_set', class_folder))
    # Copy images to each set subfolder
    for i, image_file in enumerate(image_files):
        if i < num_train:
            set_folder = 'training_set'
        elif i < num_train + num_val:
            set_folder = 'validation_set'
        elif i < num_train + num_val + num_test:
            set_folder = 'test_set'
        else:
            set_folder = 'fake_test_set'
        shutil.copy(os.path.join(input_path, class_folder, image_file), os.path.join(set_folder, class_folder, image_file))
