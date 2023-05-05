#from dataset.dataset import Dataset
from model.model import CNN
from application.train import Training
from application.test import test_model
import torch
import os
import shutil

from dataset.dataset import load_dataset


def main(epochs: int = 1, batch_size: int = 10, learning_rate: float = 1e-4):
    best_net: str = ''

    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # set device to gpu if available
    print("using device ", my_device)
    
    train_loader, validation_loader, test_loader= load_dataset(batch_size=batch_size)
    

    model = CNN(class_count=2, device=my_device)


    training = Training(model, train_loader, validation_loader, test_loader, epochs, learning_rate, device=my_device)
    training.train_model()
    

    test_accuracy, f1 = test_model(test_loader=test_loader, network=model, device=my_device)

    print("\nTest accuracy: %f" % test_accuracy, "\nTest f1: %f" % f1)


def move(pneumonia_path, viral_path, bacterial_path):
    # Create the new viral and bacterial folders if they don't already exist
    os.makedirs(viral_path, exist_ok=True)
    os.makedirs(bacterial_path, exist_ok=True)

    # Loop through the pneumonia folder and move the images to the appropriate folder
    for filename in os.listdir(pneumonia_path):
        if "virus" in filename:
            shutil.move(os.path.join(pneumonia_path, filename), os.path.join(viral_path, filename))
        elif "bacteria" in filename:
            shutil.move(os.path.join(pneumonia_path, filename), os.path.join(bacterial_path, filename))

if __name__=="__main__":
    main()

    # Create a copy of chest_xray and call it chest_xray_pneumonia

    # pneumonia_folder_path = "dataset/chest_xray_pneumonia/train/PNEUMONIA"
    # viral_folder_path = "dataset/chest_xray_pneumonia/train/viral"
    # bacterial_folder_path = "dataset/chest_xray_pneumonia/train/bacterial"
    # move(pneumonia_folder_path, viral_folder_path, bacterial_folder_path)
    #
    # pneumonia_folder_path = "dataset/chest_xray_pneumonia/val/PNEUMONIA"
    # viral_folder_path = "dataset/chest_xray_pneumonia/val/viral"
    # bacterial_folder_path = "dataset/chest_xray_pneumonia/val/bacterial"
    # move(pneumonia_folder_path, viral_folder_path, bacterial_folder_path)
    #
    # pneumonia_folder_path = "dataset/chest_xray_pneumonia/test/PNEUMONIA"
    # viral_folder_path = "dataset/chest_xray_pneumonia/test/viral"
    # bacterial_folder_path = "dataset/chest_xray_pneumonia/test/bacterial"
    # move(pneumonia_folder_path, viral_folder_path, bacterial_folder_path)
