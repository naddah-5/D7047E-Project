#from dataset.dataset import Dataset
from model.model import CNN
from application.train import Training
from application.test import test_model
import torch

from dataset.dataset import load_dataset


def main(epochs: int = 4, batch_size: int = 8, learning_rate: float = 1e-4):
    best_net: str = ''

    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # set device to gpu if available
    print("using device ", my_device)
    
    train_loader, validation_loader, test_loader= load_dataset(batch_size=batch_size)
    

    model = CNN(class_count=2, device=my_device)


    training = Training(model, train_loader, validation_loader, test_loader, epochs, learning_rate, device=my_device)
    training.train_model()
    

    test_accuracy, f1 = test_model(test_loader=test_loader, network=model, device=my_device)

    print("\nTest accuracy: %f" % test_accuracy, "\nTest f1: %f" % f1)


import os
import shutil
def hack():
    # Set the path to the folder containing the pneumonia images
    pneumonia_folder_path = "dataset/chest_xray_pneumonia/train/PNEUMONIA"

    # Set the paths to the new viral and bacterial folders
    viral_folder_path = "dataset/chest_xray_pneumonia/train/viral"
    bacterial_folder_path = "dataset/chest_xray_pneumonia/train/bacterial"

    # Create the new viral and bacterial folders if they don't already exist
    os.makedirs(viral_folder_path, exist_ok=True)
    os.makedirs(bacterial_folder_path, exist_ok=True)

    # Loop through the pneumonia folder and move the images to the appropriate folder
    for filename in os.listdir(pneumonia_folder_path):
        if "virus" in filename:
            shutil.move(os.path.join(pneumonia_folder_path, filename), os.path.join(viral_folder_path, filename))
        elif "bacteria" in filename:
            shutil.move(os.path.join(pneumonia_folder_path, filename), os.path.join(bacterial_folder_path, filename))

    return


if __name__=="__main__":
    # main()
    hack()
