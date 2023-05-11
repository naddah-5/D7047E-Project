#Used to run already finished models. 

#from dataset.dataset import Dataset
from model.model import CNN
from application.train import Training
from application.test import test_model
from model.Resnet_feature_extract import ResNet_extract
from model.Resnet_fine_tuning import ResNet_tune
from model.Inception_V3_fine_tuning import Inception_tune
from model.Inception_V3_feature_extract import Inception_extract
from application.validate import validate_model
import torch
import torch.nn as nn
import torchvision

from dataset.dataset import load_dataset

def final(epochs: int = 4, batch_size: int = 3, learning_rate: float = 1e-4):
    best_net: str = ''

    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # set device to gpu if available
    print("using device ", my_device)
    
    train_loader, validation_loader, test_loader= load_dataset(batch_size=batch_size)
    
    state_dict = torch.load('best_network.pt')
    model = ResNet_tune(class_count=2, device=my_device)
    model.load_state_dict(state_dict)
    model.eval()

    loss_val, accuracy_val ,f1_val = validate_model(val_loader=validation_loader, loss_function=nn.CrossEntropyLoss(), network=model, device=my_device)

    print("\nValidation accuracy: %f" % accuracy_val, "\nValidation f1: %f" % f1_val)

    test_accuracy, f1 = test_model(test_loader=test_loader, network=model, device=my_device)

    print("\nTest accuracy: %f" % test_accuracy, "\nTest f1: %f" % f1)

if __name__=="__main__":
    final()