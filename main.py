#from dataset.dataset import Dataset
from model.model import CNN
from application.train import Training
from application.test import test_model
from application.visualize_deconv import visualize
import torch

from dataset.dataset import load_dataset

from model.deconv import DeconvCNN
import cv2
import numpy as np


def main(epochs: int = 100, batch_size: int = 12, learning_rate: float = 1e-4):
    best_net: str = ''

    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # set device to gpu if available
    #my_device = torch.device("cpu")
    print("using device ", my_device)
    
    train_loader, validation_loader, test_loader= load_dataset(batch_size=batch_size)
    

    model = CNN(class_count=2, device=my_device)
    model.to(my_device)

    deconv_model = DeconvCNN(class_count=2, device=my_device)
    deconv_model.to(my_device)


    training = Training(model, train_loader, validation_loader, test_loader, epochs, learning_rate, device=my_device)
    training.train_model()

    test_accuracy, f1 = test_model(test_loader=test_loader, network=model, device=my_device)

    #input_image = torch.randn(1, 1, 512, 512)

    

    print("\nTest accuracy: %f" % test_accuracy, "\nTest f1: %f" % f1)


    # print(our_features.shape)
    # print('#############################\n',deconv_output.shape)
    # print('\n',deconv_output.type)

    # print(data_temp.shape)
    # visualize(data_temp, deconv_output)
    


if __name__=="__main__":
    main()


