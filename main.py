#from dataset.dataset import Dataset
from model.model import CNN
from application.train import Training
from application.test import test_model
import torch

from dataset.dataset import load_dataset


def main(epochs: int = 30, batch_size: int = 8, learning_rate: float = 1e-4):
    best_net: str = ''

    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # set device to gpu if available
    print("using device ", my_device)
    
    train_loader, validation_loader, test_loader= load_dataset(batch_size=batch_size)
    

    model = CNN(class_count=2, device=my_device)


    training = Training(model, train_loader, validation_loader, test_loader, epochs, learning_rate, device=my_device)
    training.train_model()
    

    test_accuracy, f1 = test_model(test_loader=test_loader, network=model, device=my_device)

    print("\nTest accuracy: %f" % test_accuracy, "\nTest f1: %f" % f1)

if __name__=="__main__":
    main()
