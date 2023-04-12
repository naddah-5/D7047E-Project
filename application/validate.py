import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import datetime


def validate_model(val_loader, loss_function, network, epoch: int, device: str):
        correct = 0
        total = 0
        accuracy = 0
        for _, (data, labels) in enumerate(val_loader):
            data, labels=data.to(device), labels.to(device)
            predictions = network.forward(data)

            _, predicted = torch.max(predictions.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = loss_function(predictions, labels)

        accuracy = correct / total


        return loss, accuracy
