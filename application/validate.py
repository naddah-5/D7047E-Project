import torch
from sklearn.metrics import f1_score


def validate_model(val_loader, loss_function, network, device: str):
        correct = 0
        total = 0
        accuracy = 0

        y_true = []
        y_pred = []

        for _, (data, labels) in enumerate(val_loader):
            data, labels=data.to(device), labels.to(device)
            predictions = network.forward(data)

            _, predicted = torch.max(predictions.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = loss_function(predictions, labels)

            y_true += labels
            y_pred += predicted

        accuracy = correct / total
        f1 = f1_score(y_true, y_pred)

        return loss, accuracy ,f1
