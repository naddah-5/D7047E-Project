import torch
from sklearn.metrics import f1_score


def test_model(test_loader, network, device: str):
    correct = 0
    total = 0
    accuracy = 0

    y_true = []
    y_pred = []

    for _, (data, labels) in enumerate(test_loader, 0):
        data, labels = data.to(device), labels.to(device)
        predictions = network.forward(data)

        _, predicted = torch.max(predictions.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        y_true += labels
        y_pred += predicted

    accuracy = correct / total
    f1 = f1_score(y_true, y_pred)
    return accuracy, f1
