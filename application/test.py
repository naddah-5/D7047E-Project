import torch
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


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

        y_true += labels.cpu().numpy().tolist()
        y_pred += predicted.cpu().numpy().tolist()

    accuracy = correct / total
    f1 = f1_score(y_true, y_pred)
    print(confusion_matrix(y_true, y_pred))
    return accuracy, f1
