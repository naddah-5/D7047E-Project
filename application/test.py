import torch

def test_model(test_loader, network, device: str):
    correct = 0
    total = 0
    accuracy = 0
    for _, (data, labels) in enumerate(test_loader, 0):
        data, labels=data.to(device), labels.to(device)
        predictions = network.forward(data)

        _, predicted = torch.max(predictions.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy