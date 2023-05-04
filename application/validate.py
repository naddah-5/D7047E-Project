import torch


def validate_model(val_loader, loss_function, network, device: str):
        correct = 0
        total = 0
        accuracy = 0
        for _, (data, labels) in enumerate(val_loader):
            data, labels=data.to(device), labels.to(device)
            predictions, _, _ = network.forward(data)

            _, predicted = torch.max(predictions.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = loss_function(predictions, labels)

        accuracy = correct / total


        return loss, accuracy
