import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score

from .validate import validate_model

class Training():

    def __init__(self, network, train_loader, val_loader, test_loader, epochs: int, learning_rate, best_net=None,device: str='cpu', debug_prediction: bool = False):
        self.network = network
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(network.parameters(), lr=learning_rate)
        self.writer = SummaryWriter()
        self.best_net = best_net
        self.device=device
        self.debug_prediction = debug_prediction
        self.predictions = []

    def train_model(self):
        best_accuracy = 0
        iteration = 0
        
        for epoch in range(self.epochs):
            correct = 0
            total = 0
            accuracy = 0

            y_true = []
            y_pred = []

            for batch_nr, (data, labels) in enumerate(self.train_loader):
                self.network.train()
                iteration += 1
                data, labels=data.to(self.device), labels.to(self.device)
                predictions = self.network.forward(data)

                _, predicted = torch.max(predictions.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                y_true += labels.cpu().numpy().tolist()
                y_pred += predicted.cpu().numpy().tolist()

                self.predictions.append(predicted)

                loss = self.loss_function(predictions, labels)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                print(
                    f'\rEpoch {epoch + 1}/{self.epochs} [{batch_nr + 1}/{len(self.train_loader)}] - Loss {loss:.10f}' ,
                    end=''
                )

            accuracy = correct / total
            f1 = f1_score(y_true, y_pred)
            self.writer.add_scalar('Loss/train', loss, (epoch + 1))
            self.writer.add_scalar('Accuracy/train', accuracy, (epoch + 1))
            self.writer.add_scalar('f1/train', f1, (epoch + 1))

            loss, accuracy, f1 = validate_model(val_loader=self.val_loader, loss_function=self.loss_function, network=self.network, device=self.device)

            if self.debug_prediction:
                for i in range(10):
                    print(self.predictions[-i])

            loss, accuracy, _ = validate_model(val_loader=self.val_loader, loss_function=self.loss_function, network=self.network, device=self.device)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(self.network.state_dict(), "best_network.pt")
                print("\nFound better network, accuracy -", accuracy)
            self.writer.add_scalar('Loss/validation', loss, (epoch + 1))
            self.writer.add_scalar('Accuracy/validation', accuracy, (epoch + 1))
            self.writer.add_scalar('f1/validation', f1, (epoch + 1))

        print("\nBest model had a validation accuracy of -", best_accuracy)

        return