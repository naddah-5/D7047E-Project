import torch
import torch.nn as nn
import copy

class CNN(nn.Module):
    def __init__(self, class_count: int = 2, dropout: float = 0.6, drop_in: float = 0.8, device: str = 'cpu'):
        super().__init__()

        self.dropin = drop_in
        self.dropout = dropout
        
        self.A1 = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),

            nn.Conv2d(1, 300, 7),
            nn.BatchNorm2d(300),
            nn.LeakyReLU(),
            nn.MaxPool2d(4, 4),

            nn.Conv2d(300, 1, 1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),

            nn.Conv2d(1, 500, 5),
            nn.BatchNorm2d(500),
            nn.LeakyReLU(),
            nn.MaxPool2d(4, 4),

            nn.Conv2d(500, 1, 1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),

            nn.Conv2d(1, 500, 3),
            nn.BatchNorm2d(500),
            nn.LeakyReLU(),

            nn.Conv2d(500, 100, 1),
            nn.BatchNorm2d(100),
            nn.LeakyReLU(),
        )

        self.classifierA = nn.Sequential(
            nn.Linear(10_000, 8_192),
            nn.LeakyReLU(),
        )

        self.classifierB = nn.Sequential(
            nn.Linear(8_192, 8_192),
            nn.LeakyReLU(),
        )

        self.classifierC = nn.Sequential(
            nn.Linear(8_192, 2_048),
            nn.Tanh(),
        )

        self.classifierD = nn.Sequential(
            nn.Linear(2_048, 1_024),
            nn.LeakyReLU(),
        )

        self.classifierD = nn.Sequential(
            nn.Linear(1_024, class_count),
            nn.Softmax(dim=1)
        )

        self.to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.A1(x)

        x = torch.flatten(x, 1)

        x = self.classifierA(x)
        nn.Dropout(p=self.dropin),

        x = self.classifierB(x)
        nn.Dropout(p=self.dropout),

        x = self.classifierC(x)
        nn.Dropout(p=self.dropout),

        x = self.classifierD(x)
        return x
    