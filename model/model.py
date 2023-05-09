import torch
import torch.nn as nn
import copy

class CNN(nn.Module):
    def __init__(self, class_count: int = 2, dropout: float = 0.8, drop_in: float = 0.9, device: str = 'cpu'):
        super().__init__()


        
        self.A1 = nn.Sequential(
            nn.Conv2d(1, 100, 3),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(100, 200, 5),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(200, 300, 7),
            nn.MaxPool2d(2, 2),
        )

        self.B1 = nn.Sequential(
            nn.Conv2d(300, 600, 3),                 # B1
            nn.Dropout2d(p=0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(600, 600, 3),
            nn.Dropout2d(p=0.2),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_in),
            
            nn.Linear(9_600, 1_000),                 # fc1
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(1_000, 1_000),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(1_000, 1_000),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(1_000, 1_000),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(1_000, 100),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(100, class_count),             # fc5
        )

        self.to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.A1(x)

        x = self.B1(x)

        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

