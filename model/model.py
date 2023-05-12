import torch
import torch.nn as nn
import copy

class CNN(nn.Module):
    def __init__(self, class_count: int = 2, dropout: float = 0.8, drop_in: float = 0.9, device: str = 'cpu'):
        super().__init__()


        
        self.A1 = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.Conv2d(1, 200, 7),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(3, 3),
            nn.Conv2d(200, 200, 5),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(3, 3),
            nn.Conv2d(200, 200, 3),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(2, 2),
        )

        self.B1 = nn.Sequential(
            nn.Conv2d(200, 500, 3),                 # B1
            nn.Dropout2d(p=0.8),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(500, 500, 3),
            nn.Dropout2d(p=0.8),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_in),
            
            nn.Linear(500, 8_192),                 # fc1
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(8_192, 8_192),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(8_192, 4_096),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(4_096, 4_096),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(4096, 1_000),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(1_000, class_count),             # fc5
        )

        self.to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.A1(x)

        x = self.B1(x)

        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x