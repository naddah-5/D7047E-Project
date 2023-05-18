import torch
import torch.nn as nn
import copy

class CNN(nn.Module):
    def __init__(self, class_count: int = 2, dropout: float = 0.6, drop_in: float = 0.8, device: str = 'cpu'):
        super().__init__()


        
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

            nn.Conv2d(1, 300, 5),
            nn.BatchNorm2d(300),
            nn.LeakyReLU(),
            nn.MaxPool2d(4, 4),

            nn.Conv2d(300, 1, 1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),

            nn.Conv2d(1, 300, 3),
            nn.BatchNorm2d(300),
            nn.LeakyReLU(),

            nn.Conv2d(300, 100, 1),
            nn.BatchNorm2d(100),
            nn.LeakyReLU(),
        )

        self.classifier = nn.Sequential(
            
            nn.Linear(10_000, 8_192),
            nn.LeakyReLU(),
            nn.Dropout(p=drop_in),

            nn.Linear(8_192, 8_192),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(8_192, 4_096),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(4_096, 2_048),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(2_048, 2_048),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(2_048, 512),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(512, class_count),
            nn.Softmax(dim=1)
        )

        self.to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.A1(x)

        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    