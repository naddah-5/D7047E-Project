import torch
import torch.nn as nn
import copy

class CNN(nn.Module):
    def __init__(self, class_count: int = 2, dropout: float = 0.8, drop_in: float = 0.95, device: str = 'cpu'):
        super().__init__()

        self.dropin = nn.Dropout(p=drop_in)
        self.dropout = nn.Dropout(p=dropout)
        
        self.FeatureA1 = nn.Sequential(
            nn.Conv2d(1, 300, 7),
            nn.BatchNorm2d(300),
            nn.LeakyReLU(),
            nn.MaxPool2d(4, 4),
            
            nn.Conv2d(300, 1, 1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
        )


        self.FeatureA2 = nn.Sequential(
            nn.Conv2d(1, 500, 5),
            nn.BatchNorm2d(500),
            nn.LeakyReLU(),
            nn.MaxPool2d(4, 4),

            nn.Conv2d(500, 1, 1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
            )


        self.FeatureA3 = nn.Sequential(
            nn.Conv2d(1, 500, 3),
            nn.BatchNorm2d(500),
            nn.LeakyReLU(),

            nn.Conv2d(500, 100, 1),
            nn.BatchNorm2d(100),
            nn.LeakyReLU(),
        )

        self.classifierA1 = nn.Sequential(
            nn.Linear(10_000, 8_192),
            nn.LeakyReLU(),
        )

        self.classifierA2 = nn.Sequential(
            nn.Linear(8_192, 8_192),
            nn.LeakyReLU(),
        )

        self.classifierA3 = nn.Sequential(
            nn.Linear(8_192, 2_048),
            nn.Tanh(),
        )

        self.classifierA4 = nn.Sequential(
            nn.Linear(2_048, 1_024),
            nn.LeakyReLU(),
        )

        self.classifierA5 = nn.Sequential(
            nn.Linear(1_024, class_count),
            nn.Softmax(dim=1)
        )

        self.to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.FeatureA1(x)
        x = self.FeatureA2(x)
        x = self.FeatureA3(x)

        x = torch.flatten(x, 1)

        x = self.dropin(self.classifierA1(x))
        
        x = self.dropout(self.classifierA2(x))

        x = self.dropout(self.classifierA3(x))

        x = self.dropout(self.classifierA4(x))

        x = self.classifierA5(x)
        
        return x
    