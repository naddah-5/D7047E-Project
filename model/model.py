import torch
import torch.nn as nn
import copy

class CNN(nn.Module):
    def __init__(self, class_count: int = 2, dropout: float = 0.6, drop_in: float = 0.8, drop_tail = 0.4, device: str = 'cpu'):
        super().__init__()

        self.dropin = nn.Dropout(p=drop_in)
        self.dropout = nn.Dropout(p=dropout)
        self.droptail = nn.Dropout(p=drop_tail)
        
        self.FeatureA1 = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 200, 7),
            nn.BatchNorm2d(200),
            nn.LeakyReLU(),
            nn.MaxPool2d(4, 4),
            
            nn.Conv2d(200, 1, 1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
        )


        self.FeatureA2 = nn.Sequential(
            nn.Conv2d(1, 400, 5),
            nn.BatchNorm2d(400),
            nn.LeakyReLU(),
            nn.MaxPool2d(4, 4),

            nn.Conv2d(400, 1, 1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
            )


        self.FeatureA3 = nn.Sequential(
            nn.Conv2d(1, 400, 3),
            nn.BatchNorm2d(400),
            nn.LeakyReLU(),

            nn.Conv2d(400, 100, 1),
            nn.BatchNorm2d(100),
            nn.LeakyReLU(),
        )

        self.classifierA1 = nn.Sequential(
            nn.Linear(10_000, 8_192),
            nn.LeakyReLU(),
        )

        self.classifierB1 = nn.Sequential(
            nn.Linear(8_192, 8_192),
            nn.LeakyReLU(),
        )


        self.classifierC1 = nn.Sequential(
            nn.Linear(8_192, 2_048),
            nn.Tanh(),
        )

        self.classifierC2 = nn.Sequential(
            nn.Linear(2_048, 1_024),
            nn.LeakyReLU(),
        )

        self.classifierD1 = nn.Sequential(
            nn.Linear(1_024, class_count),
            nn.LeakyReLU()
        )

        self.to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.FeatureA1(x)
        x = self.FeatureA2(x)
        x = self.FeatureA3(x)

        x = torch.flatten(x, 1)

        x = self.dropin(self.classifierA1(x))
        
        x = self.dropout(self.classifierB1(x))

        x = self.droptail(self.classifierC1(x))

        x = self.droptail(self.classifierC2(x))

        x = self.classifierD1(x)
        
        return x
    