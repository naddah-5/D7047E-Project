import torch
import torch.nn as nn
import copy

class CNN(nn.Module):
    def __init__(self, class_count: int = 2, dropout: float = 0.6, drop_in: float = 0.8, drop_tail = 0.5, device: str = 'cpu'):
        super().__init__()

        self.dropin = nn.Dropout(p=drop_in)
        self.dropout = nn.Dropout(p=dropout)
        self.droptail = nn.Dropout(p=drop_tail)
        
        self.FeatureA1 = nn.Sequential(
            nn.Conv2d(1, 10, 1),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 400, 5),
            nn.BatchNorm2d(400),
            nn.LeakyReLU(),
            nn.MaxPool2d(4, 4),
            
            nn.Conv2d(400, 10, 1),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(),
        )


        self.FeatureA2 = nn.Sequential(
            nn.Conv2d(10, 400, 5),
            nn.BatchNorm2d(400),
            nn.LeakyReLU(),
            nn.MaxPool2d(4, 4),

            nn.Conv2d(400, 10, 1),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(),
            )


        self.FeatureA3 = nn.Sequential(
            nn.Conv2d(10, 400, 5),
            nn.BatchNorm2d(400),
            nn.LeakyReLU(),

            nn.Conv2d(400, 10, 1),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(),
        )

        self.FeatureA4 = nn.Sequential(
            nn.Conv2d(10, 400, 5),
            nn.BatchNorm2d(400),
            nn.LeakyReLU(),

            nn.Conv2d(400, 100, 1),
            nn.BatchNorm2d(100),
            nn.LeakyReLU(),
        )

        self.classifierA1 = nn.Sequential(
            nn.Linear(1_600, 1_800),
            nn.LeakyReLU(),
        )

        self.classifierB1 = nn.Sequential(
            nn.Linear(1_800, 1_800),
            nn.LeakyReLU(),
        )


        self.classifierC1 = nn.Sequential(
            nn.Linear(1_800, 1_800),
            nn.Tanh(),
        )

        self.classifierC2 = nn.Sequential(
            nn.Linear(1_800, 1_800),
            nn.LeakyReLU(),
        )

        self.classifierD1 = nn.Sequential(
            nn.Linear(1_800, class_count),
            nn.Softmax(dim=1)
        )

        self.to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.FeatureA1(x)
        x = self.FeatureA2(x)
        x = self.FeatureA3(x)
        x = self.FeatureA4(x)

        x = torch.flatten(x, 1)

        x = self.dropin(self.classifierA1(x))
        
        x = self.dropout(self.classifierB1(x))

        x = self.droptail(self.classifierC1(x))

        x = self.droptail(self.classifierC2(x))

        x = self.classifierD1(x)
        
        return x
    