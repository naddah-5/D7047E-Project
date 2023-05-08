import torch
import torch.nn as nn
import copy

class CNN(nn.Module):
    def __init__(self, class_count: int = 2, dropout: float = 0.8, drop_in: float = 0.9, device: str = 'cpu'):
        super().__init__()


        
        self.A0 = nn.Sequential(
            nn.Conv2d(1, 100, 3),
            nn.Dropout2d(p=drop_in),
        )

        self.A1 = nn.Sequential(
            nn.Conv2d(100, 100, 3, padding='same'),
            nn.Dropout2d(p=dropout),
        )

        self.A2 = nn.Sequential(
            nn.Conv2d(100, 100, 3),
            nn.Dropout2d(p=dropout),
        )

        self.A3 = nn.Sequential(
            nn.Conv2d(100, 100, 3, padding='same'),
            nn.Dropout2d(p=dropout)
        )

        self.A4 = nn.Sequential(
            nn.Conv2d(100, 100, 3),
            nn.Dropout2d(p=dropout),
        )

        self.A5 = nn.Sequential(
            nn.Conv2d(100, 100, 3, padding='same'),
            nn.Dropout2d(p=dropout)
        )

        self.A6 = nn.Sequential(
            nn.Conv2d(100, 100, 3),
            nn.Dropout2d(p=dropout),
            nn.MaxPool2d(2, 2)
        )

        self.A7 = nn.Sequential(
            nn.Conv2d(100, 100, 3, padding='same'),
            nn.Dropout2d(p=dropout)
        )

        self.A8 = nn.Sequential(
            nn.Conv2d(100, 100, 3),
            nn.Dropout2d(p=dropout),
            nn.MaxPool2d(2, 2)
        )

        self.A9 = nn.Sequential(
            nn.Conv2d(100, 100, 3, padding='same'),
            nn.Dropout2d(p=dropout)
        )

        self.B1 = nn.Sequential(
            nn.Conv2d(100, 100, 3),                 # B1
            nn.MaxPool2d(2, 2),
            nn.Conv2d(100, 50, 3),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_in),
            
            nn.Linear(6_050, 1_000),                 # fc1
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
        x = self.A0(x)
        skipX = copy.copy(x)

        x = self.A1(x)

        x = torch.add(x, skipX)
        x = self.A2(x)
        skipX = copy.copy(x)

        x = self.A3(x)

        x = torch.add(x, skipX)
        x = self.A4(x)
        skipX = copy.copy(x)

        x = self.A5(x)
        
        x = torch.add(x, skipX)
        x = self.A6(x)
        skipX = copy.copy(x)

        x = self.A7(x)

        x = torch.add(x, skipX)
        x = self.A8(x)
        skipX = copy.copy(x)

        x = self.A9(x)

        x = torch.add(x, skipX)

        B = self.B1(x)

        x = torch.flatten(B, 1)
        x = self.classifier(x)
        return x

