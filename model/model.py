import torch
import torch.nn as nn
import copy


class CNN(nn.Module):
    def __init__(self, class_count: int = 2, dropout: float = 0.8, drop_in: float = 0.9, drop_tail = 0.4, device: str = 'cpu'):
        super().__init__()
        self.dropin = nn.Dropout(p=drop_in)
        self.dropout = nn.Dropout(p=dropout)
        self.droptail = nn.Dropout(p=drop_tail)

        self.A0 = nn.Sequential(
            nn.Conv2d(1, 200, 7),
            nn.LeakyReLU(),
            nn.MaxPool2d(4, 4))

        self.A1 = nn.Sequential(
            nn.Conv2d(200, 200, 3, padding='same'),
            nn.LeakyReLU())

        self.A2 = nn.Sequential(
            nn.Conv2d(200, 200, 5),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2))

        self.A3 = nn.Sequential(
            nn.Conv2d(200, 200, 3, padding='same'),
            nn.LeakyReLU())

        self.A4 = nn.Sequential(
            nn.Conv2d(200, 200, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2))

        self.A5 = nn.Sequential(
            nn.Conv2d(200, 200, 3, padding='same'),
            nn.LeakyReLU())

        self.A6 = nn.Sequential(
            nn.Conv2d(200, 200, 3),
            nn.LeakyReLU())

        self.A7 = nn.Sequential(
            nn.Conv2d(200, 200, 3, padding='same'),
            nn.LeakyReLU())

        self.A8 = nn.Sequential(
            nn.Conv2d(200, 200, 3),
            nn.LeakyReLU())

        self.A9 = nn.Sequential(
            nn.Conv2d(200, 200, 3, padding='same'),
            nn.LeakyReLU())

        # Classifiers

        self.classifier1 = nn.Sequential(
            nn.Linear(9_800, 5_000),
            nn.LeakyReLU())

        self.classifier2 = nn.Sequential(
            nn.Linear(5_000, 3_000),
            nn.LeakyReLU())

        self.classifier3 = nn.Sequential(
            nn.Linear(3_000, 3_000),
            nn.Tanh())

        self.classifier4 = nn.Sequential(
            nn.Linear(3_000, 3_000),
            nn.Tanh())

        self.classifier5 = nn.Sequential(
            nn.Linear(3_000, 100),
            nn.LeakyReLU())

        self.classifier6 = nn.Sequential(
            nn.Linear(100, class_count),
            nn.Softmax(dim=1))

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.A0(x)
        x1 = self.A1(x0)
        x2 = self.A2((x0 + x1) / 2)
        x3 = self.A3(x2)
        x4 = self.A4((x2 + x3) / 2)
        x5 = self.A5(x4)
        x6 = self.A6((x4 + x5) / 2)
        x7 = self.A7(x6)
        x8 = self.A8((x6 + x7) / 2)
        x9 = self.A9(x8)

        x = torch.flatten((x8 + x9) / 2, 1)

        x = self.dropin(self.classifier1(x))
        x = self.dropout(self.classifier2(x))
        x = self.droptail(self.classifier3(x))
        x = self.droptail(self.classifier4(x))
        x = self.droptail(self.classifier5(x))
        x = self.classifier6(x)

        return x
