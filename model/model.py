import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, class_count: int = 2, dropout: float = 0.8, drop_in: float = 0.9, device: str = 'cpu'):
        super().__init__()


        
        self.A1 = nn.Sequential(
            nn.Conv2d(1, 300, 3, padding='same'),
            nn.MaxPool2d(2, 2),
        )

        self.A2 = nn.Sequential(
            nn.Conv2d(1, 300, 5, padding='same'),
            nn.MaxPool2d(2, 2),
        )

        self.A3 = nn.Sequential(
            nn.Conv2d(1, 300, 7, padding='same'),
            nn.MaxPool2d(2, 2),
        )

        self.B1 = nn.Sequential(
            nn.Conv2d(900, 100, 3),     # B1
            nn.MaxPool2d(3, 3),
            nn.Conv2d(100, 50, 3),
            nn.MaxPool2d(4, 4)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_in),
            
            nn.Linear(3200, 1_000),                  #fc1
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

            nn.Linear(100, class_count),             #fc5
        )

        self.to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xA1 = self.A1(x)
        xA2 = self.A2(x)
        xA3 = self.A3(x)

        A = torch.cat((xA1, xA2, xA3), dim=1)

        B = self.B1(A)

        x = torch.flatten(B, 1)
        x = self.classifier(x)
        return x

