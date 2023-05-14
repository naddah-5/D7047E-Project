import torch
import torch.nn as nn
import copy

class CNN(nn.Module):
    def __init__(self, class_count: int = 2, dropout: float = 0.6, drop_in: float = 0.5, device: str = 'cpu'):
        super().__init__()


        
        self.A1 = nn.Sequential(
            #nn.Conv2d(1, 1, 1),
            nn.Conv2d(1, 200, 7),
            nn.MaxPool2d(4, 4),
            #nn.Conv2d(100, 100, 1),
            nn.Conv2d(200, 200, 5),
            nn.MaxPool2d(4, 4),
            #nn.Conv2d(100, 100, 1),
            nn.Conv2d(200, 1_000, 3),
        )

        self.classifier = nn.Sequential(
            
            nn.Linear(9_000, 8_192),
            nn.LeakyReLU(),
            nn.Dropout(p=drop_in),

            nn.Linear(8_192, 8_192),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(8_192, 4_096),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(4_096, 2_048),
            nn.Tanh(),
            nn.Dropout(p=dropout),

            nn.Linear(2_048, 2_048),
            nn.Tanh(),
            nn.Dropout(p=dropout),

            nn.Linear(2_048, 128),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(128, class_count),             # fc5
        )

        self.to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.A1(x)

        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    

 # Less dropout, smaller dimension collapse, slightly deeper and wider.