import torch
import torch.nn as nn
import copy

class CNN(nn.Module):
    def __init__(self, class_count: int = 2, dropout: float = 0.8, drop_in: float = 0.9, device: str = 'cpu'):
        super().__init__()


        
        self.A1 = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.BatchNorm2d(1), 
            nn.ReLU(),

            nn.Conv2d(1, 100, 7),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(100, 100, 5),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(100, 100, 3),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.B1 = nn.Sequential(
            nn.Conv2d(100, 500, 3),                 # B1
            nn.BatchNorm2d(500),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(500, 500, 3),
            nn.BatchNorm2d(500),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier1 = nn.Sequential(
            #nn.Dropout(p=drop_in),
            
            nn.Linear(8_000, 5_000),                 # fc1
            nn.ReLU(),
            #nn.Dropout(p=dropout),

            #nn.Linear(8_000, 8_000),
            #nn.LeakyReLU(),
            #nn.Dropout(p=dropout),

            nn.Linear(5_000, 1_000),
            nn.ReLU(),
            #nn.Dropout(p=dropout),

            nn.Linear(1_000, 500),
            nn.ReLU(),
            #nn.Dropout(p=dropout),
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(500, 100),
            nn.ReLU(),
            #nn.Dropout(p=dropout),
        )


        self.classifier3 = nn.Sequential(
              
            nn.Linear(100, class_count),             # fc5
            nn.Softmax(dim=1)
        )
        self.dropout = nn.Dropout(p=dropout)
        self.to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.A1(x)

        x = self.B1(x)

        x = torch.flatten(x, 1)
        x = self.classifier1(x)
        x = self.dropout(self.classifier2(x))
        x = self.classifier3(x)
        return x