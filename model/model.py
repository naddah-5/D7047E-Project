import torch
import torch.nn as nn
import copy

class CNN(nn.Module):
    def __init__(self, class_count: int = 2, dropout: float = 0.8, drop_in: float = 0.9, device: str = 'cpu'):
        super().__init__()


        
        self.A1 = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.Conv2d(1, 100, 7),
        )
        self.poolA1 = nn.MaxPool2d(2, 2, return_indices=True)

        self.A1_5 = nn.Sequential(
            nn.Conv2d(100, 100, 5),
        )
        self.poolA1_5 = nn.MaxPool2d(2, 2, return_indices=True)

        self.A1_5_5 = nn.Sequential(
            nn.Conv2d(100, 100, 3),
        )
        self.poolA1_5_5 = nn.MaxPool2d(2, 2, return_indices=True)

        self.B1 = nn.Sequential(
            nn.Conv2d(100, 500, 3),                 # B1
        )
        self.poolB1 = nn.MaxPool2d(2, 2, return_indices=True)
        
        self.B1_5 = nn.Sequential(
            nn.Conv2d(500, 500, 3),
        )
        self.poolB1_5 = nn.MaxPool2d(2, 2, return_indices=True)



        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_in),
            
            nn.Linear(8_000, 8_000),                 # fc1
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(8_000, 8_000),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(8_000, 5_000),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(5_000, 5_000),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(5_000, 1_000),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(1_000, class_count),             # fc5
        )

        self.to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sizes = []

        x = self.A1(x)
        sizes.append(x.shape)
        x , pool_indices1  = self.poolA1(x)

        x = self.A1_5(x)
        sizes.append(x.shape)
        x , pool_indices2  = self.poolA1_5(x)

        x = self.A1_5_5(x)
        sizes.append(x.shape)
        x , pool_indices3  = self.poolA1_5_5(x)

        x = self.B1(x)
        sizes.append(x.shape)
        x , pool_indices4  = self.poolB1(x)

        x = self.B1_5(x)
        sizes.append(x.shape)
        x , pool_indices5  = self.poolB1_5(x)

        our_features = x
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        
        pool_indices = (pool_indices1, pool_indices2, pool_indices3, pool_indices4, pool_indices5)
        return x, our_features, pool_indices, sizes