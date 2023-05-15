import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, class_count: int = 2, dropout: float = 0.5, drop_in: float = 0.8, device: str = 'cpu'):
        super().__init__()

        self.features1 = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )
        
        self.features2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )
        
        self.features3 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            #nn.MaxPool2d(2, 2),
            nn.ReLU()
        )
        
        self.skipLayer = nn.Sequential(
            nn.Conv2d(32, 128, 7),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_in),
            nn.Linear(346112, 100),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(100, class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.features1(x)
        Residual = x

        x = self.features2(x)
        x = self.features3(x)

        Residual = self.skipLayer(Residual)  
        #print(Residual.shape)
        
        #print(x.shape)
        x = x + Residual

        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
