import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, class_count: int = 2, dropout: float = 0.5, drop_in: float = 0.8, device: str = 'cpu'):
        super().__init__()

        self.features1 = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.BatchNorm2d(32),  # Added batch normalization
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.features2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding='same'),
            nn.BatchNorm2d(64),  # Added batch normalization
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.features3 = nn.Sequential(
            nn.Conv2d(64, 128, 5, padding='same'),
            nn.BatchNorm2d(128),  # Added batch normalization
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 100, 5, padding='same'),
            nn.BatchNorm2d(100),  # Added batch normalization
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.skipLayer1 = nn.Sequential(
            nn.Conv2d(32, 128, 3, padding='same'),
            nn.BatchNorm2d(128),  # Added batch normalization
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 100, 3, padding='same'),
            nn.BatchNorm2d(100),  # Added batch normalization
            nn.ReLU(),
            nn.MaxPool2d(4, 4)
        )

        self.skipLayer2 = nn.Sequential(
            nn.Conv2d(32, 128, 7, padding='same'),
            nn.BatchNorm2d(128),  # Added batch normalization
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 100, 7, padding='same'),
            nn.BatchNorm2d(100),  # Added batch normalization
            nn.ReLU(),
            nn.MaxPool2d(4, 4)
        )

        self.combine = nn.Sequential(
            nn.Conv2d(300, 100, 3),
            nn.BatchNorm2d(100),  # Added batch normalization
            nn.ReLU(),
            nn.MaxPool2d(3, 3),
            nn.Conv2d(100, 50, 3),
            nn.BatchNorm2d(50),  # Added batch normalization
            nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            #nn.Dropout(p=drop_in),
            nn.Linear(16900, 8000),
            nn.ReLU(),
            #nn.Dropout(p=dropout),
            nn.Linear(8000, 1000),
            nn.ReLU(),
            #nn.Dropout(p=dropout),
            nn.Linear(1000, 100),
            nn.ReLU(),
            #nn.Dropout(p=dropout),
            nn.Linear(100, class_count)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:


        x = self.features1(x)
        skip7x7 = x
        skip3x3 = x
        
        skip3x3 = self.skipLayer1(skip3x3)  
        #skip7x7 = self.skipLayer2(skip7x7)

        x = self.features2(x)
        x = self.features3(x)

        #Residual = self.skipLayer(Residual)  
        #print(Residual.shape)
        
        #print(x.shape)
        #print(skip3x3.shape)
        #print(skip7x7.shape)
        
        #comb = torch.cat((x, skip3x3, skip7x7), dim=1)
        #print(x.shape)
        
        #x = x + Residual
        #comb = self.combine(comb)
        #comb = (x + skip3x3 + skip7x7) / 3
        comb = (x + skip3x3) / 2
        #print(comb.shape)

        x = torch.flatten(comb, 1)
        x = self.classifier(x)
        return x
