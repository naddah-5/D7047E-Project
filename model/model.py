import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, class_count: int = 2, dropout: float = 0.8, drop_in: float = 0.9, device: str = 'cpu'):
        super().__init__()


        
        self.features = nn.Sequential(
            nn.Conv2d(1, 100, 5),

            nn.MaxPool2d(2, 2),
            nn.Conv2d(100, 200, 5),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(200, 200, 5),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(200, 200, 5),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(200, 200, 5),
            nn.MaxPool2d(2, 2)
        )

        #self.dropout = nn.Dropout(p=dropout)

    #    self.soft = nn.Softmax(dim=1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_in),
            
            nn.Linear(1_800, 1_800),                  #fc1
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(1_800, 1_800),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(1_800, 1_800),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(1_800, 1_800),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(1_800, 100),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(100, class_count),             #fc5
        )

        self.to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

