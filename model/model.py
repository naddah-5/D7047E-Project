import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, class_count: int = 2, dropout: float = 0.5, drop_in: float = 0.8, device: str = 'cpu'):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 60, 5),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(60, 160, 5),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(160, 320, 5),
        )

        #self.dropout = nn.Dropout(p=dropout)

    #    self.soft = nn.Softmax(dim=1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_in),
            nn.Linear(768320, 21),                  #fc1
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(21, class_count),             #fc5
        )

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

