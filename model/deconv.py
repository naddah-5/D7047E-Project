import torch
import torch.nn as nn

class DeconvCNN(nn.Module):
    def __init__(self, class_count: int = 10, dropout: float = 0.5, drop_in: float = 0.8, device: str = 'cpu'):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(class_count, 21),         #fc5
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(21, 320),                 #fc1
            nn.Dropout(p=drop_in),
        )

        self.features = nn.Sequential(
            nn.ConvTranspose2d(320, 160, 5),
            nn.MaxUnpool2d(2, 2),
            nn.ConvTranspose2d(160, 60, 5),
            nn.MaxUnpool2d(2, 2),
            nn.ConvTranspose2d(60, 1, 5),
        )

    def forward(self, x: torch.Tensor, pooling_indices_1, pooling_indices_2, input_size) -> torch.Tensor:
        x = self.classifier(x)
        x = x.view(input_size) # Reshape to match the output shape of the original CNN's features
        x = self.features[0](x)
        x = self.features[1](x, pooling_indices_2)
        x = self.features[2](x)
        x = self.features[3](x, pooling_indices_1)
        x = self.features[4](x)
        return x
