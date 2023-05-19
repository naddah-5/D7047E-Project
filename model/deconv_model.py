import torch
import torch.nn as nn
from torch.functional import F

class DeconvCNN(nn.Module):
    def __init__(self, class_count: int = 2, dropout: float = 0.5, drop_in: float = 0.8, device: str = 'cpu'):
        super().__init__()
        self.dropin = nn.Dropout(p=drop_in)
        self.dropout = nn.Dropout(p=dropout)
        #self.droptail = nn.Dropout(p=drop_tail)
        
        self.deFeatureA1 = nn.Sequential(
            nn.BatchNorm2d(200),
            nn.ConvTranspose2d(200, 1, 7),
            nn.BatchNorm2d(1),
            nn.ConvTranspose2d(1, 1, 1),
            nn.LeakyReLU(),

        )
        self.unpool1_5 = nn.MaxUnpool2d(4, 4)

        self.deFeatureA1_5 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ConvTranspose2d(1, 200, 1),
            nn.LeakyReLU(),
        )

        self.deFeatureA2 = nn.Sequential(
            nn.BatchNorm2d(400),
            nn.ConvTranspose2d(400, 1, 5),
            nn.LeakyReLU(),
            )
        
        self.unpool2_5 = nn.MaxUnpool2d(4, 4)

        self.deFeatureA2_5 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ConvTranspose2d(1, 400, 1),
            nn.LeakyReLU(),
            )


        self.deFeatureA3 = nn.Sequential(
            
            nn.BatchNorm2d(100),
            nn.ConvTranspose2d(100, 400, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(400),
            nn.ConvTranspose2d(400, 1, 3),
            nn.LeakyReLU(),
           
        )

        self.to(device)
        

    def forward(self, x: torch.Tensor, pool_indices, input_size, sizes) -> torch.Tensor:
        
        x = self.deFeatureA3(x)

        x = self.deFeatureA2_5(x)
        x = self.unpool2_5(x, pool_indices[1], sizes[1])
        x = self.deFeatureA2(x)

        x = self.deFeatureA1_5(x)
        x = self.unpool1_5(x, pool_indices[0], sizes[0])
        x = self.deFeatureA1(x)
        
        return x