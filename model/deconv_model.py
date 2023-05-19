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
            nn.ConvTranspose2d(100, 1, 7),
            nn.ConvTranspose2d(1, 1, 1),
        )
        self.unpoolA1 = nn.MaxUnpool2d(2, 2)

        self.deFeatureA1_5 = nn.Sequential(
            nn.ConvTranspose2d(100, 100, 5),
        )
        self.unpoolA1_5 = nn.MaxUnpool2d(2, 2)

        self.deFeatureA1_5_5 = nn.Sequential(
            nn.ConvTranspose2d(100, 100, 3),
        )
        self.unpoolA1_5_5 = nn.MaxUnpool2d(2, 2)

        self.deFeatureB1 = nn.Sequential(
            nn.ConvTranspose2d(500, 100, 3),
        )
        self.unpoolB1 = nn.MaxUnpool2d(2, 2)

        self.deFeatureB1_5 = nn.Sequential(
            nn.ConvTranspose2d(500, 500, 3),
        )
        self.unpoolB1_5 = nn.MaxUnpool2d(2, 2)
        
        self.to(device)
        

    def forward(self, x: torch.Tensor, pool_indices, sizes) -> torch.Tensor:
        
        x = self.unpoolB1_5(x, pool_indices[4], sizes[4])
        x = self.deFeatureB1_5(x)
        
        x = self.unpoolB1(x, pool_indices[3], sizes[3])
        x = self.deFeatureB1(x)

        x = self.unpoolA1_5_5(x, pool_indices[2], sizes[2])
        x = self.deFeatureA1_5_5(x)

        x = self.unpoolA1_5(x, pool_indices[1], sizes[1])
        x = self.deFeatureA1_5(x)

        x = self.unpoolA1(x, pool_indices[0], sizes[0])
        x = self.deFeatureA1(x)
        
        return x