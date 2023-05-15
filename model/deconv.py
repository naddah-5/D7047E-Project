import torch
import torch.nn as nn
from torch.functional import F

class DeconvCNN(nn.Module):
    def __init__(self, class_count: int = 2, dropout: float = 0.5, drop_in: float = 0.8, device: str = 'cpu'):
        super().__init__()

        
        # self.unpool6 = nn.MaxUnpool2d(2, 2)
        # self.deconv6 = nn.ConvTranspose2d(300, 250, 5)

        # self.unpool5 = nn.MaxUnpool2d(2, 2)
        # self.deconv5 = nn.ConvTranspose2d(250, 200, 5)

        self.unpool4 = nn.MaxUnpool2d(4, 4)
        self.bn4 = nn.BatchNorm2d(800)
        self.deconv4 = nn.ConvTranspose2d(800, 400, 3)
        

        self.unpool3 = nn.MaxUnpool2d(4, 4)
        self.bn3 = nn.BatchNorm2d(400)
        self.deconv3 = nn.ConvTranspose2d(400, 200, 3)
        

        self.unpool2 = nn.MaxUnpool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(200)
        self.deconv2 = nn.ConvTranspose2d(200, 100, 5)
        
        
        self.unpool1 = nn.MaxUnpool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(100)
        self.deconv1 = nn.ConvTranspose2d(100, 1, 5)




    def forward(self, x: torch.Tensor, pool_indices, input_size, sizes) -> torch.Tensor:
        
        # x = self.unpool6(x, pool_indices[5], output_size=sizes[5])
        # x = F.relu(self.deconv6(x))

        # x = self.unpool5(x, pool_indices[4], output_size=sizes[4])
        # x = F.relu(self.deconv5(x))

        x = self.unpool4(x, pool_indices[3], output_size=sizes[3])
        x = self.bn4(x)
        x = F.relu(self.deconv4(x))

        x = self.unpool3(x, pool_indices[2], output_size=sizes[2])
        x = self.bn3(x)
        x = F.relu(self.deconv3(x))

        x = self.unpool2(x, pool_indices[1], output_size=sizes[1])
        x = self.bn2(x)
        x = F.relu(self.deconv2(x))

        x = self.unpool1(x, pool_indices[0], output_size=sizes[0])
        x = self.bn1(x)
        x = F.relu(self.deconv1(x))


        
        return x