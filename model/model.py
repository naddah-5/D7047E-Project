import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, class_count: int = 2, dropout: float = 0.5, drop_in: float = 0.8, device: str = 'cpu'):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 100, 5)
        self.bn1 = nn.BatchNorm2d(100)
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv2 = nn.Conv2d(100, 200, 5)
        self.bn2 = nn.BatchNorm2d(200)
        self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv3 = nn.Conv2d(200, 400, 3)
        self.bn3 = nn.BatchNorm2d(400)
        self.pool3 = nn.MaxPool2d(4, 4, return_indices=True)

        self.conv4 = nn.Conv2d(400, 800, 3)
        self.bn4 = nn.BatchNorm2d(800)
        self.pool4 = nn.MaxPool2d(4, 4, return_indices=True)
        
        # self.conv5 = nn.Conv2d(200, 250, 5)
        # self.bn5 = nn.BatchNorm2d(250)
        # self.pool5 = nn.MaxPool2d(2, 2, return_indices=True)

        # self.conv6 = nn.Conv2d(250, 300, 5)
        # self.bn6 = nn.BatchNorm2d(300)
        # self.pool6 = nn.MaxPool2d(2, 2, return_indices=True)

        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_in),
            nn.Linear(7200, 6400),                  #fc1
            nn.LeakyReLU(),
            nn.Dropout(p=drop_in),
            nn.Linear(6400, 1133),                  #fc2
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(1133, 370),                  #fc3
            nn.LeakyReLU(),
            nn.Dropout(p=drop_in),
            nn.Linear(370, 120),                  #fc4
            nn.LeakyReLU(),
            nn.Dropout(p=drop_in),
            nn.Linear(120, 40),                  #fc5
            nn.LeakyReLU(),
            nn.Dropout(p=drop_in),
            nn.Linear(40, class_count),             #fc6
        )

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
            sizes = []
            x = self.conv1(x)
            sizes.append(x.shape)
            x = self.bn1(x)
            x, pool1_indices = self.pool1(x)

            x = self.conv2(x)
            sizes.append(x.shape)
            x = self.bn2(x)
            x, pool2_indices = self.pool2(x)
            
            x = self.conv3(x)
            sizes.append(x.shape)
            x = self.bn3(x)
            x, pool3_indices = self.pool3(x)
            
            x = self.conv4(x)
            sizes.append(x.shape)
            x = self.bn4(x)
            x, pool4_indices = self.pool4(x)
            
            # x = self.conv5(x)
            # sizes.append(x.shape)
            # #x = self.bn5(x)
            # x, pool5_indices = self.pool5(x)
            
            # x = self.conv6(x)
            # sizes.append(x.shape)
            # #x = self.bn6(x)
            # x, pool6_indices = self.pool6(x) ##???????????

            our_features = x
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            
            pool_indices = (pool1_indices, pool2_indices, pool3_indices, pool4_indices)#, pool5_indices, pool6_indices)

            return x, our_features, pool_indices, sizes
