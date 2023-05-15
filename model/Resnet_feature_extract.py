import torch
import torch.nn as nn
import torchvision

class ResNet_extract(nn.Module):
    def __init__(self, class_count: int = 2, device: str = 'cpu'):
        super(ResNet_extract, self).__init__()
        self.model = torchvision.models.resnet50(weights='DEFAULT')
        for param in self.model.parameters():
            param.requires_grad = False #turn training layers 
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, class_count) # added layers for outputs

        self.to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x