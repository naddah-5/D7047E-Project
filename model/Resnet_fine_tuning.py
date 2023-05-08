import torch
import torch.nn as nn
import torchvision

class ResNet_tune(nn.Module):
    def __init__(self, class_count: int = 2, device: str = 'cpu'):
        super(ResNet_tune, self).__init__()
        self.model = torchvision.models.resnet50(weights='DEFAULT')
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, class_count) # Added layer for our output

        self.to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x