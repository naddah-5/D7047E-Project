import torch
import torch.nn as nn
import torchvision

class Inception_tune(nn.Module):
    def __init__(self, class_count: int = 2, device: str = 'cpu'):
        super(Inception_tune, self).__init__()
        self.model = torchvision.models.inception_v3(weights='DEFAULT')
        # Handle the auxilary net
        num_ftrs = self.model.AuxLogits.fc.in_features
        self.model.AuxLogits.fc = nn.Linear(num_ftrs, class_count)
        # Handle the primary net
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs,class_count)

        self.to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x