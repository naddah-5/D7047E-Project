import torch
import torch.nn as nn
import torchvision

class Inception_extract(nn.Module):
    def __init__(self, class_count: int = 2, device: str = 'cpu'):
        super(Inception_extract, self).__init__()
        self.model = torchvision.models.inception_v3(weights='DEFAULT')
        self.model.aux_logits = False
        for param in self.model.parameters():
            param.requires_grad = False
        # Handle the primary net
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs,class_count)

        self.to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = self.model(x)
        return x, y