import torch
import torchvision
import torchvision.transforms as transforms
import ssl 

def __init__(self, batch_size, MNIST: bool = False, SVHN: bool = False, CIFAR10: bool = False, scale: int = 32):
    self.scale = scale
    self.batch_size = batch_size

def load_dataset(batch_size: int = 10):
    transform = transforms.Compose([
        transforms.Resize(self.scale),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_val_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    proportions = [.40, .60]
    lengths = [int(p * len(test_val_set)) for p in proportions]
    lengths[-1] = len(test_val_set) - sum(lengths[:-1])

    validation_set , test_set = torch.utils.data.random_split(test_val_set, lengths)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=self.batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=2)
    
    
    labels = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, validation_loader, test_loader, labels
