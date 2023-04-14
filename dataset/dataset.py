import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import ssl 

def load_dataset(batch_size: int = 10):
    transform = transforms.Compose([
    #    transforms.Resize(self.scale),
    #    transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    
    trainset = datasets.ImageFolder('chest_xray/test', transform)
    valset = datasets.ImageFolder('chest_xray/val', transform)
    testset = datasets.ImageFolder('chest_xray/test', transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    
    labels = ('Happy', 'Sad')

    return train_loader, validation_loader, test_loader, labels
