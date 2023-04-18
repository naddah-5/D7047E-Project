import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def load_dataset(scale: list = [32, 32], batch_size: int = 10, ):
    transform = transforms.Compose([
        transforms.Resize(scale),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    
    trainset = datasets.ImageFolder('dataset/chest_xray/train/', transform)
    valset = datasets.ImageFolder('dataset/chest_xray/val', transform)
    testset = datasets.ImageFolder('dataset/chest_xray/test', transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    

    return train_loader, validation_loader, test_loader