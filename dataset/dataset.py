import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import WeightedRandomSampler


def load_dataset(scale: list = [224, 224], batch_size: int = 10):
    transform = transforms.Compose([
        transforms.Resize(scale),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    trainset = datasets.ImageFolder('dataset/chest_xray/train', transform)
    valset = datasets.ImageFolder('dataset/chest_xray/val', transform)
    testset = datasets.ImageFolder('dataset/chest_xray/test', transform)

    weights, labels = sample(trainset, 1)
    train_sampler = WeightedRandomSampler(weights, len(labels), replacement=False)

    weights, labels = sample(testset, 1)
    test_sampler = WeightedRandomSampler(weights, len(labels), replacement=False)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, sampler=test_sampler, num_workers=2)

    return train_loader, validation_loader, test_loader


# size is the % of total dataset. Ex. 1 = 100% and 0.5 = 50%
def sample(dataset, size):
    class_labels = dataset.targets
    num_normal = class_labels.count(0)
    num_pneumonia = class_labels.count(1)
    min_len = min(num_normal, num_pneumonia) * size
    labels = [0] * min_len + [1] * min_len
    weights = torch.DoubleTensor([1.0 / min_len] * min_len * 2)

    return weights, labels