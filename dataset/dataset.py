import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import WeightedRandomSampler
from application.params import update_params

#def load_dataset(scale: list = [224, 224], batch_size: int = 8):
def load_dataset(scale: list = [224, 224], batch_size: int = 8):
    transform = transforms.Compose([
        # transforms.RandomApply(
        #     transforms.RandomRotation(enumerate(range(1,359))),
        #     p=0.01
        # ),
        # transforms.RandomInvert(p=0.5),
        # transforms.RandomPerspective(distortion_scale=0.1, p=0.01),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomHorizontalFlip(p=0.1),
        # transforms.RandAugment(),
        transforms.Resize(scale),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    base_transform = transforms.Compose([
        transforms.Resize(scale),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    trainset = datasets.ImageFolder('dataset/temp/chest_xray_both/train', transform)
    valset = datasets.ImageFolder('dataset/temp/chest_xray_both/val', base_transform)
    testset = datasets.ImageFolder('dataset/temp/chest_xray_both/test', base_transform)

    class_labels = trainset.targets
    num_normal = class_labels.count(0)
    num_pneumonia = class_labels.count(1)
    min_len = min(num_normal, num_pneumonia)
    labels = [0] * min_len + [1] * min_len
    weights = torch.DoubleTensor([1.0 / min_len] * min_len * 2)

    train_sampler = WeightedRandomSampler(weights, len(labels), replacement=False)

    class_labels = testset.targets
    num_normal = class_labels.count(0)
    num_pneumonia = class_labels.count(1)
    min_len = min(num_normal, num_pneumonia)
    labels = [0] * min_len + [1] * min_len
    weights = torch.DoubleTensor([1.0 / min_len] * min_len * 2)

    test_sampler = WeightedRandomSampler(weights, len(labels), replacement=False)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, sampler=test_sampler, num_workers=2)


    new_params = {
        'scale': scale
    }
    
    update_params(new_params)
    return train_loader, validation_loader, test_loader
