import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import WeightedRandomSampler
from application.params import update_params


def load_dataset(scale: list = [256, 256], batch_size: int = 10): #512
    transform = transforms.Compose([
        transforms.Resize(scale),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    trainset = datasets.ImageFolder('dataset/RSNA_DATASET_OUR_(split_into_sets)/training_set', transform)
    valset = datasets.ImageFolder('dataset/RSNA_DATASET_OUR_(split_into_sets)/validation_set', transform)
    testset = datasets.ImageFolder('dataset/RSNA_DATASET_OUR_(split_into_sets)/fake_test_set', transform)

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
