from typing import Tuple

import torch
import torchvision.transforms as transforms


def get_dataloader(
    dataset: str, batch_size: int
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Sloppy dataloader."""
    # hard-coded normalizing params
    normalize = transforms.Normalize(
        mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]
    )

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            if "MNIST" in dataset
            else None,
            normalize,
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            if "MNIST" in dataset
            else None,
            normalize,
        ]
    )

    trainset = getattr(__import__("torchvision.datasets", fromlist=[""]), dataset)(
        root="datasets/", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    testset = getattr(__import__("torchvision.datasets", fromlist=[""]), dataset)(
        root="datasets/", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return trainloader, testloader
