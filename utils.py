from typing import Any, Tuple

import torch
from torch import nn
import torchvision.transforms as transforms


def expand_to_rgb(x):
    return x.repeat(3, 1, 1)

NORMALIZE_MAP = {
    "mnist": [(0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081)],
    "cifar10": [(0.491, 0.482, 0.447), (0.247, 0.243, 0.262)],
    "cifar100": [(0.507, 0.487, 0.441), (0.267, 0.256, 0.276)]
}

def get_dataloader(
    dataset: str, batch_size: int
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Sloppy dataloader."""
    # hard-coded normalizing params
    mean, std = NORMALIZE_MAP.get(dataset.lower(), [(0.507, 0.487, 0.441), (0.267, 0.256, 0.276)])
    normalize = transforms.Normalize(mean=mean,std=std)

    transform_train = transforms.Compose(
        [
            v
            for v in [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                (transforms.Lambda(expand_to_rgb) if "MNIST" in dataset else None),
                normalize,
            ]
            if v is not None
        ]
    )

    transform_test = transforms.Compose(
        [
            v
            for v in [
                transforms.ToTensor(),
                (transforms.Lambda(expand_to_rgb) if "MNIST" in dataset else None),
                normalize,
            ]
            if v is not None
        ]
    )

    trainset = getattr(__import__("torchvision.datasets", fromlist=[""]), dataset)(
        root="datasets/", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
    )

    testset = getattr(__import__("torchvision.datasets", fromlist=[""]), dataset)(
        root="datasets/", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
    )

    return trainloader, testloader


def check_sparsity(
    model: nn.Module,
    module_types: Tuple[Any, ...] = (
        nn.Conv2d,
        nn.Linear,
    ),
) -> float:
    """Get the ratio of zeros in weight masks."""
    n_zero = n_total = 0
    for param in model.parameters():
        if param is not None:
            n_zero += param.numel() - param.nonzero().size(0)
            n_total += param.numel()
    return (100.0 * n_zero / n_total) if n_total != 0 else 0.0
