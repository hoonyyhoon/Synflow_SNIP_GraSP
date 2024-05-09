from typing import Any, Tuple

import torch
from torch import nn
import torchvision.transforms as transforms


def expand_to_rgb(x):
    return x.repeat(3, 1, 1)


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
        trainset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    testset = getattr(__import__("torchvision.datasets", fromlist=[""]), dataset)(
        root="datasets/", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=4
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
