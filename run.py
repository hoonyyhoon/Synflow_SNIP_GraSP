import argparse
import collections

import torch

from trainer import Trainer
from utils import check_sparsity, get_dataloader

# args
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pruning tester.")
    parser.add_argument("--SEED", default=777, type=int, help="Seed number")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use")
    parser.add_argument(
        "--model",
        default="resnet18",
        type=str,
        help="Model to test, torchvision model name",
    )
    parser.add_argument(
        "--dataset",
        default="CIFAR10",
        type=str,
        help="Dataset in torchvision.datasets ex) CIFAR10, CIFAR100, MNIST",
    )
    parser.add_argument(
        "--batch_size", default=64, type=int, help="Batch size, default: 64"
    )
    parser.add_argument(
        "--method_list",
        nargs="+",
        default=["Rand", "Mag", "Synflow"],
        type=str,  # type: ignore
        help="Pruning method(Rand/Mag/Synflow) list run sequentially."  # type: ignore
        "ex) --method_list Synflow Rand Mag",
    )
    parser.add_argument(
        "--ratio_list",
        nargs="+",
        default=[0.9, 0.95],
        type=float,  # type: ignore
        help="List of pruning ratio. ex) --ratio_list 0 0.5 0.9 0.95 0.99",  # type: ignore
    )
    parser.add_argument(
        "--epoch",
        default=50,
        type=int,
        help="Number of epochs to train, default: 50",
    )

    args = parser.parse_args()
    args.method_list = ["".join(s) for s in args.method_list]

    # device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # trainloader, testloader
    trainloader, testloader = get_dataloader(args.dataset, args.batch_size)
    methods_stat = collections.defaultdict(list)
    for method in args.method_list:
        print(f"Test {method}")
        for prune_amount in args.ratio_list:
            # https://pytorch.org/docs/stable/torchvision/models.html
            # model: resnet18 alexnet vgg16 squeezenet1_0 densenet161 inception_v3,
            # googlenet shufflenet_v2_x1_0 mobilenet_v2 resnext50_32x4d,
            # wide_resnet50_2, mnasnet1_0
            net = getattr(
                __import__("torchvision.models", fromlist=[""]), args.model
            )().to(device)

            # Apply prune
            input_shape = list(trainloader.dataset.data.shape[1:])
            if len(input_shape) == 2:
                input_shape = input_shape + [3]
            pruner = getattr(
                __import__("pruning_method." + method, fromlist=[""]), method
            )(net, device, input_shape)
            pruner.prune(amount=prune_amount)

            # Train
            trainer = Trainer(net, trainloader, testloader, device)
            test_acc = trainer.train(args.epoch)

            # Remove
            pruner.remove()

            # Validate after remove: no difference before remove since  pytorch hook works
            # print(trainer.test())
            methods_stat[method].append((test_acc, check_sparsity(net)))
            del net
            del pruner
            del trainer

        # Show Result
        for method, results in methods_stat.items():
            print(f"### [{method}] ###")
            for test_acc, sparsity in results:
                print(f"Sparsity: {sparsity:.2f}% Acc: {test_acc:.2f}% ")
