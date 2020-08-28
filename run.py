import torch
import argparse
from trainer import Trainer
from utils import get_dataloader

# args
parser = argparse.ArgumentParser(description="Pruning tester.")
parser.add_argument("--SEED", default=777, type=int, help="Seed number")
parser.add_argument("--gpu", default=0, type=int, help="GPU id to use")
parser.add_argument("--model", default="resnet18", type=str, help="Model to test, torchvision model name")
parser.add_argument("--dataset", default='CIFAR10', type=str, help="Dataset in torchvision.datasets ex) CIFAR10, CIFAR100, MNIST")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size, default: 64")
parser.add_argument("--method_list", nargs='+', default=['Rand', 'Mag', 'Synflow'], type=list, help="Pruning method(Rand/Mag/Synflow) list run sequentially. ex) --method_list Synflow Rand Mag")
parser.add_argument("--ratio_list", nargs='+', default=[0, 0.5, 0.9, 0.95, 0.99], type=list, help= "List of pruning ratio. ex) --ratio_list 0 0.5 0.9 0.95 0.99")
args = parser.parse_args()
args.method_list = [''.join(s) for s in args.method_list]
args.ratio_list = [float(''.join(s)) for s in args.ratio_list]

# device
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# trainloader, testloader
trainloader, testloader = get_dataloader(args.dataset, args.batch_size)

acc_methods = dict()
for method in args.method_list:
    print(f'Test {method}')
    test_acc = []
    for prune_amount in args.ratio_list:
        # https://pytorch.org/docs/stable/torchvision/models.html
        # model: resnet18 alexnet vgg16 squeezenet1_0 densenet161 inception_v3 googlenet shufflenet_v2_x1_0 mobilenet_v2 resnext50_32x4d, wide_resnet50_2, mnasnet1_0
        net = getattr(__import__("torchvision.models", fromlist=[""]), args.model)().to(device)

        # Apply prune
        input_shape = list(trainloader.dataset.__getitem__(0)[0][:].shape)
        pruner = getattr(
            __import__("pruning_method." + method, fromlist=[""]), method
        )(net, device, input_shape)
        pruner.prune(amount=prune_amount)

        # Train
        trainer = Trainer(net, trainloader, testloader, device)
        test_acc.append(trainer.train(1))

        del net
        del pruner
        del trainer
    acc_methods[method] = test_acc

# Show Result
for method, test_accs in acc_methods.items():
    print(f'### [{method}] ###')
    for acc, prune_amount in zip(test_accs, args.ratio_list):
        print(f"Sparsity: {100*prune_amount:.2f}% Acc: {acc:.2f}% ")
