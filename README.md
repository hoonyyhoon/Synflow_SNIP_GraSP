# Comparison of Synflow/SNIP/GraSP 
Implementation of Synaptic flow, Single-shot Network Pruning, Gradient Signal Preservation in pytorch v2.3.  
Aims to compare pruning method.

1. Synaptic Flow: Pruning Neural Networks Without Any Data By Iteratively Conserving Synaptic Flow(NeurIPS 2020)
Paper: https://arxiv.org/pdf/2006.05467.pdf  
Official repo: https://github.com/ganguli-lab/Synaptic-Flow  

2. SNIP: Single-shot Network Pruning based on Connection Sensitivity(ICLR2019)  
Paper: https://arxiv.org/pdf/1810.02340.pdf
Official repo: https://github.com/namhoonlee/snip-public

3. GraSP: Picking Winning Tickets Before Training By Preserving Gradient Flow(ICLR2020)  
Paper: https://openreview.net/pdf?id=SkgsACVKPH  
Official repo: https://github.com/alecwangcq/GraSP  

## Onging process
Follwing pruning method(Pruning at initialization prior to training or while training) will be implemented.
 - [x] Synaptic flow
 - [ ] GraSP
 - [x] SNIP
 - [x] Random
 - [x] Magnitude
 - [ ] Plot

## Run
```bash
❯ python run.py --help
usage: run.py [-h] [--SEED SEED] [--gpu GPU] [--model MODEL]
              [--dataset DATASET] [--batch_size BATCH_SIZE]
              [--method_list METHOD_LIST [METHOD_LIST ...]]
              [--ratio_list RATIO_LIST [RATIO_LIST ...]]

Pruning tester.

optional arguments:
  -h, --help            show this help message and exit
  --SEED SEED           Seed number
  --gpu GPU             GPU id to use
  --model MODEL         Model to test, torchvision model name
  --dataset DATASET     Dataset in torchvision.datasets ex) CIFAR10, CIFAR100,
                        MNIST
  --batch_size BATCH_SIZE
                        Batch size, default: 128
  --method_list METHOD_LIST [METHOD_LIST ...]
                        Pruning method(Rand/Mag/Synflow) list run
                        sequentially. ex) --method_list Synflow Rand Mag
  --ratio_list RATIO_LIST [RATIO_LIST ...]
                        List of pruning ratio. ex) --ratio_list 0 0.5 0.9 0.95
                        0.99
```

Example
```bash
python run.py --model resnet18 --dataset MNIST --method_list Rand Mag Synflow SNIP --ratio_list 0.5 0.9 0.99  
```

