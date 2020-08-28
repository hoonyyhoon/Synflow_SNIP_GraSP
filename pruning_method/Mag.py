from pruning_method.pruner import Pruner
from typing import Dict, Any, List, Iterable
import torch.nn as nn
import torch
import torch.nn.utils.prune as prune

class Mag(Pruner):
    def __init__(self, net: nn.Module, device: torch.device, input_shape: List[int]) -> None:
        """Initialize."""
        super(Mag, self).__init__(net, device, input_shape)

        self.params_to_prune = self.get_params(
            (
                (nn.Conv2d, "weight"),
                (nn.Conv2d, "bias"),
                (nn.Linear, "weight"),
                (nn.Linear, "bias")
            )
        )

    def prune(self, amount):
        """Prune with magnitude(L1)."""
        print(f"Start prune, target_sparsity: {amount*100:.2f}%")
        prune.global_unstructured(
            self.params_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
        sparsity = self.mask_sparsity()
        print(f"Pruning Done, sparsity: {sparsity:.2f}%")
