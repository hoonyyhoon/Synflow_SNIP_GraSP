from typing import List

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from pruning_method.pruner import Pruner


class Mag(Pruner):
    def __init__(
        self, net: nn.Module, device: torch.device, input_shape: List[int]
    ) -> None:
        """Initialize."""
        super(Mag, self).__init__(net, device, input_shape)

        self.params_to_prune = self.get_params(
            (
                (nn.Conv2d, "weight"),
                (nn.Conv2d, "bias"),
                (nn.Linear, "weight"),
                (nn.Linear, "bias"),
            )
        )

    def get_prune_score(self):
        """Get prune score."""
        pass

    def prune(self, amount: int):
        """Prune with magnitude(L1)."""
        print(f"Start prune, target_sparsity: {amount*100:.2f}%")
        prune.global_unstructured(
            self.params_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
        sparsity = self.mask_sparsity()
        print(f"Pruning Done, sparsity: {sparsity:.2f}%")
