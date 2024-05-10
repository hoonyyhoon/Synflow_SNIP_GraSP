from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from pruning_method.pruner import Pruner


class SNIP(Pruner):
    def __init__(
        self, net: nn.Module, device: torch.device, input_shape: List[int], dataloader: torch.utils.data.DataLoader, criterion
    ) -> None:
        super(SNIP, self).__init__(net, device, input_shape, dataloader, criterion)

        self.params_to_prune = self.get_params(
            (
                (nn.Conv2d, "weight"),
                (nn.Conv2d, "bias"),
                (nn.Linear, "weight"),
                (nn.Linear, "bias"),
            )
        )
        self.params_to_prune_orig = self.get_params(
            (
                (nn.Conv2d, "weight_orig"),
                (nn.Conv2d, "bias_orig"),
                (nn.Linear, "weight_orig"),
                (nn.Linear, "bias_orig"),
            )
        )
        prune.global_unstructured(
            self.params_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=0.0,
        )
        # https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
        # To get gradient of each weight(after prune at least one time)
        self.params_to_prune_orig = self.get_params(
            (
                (nn.Conv2d, "weight_orig"),
                (nn.Conv2d, "bias_orig"),
                (nn.Linear, "weight_orig"),
                (nn.Linear, "bias_orig"),
            )
        )

    def prune(self, amount: int):
        unit_amount = 1 - ((1 - amount) ** 0.01)
        print(f"Start prune, target_sparsity: {amount*100:.2f}%")
        for _ in range(100):
            self.global_unstructured(
                pruning_method=prune.L1Unstructured, amount=unit_amount
            )
        sparsity = self.mask_sparsity()
        print(f"Pruning Done, sparsity: {sparsity:.2f}%")

    def get_prune_score(self) -> List[float]:
        """Run prune algorithm and get score."""

        for data, target in self.dataloader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            self.criterion(output, target).backward()

        scores = []
        for (po, no) in self.params_to_prune_orig:
            score = getattr(po, no).grad.to("cpu").detach().abs_()
            scores.append(score)
            getattr(po, no).grad.data.zero_()
        return scores
