from pruning_method.pruner import Pruner
from typing import Dict, Any, List, Iterable
import torch.nn as nn
import torch
import torch.nn.utils.prune as prune

class Synflow(Pruner):
    def __init__(self, net: nn.Module, device: torch.device, input_shape: List[int]) -> None:
        super(Synflow, self).__init__(net, device, input_shape)

        self.params_to_prune = self.get_params(
            (
                (nn.Conv2d, "weight"),
                (nn.Conv2d, "bias"),
                (nn.Linear, "weight"),
                (nn.Linear, "bias")
            )
        )
        prune.global_unstructured(
            self.params_to_prune, pruning_method=prune.L1Unstructured, amount=0.0,
        )
        # https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
        # To get gradient of each weight(after prune at least one time)
        self.params_to_prune_orig = self.get_params(
            (
                (nn.Conv2d, "weight_orig"),
                (nn.Conv2d, "bias_orig"),
                (nn.Linear, "weight_orig"),
                (nn.Linear, "bias_orig")
            )
        )

        PRUNING_TYPE = 'unstructured'

    def prune(self, amount):
        unit_amount = 1- ((1-amount) ** 0.01)
        print(f"Start prune, target_sparsity: {amount*100:.2f}%")
        for _ in range(100):
            self.global_unstructured(pruning_method=prune.L1Unstructured, amount=unit_amount)
        sparsity = self.mask_sparsity()
        print(f"Pruning Done, sparsity: {sparsity:.2f}%")

    def get_prune_score(self) -> List[float]:
        """Run prune algorithm and get score."""
        # Synaptic flow
        signs = self.linearize()
        input_ones = torch.ones([1] + self.input_shape).to(self.device)
        self.model.eval()
        output = self.model(input_ones)
        torch.sum(output).backward()

        # get score function R
        scores = []
        cnt_zero = 0
        for (p, n), (po, no) in zip(self.params_to_prune, self.params_to_prune_orig):
            score = (getattr(p, n) * getattr(po, no).grad).to("cpu").detach().abs_()
            scores.append(score)
            getattr(po, no).grad.data.zero_()

        self.nonlinearize(signs)
        self.model.train()
        return scores

    @torch.no_grad()
    def linearize(self):
        signs = {}
        for name, param in self.model.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    @torch.no_grad()
    def nonlinearize(self, signs):
        for name, param in self.model.state_dict().items():
            param.mul_(signs[name])

