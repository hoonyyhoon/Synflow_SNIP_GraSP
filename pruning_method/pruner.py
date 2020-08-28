from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Any, Tuple, List, Iterable
import torch.nn.utils.prune as prune

class Pruner(ABC):
    """Pruner abstract class."""
    def __init__(self, net: nn.Module, device: torch.device, input_shape: List[int]) -> None:
        """Initialize."""
        super(Pruner, self).__init__()
        self.model = net
        self.device = device
        self.input_shape = input_shape

    @abstractmethod
    def prune(self, amount):
        """Prune."""
        pass

    def global_unstructured(self, pruning_method, **kwargs):
        """ Based on
        https://pytorch.org/docs/stable/_modules/torch/nn/utils/prune.html#global_unstructured.
        Modify scores depending on the algorithm.
        """
        assert isinstance(self.params_to_prune, Iterable)

        scores = self.get_prune_score()

        t = torch.nn.utils.parameters_to_vector(scores)
        # similarly, flatten the masks (if they exist), or use a flattened vector
        # of 1s of the same dimensions as t
        default_mask = torch.nn.utils.parameters_to_vector(
            [
                getattr(module, name + "_mask", torch.ones_like(getattr(module, name)))
                for (module, name) in self.params_to_prune
            ]
        )

        # use the canonical pruning methods to compute the new mask, even if the
        # parameter is now a flattened out version of `parameters`
        container = prune.PruningContainer()
        container._tensor_name = "temp"  # to make it match that of `method`
        method = pruning_method(**kwargs)
        method._tensor_name = "temp"  # to make it match that of `container`
        if method.PRUNING_TYPE != "unstructured":
            raise TypeError(
                'Only "unstructured" PRUNING_TYPE supported for '
                "the `pruning_method`. Found method {} of type {}".format(
                    pruning_method, method.PRUNING_TYPE
                )
            )

        container.add_pruning_method(method)

        # use the `compute_mask` method from `PruningContainer` to combine the
        # mask computed by the new method with the pre-existing mask
        final_mask = container.compute_mask(t, default_mask)

        # Pointer for slicing the mask to match the shape of each parameter
        pointer = 0
        for module, name in self.params_to_prune:
            param = getattr(module, name)
            # The length of the parameter
            num_param = param.numel()
            # Slice the mask, reshape it
            param_mask = final_mask[pointer : pointer + num_param].view_as(param)
            # Assign the correct pre-computed mask to each parameter and add it
            # to the forward_pre_hooks like any other pruning method
            prune.custom_from_mask(module, name, param_mask)

            # Increment the pointer to continue slicing the final_mask
            pointer += num_param

    def get_params(self, extract_conditions: Tuple[Tuple[Any, str], ...]) -> Tuple[Tuple[nn.Module, str], ...]:
        """Get parameters(weight and bias) tuples for pruning."""
        t = []
        for module in self.model.modules():
            for module_type, param_name in extract_conditions:
                # it returns true when we try hasattr(even though it returns None)
                if (
                    isinstance(module, module_type)
                    and hasattr(module, param_name)
                    and getattr(module, param_name) is not None
                ):
                    t += [(module, param_name)]
        return tuple(t)

    def mask_sparsity(
        self,
        module_types: Tuple[Any, ...] = (
            nn.Conv2d,
            nn.Linear,
        ),
    ) -> float:
        """Get the ratio of zeros in weight masks."""
        n_zero = n_total = 0
        for module, param_name in self.params_to_prune:
            match = next((m for m in module_types if type(module) is m), None)
            if not match:
                continue
            param_mask_name = param_name + "_mask"
            if hasattr(module, param_mask_name):
                param = getattr(module, param_mask_name)
                n_zero += int(torch.sum(param == 0.0).item())
                n_total += param.nelement()

        return (100.0 * n_zero / n_total) if n_total != 0 else 0.0
