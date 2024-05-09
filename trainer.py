import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

from scheduler import CosineAnnealingWarmupRestarts


def _count_correct(outputs: torch.Tensor, labels: torch.Tensor, n_correct: int) -> int:
    """Count correct and accumulate on n_correct."""
    _, preds = torch.max(F.softmax(outputs, dim=1).data, 1)
    return n_correct + int((preds == labels).sum().cpu())


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        traindl: torch.utils.data.dataloader.DataLoader,
        testdl: torch.utils.data.dataloader.DataLoader,
        device: torch.device,
        epoch: int,
    ) -> None:
        """Initialize trainer."""

        # setup loss, etc.
        self.model = model
        self.traindl = traindl
        self.testdl = testdl
        self.device = device

        # Train
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5
        )
        self.scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=epoch//4,
            cycle_mult=1.0,
            max_lr=0.1,
            min_lr=0.0001,
            warmup_steps=max(10, epoch//(4*4)),
            gamma=0.5,
        )

    def train(self, epochs: int) -> float:
        """Train model, return best acc."""
        max_testacc = 0.0
        self.model.train()
        for epoch in range(epochs):
            n_correct = 0
            losses = []
            with tqdm(self.traindl, unit="batch") as iepoch:
                for inputs, labels in iepoch:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    losses.append(loss.item())
                    self.optimizer.step()
                    self.scheduler.step()
                    n_correct = _count_correct(outputs, labels, n_correct)
                acc = 100 * n_correct / len(self.traindl.dataset)
                print(
                    f"{epoch}/{epochs} [Train] Loss: {sum(losses)/len(losses):.3f} Acc: {acc:.3f}%"
                )
            max_testacc = max(max_testacc, self.test())
        return max_testacc

    @torch.no_grad()
    def test(self) -> float:
        """Test model."""
        n_correct = 0
        losses = []
        self.model.eval()
        with tqdm(self.testdl, unit="batch") as iepoch:
            for inputs, labels in iepoch:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                losses.append(loss)
                n_correct = _count_correct(outputs, labels, n_correct)
        acc = 100 * n_correct / len(self.testdl.dataset)
        print(f"[Test] Loss: {sum(losses)/len(losses):.3f} Acc: {acc:.3f}%")
        return acc
