import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import progressbar

class Trainer:
    def __init__(self, model, traindl, testdl, device) -> None:
        """Initialize trainer."""

        # setup loss, etc.
        self.model = model
        self.traindl = traindl
        self.testdl = testdl
        self.device = device

        # Train
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)


    def train(self, epochs: int) -> float:
        """Train model, return best acc."""
        max_testacc = 0.0
        for epoch in progressbar.progressbar(range(epochs)):
            n_correct = 0
            losses = []
            for inputs, labels in progressbar.progressbar(self.traindl):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                losses.append(loss.item())
                self.optimizer.step()
                n_correct = self._count_correct(outputs, labels, n_correct)
            print(f'[Train] Loss: {sum(losses)/len(losses):.3f} Acc: {100 * n_correct / len(self.traindl.dataset):.3f}%')

            max_testacc = max(max_testacc, self._test())
        return max_testacc

    @torch.no_grad()
    def _test(self) -> float:
        """Test model."""
        n_correct = 0
        losses = []
        for inputs, labels in progressbar.progressbar(self.testdl):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            losses.append(loss)
            n_correct = self._count_correct(outputs, labels, n_correct)
        acc = 100 * n_correct / len(self.testdl.dataset)
        print(f'[Test] Loss: {sum(losses)/len(losses):.3f} Acc: {acc:.3f}%')
        return acc

    def _count_correct(self, outputs, labels, n_correct) -> int:
        """Count correct and accumulate on n_correct."""
        _, preds = torch.max(F.softmax(outputs, dim=1).data, 1)
        return n_correct + int((preds == labels).sum().cpu())

