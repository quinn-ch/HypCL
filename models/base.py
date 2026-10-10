"""Class-incremental learner state and shared evaluation metrics."""
from abc import ABC, abstractmethod

import numpy as np
import torch


class BaseLearner(ABC):
    def __init__(self, config, device, workers):
        self.config = config
        self._device = torch.device(device)
        self.num_workers = workers
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self.topk = 5

    @abstractmethod
    def incremental_train(self, data_manager):
        """Learn the next task using only its training examples."""

    @abstractmethod
    def eval_task(self):
        """Evaluate the current model on all classes seen so far."""

    def after_task(self):
        self._known_classes = self._total_classes

    def _evaluate(self, predictions, targets):
        correct = predictions[:, 0] == targets
        top1 = float(np.around(100 * correct.sum() / len(targets), 2))
        topk = (predictions == targets[:, None]).sum()
        grouped = {"total": top1}
        for name, mask in (
            ("old", targets < self._known_classes),
            ("new", targets >= self._known_classes),
        ):
            grouped[name] = (
                float(np.around(100 * correct[mask].sum() / mask.sum(), 2))
                if mask.any()
                else None
            )
        return {
            "top1": top1,
            f"top{self.topk}": float(np.around(100 * topk / len(targets), 2)),
            "grouped": grouped,
        }
