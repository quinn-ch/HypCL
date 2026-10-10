"""The dataset-dependent hyperparameters used by the main experiments."""
from dataclasses import dataclass
import json
import math
from pathlib import Path


# Names match the label/prompt resources and the prepared dataset directories.
DATASETS = {
    "cifar100": (100, None),
    "imagenetr": (200, "imagenetr"),
    "cub": (200, "cub"),
    "cars": (100, "cars"),
    "aircraft": (100, "aircraft"),
    "ucf101": (100, "ucf"),
    "sun": (300, "sun"),
    "food101": (100, "food101"),
    "objectnet": (200, "objectnet"),
}


@dataclass(frozen=True)
class Experiment:
    dataset: str
    initial_classes: int
    increment: int
    epochs: int
    learning_rate: float
    bandwidth: int
    prototype_weight: float
    prototype_space: str

    def __post_init__(self):
        if self.dataset not in DATASETS:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        classes = DATASETS[self.dataset][0]
        for key in ("initial_classes", "increment", "epochs", "bandwidth"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if (
            self.initial_classes > classes
            or (classes - self.initial_classes) % self.increment
        ):
            raise ValueError("The class split must cover the dataset in complete tasks")
        if self.bandwidth >= 768:
            raise ValueError(
                "bandwidth must be smaller than the visual hidden dimension"
            )
        for key in ("learning_rate", "prototype_weight"):
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{key} must be a finite positive number")
        if self.prototype_weight >= 1:
            raise ValueError("prototype_weight must lie strictly between 0 and 1")
        if self.prototype_space not in ("base", "adapted"):
            raise ValueError("prototype_space must be 'base' or 'adapted'")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    @property
    def task_sizes(self):
        remaining = DATASETS[self.dataset][0] - self.initial_classes
        return [self.initial_classes] + [self.increment] * (remaining // self.increment)
