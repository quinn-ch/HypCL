"""Class order, incremental task splits, and the Trainer's dataset interface."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader

from utils.config import DATASETS
from utils.data import image_transform, load_dataset
from utils.toolkit import CLASS_ORDER_SEED

RESOURCES = Path(__file__).resolve().parent


class DataManager:
    def __init__(self, config, data_root, download=False):
        self.images, self.targets, self.use_paths = load_dataset(
            config.dataset, data_root, download
        )
        names = json.loads((RESOURCES / "labels.json").read_text())[config.dataset]
        self._data_to_prompt = json.loads((RESOURCES / "templates.json").read_text())[
            config.dataset
        ]
        np.random.seed(CLASS_ORDER_SEED)
        self._class_order = np.random.permutation(DATASETS[config.dataset][0]).tolist()
        self._class_to_label = [names[i] for i in self._class_order]
        mapping = np.argsort(self._class_order)
        for split in self.targets:
            self.targets[split] = mapping[self.targets[split]]
        self._increments = config.task_sizes
        self.transform = image_transform()

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_total_classnum(self):
        return len(self._class_order)

    def get_dataset(self, indices, source, mode):
        if source not in ("train", "test") or mode not in ("train", "test"):
            raise ValueError("source and mode must be 'train' or 'test'")
        # Group by incremental class, preserving within-class sample order.
        # A plain boolean mask would change minibatches under the same seed.
        selected = np.concatenate(
            [np.flatnonzero(self.targets[source] == i) for i in indices]
        )
        return DummyDataset(
            self.images[source][selected],
            self.targets[source][selected],
            self.transform,
            self.use_paths,
        )


class DummyDataset(Dataset):
    """Return (sample index, image, incremental class ID), as in the original code."""

    def __init__(self, images, labels, transform, use_paths):
        self.images, self.labels = images, labels
        self.transform, self.use_paths = transform, use_paths

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = (
            pil_loader(self.images[index])
            if self.use_paths
            else Image.fromarray(self.images[index])
        )
        return index, self.transform(image), int(self.labels[index])
