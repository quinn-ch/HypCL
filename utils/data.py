"""Load the paper's prepared datasets and define the shared image transform."""
from pathlib import Path

import numpy as np
from torchvision import datasets, transforms

from utils.config import DATASETS


def load_dataset(dataset_name, data_root, download=False):
    root = Path(data_root)
    classes, directory = DATASETS[dataset_name]
    use_paths = dataset_name != "cifar100"
    images, targets, class_maps = {}, {}, []
    for split in ("train", "test"):
        if use_paths:
            folder = root / directory / split
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing prepared dataset split: {folder}")
            dataset = datasets.ImageFolder(folder)
            images[split] = np.array([str(path) for path, _ in dataset.samples])
            targets[split] = np.array(dataset.targets)
            class_maps.append(dataset.class_to_idx)
        else:
            dataset = datasets.CIFAR100(
                str(root), train=split == "train", download=download
            )
            images[split] = dataset.data
            targets[split] = np.array(dataset.targets)
        if len(np.unique(targets[split])) != classes:
            raise ValueError(
                f"Expected {classes} classes in {dataset_name}/{split}; "
                "use the paper's prepared subset"
            )
    if class_maps and class_maps[0] != class_maps[1]:
        raise ValueError("Train and test folders have different class mappings")
    return images, targets, use_paths


def image_transform():
    # Both splits use the same deterministic preprocessing in these experiments.
    return transforms.Compose(
        [
            transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
