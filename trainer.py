"""Experiment setup and the original train → evaluate → after_task lifecycle."""
from dataclasses import asdict
import importlib.metadata
import logging
from pathlib import Path
import sys
import time

import torch

from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import (
    BATCH_SIZE,
    CLASS_ORDER_SEED,
    PYTHON_SEED,
    TORCH_SEED,
    set_seed,
)


def train(config, data_root, output, device="cuda:0", workers=16, download=False):
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA devices are supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select a GPU-enabled environment")
    if type(workers) is not int or workers < 0:
        raise ValueError("workers must be a non-negative integer")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] %(message)s",
        handlers=[
            logging.FileHandler(output / "train.log"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    try:
        logging.info("Config: %s", asdict(config))
        logging.info(
            "Environment: %s",
            {
                "python": sys.version,
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "torch", "torchvision", "open_clip_torch", "numpy", "timm"
                    )
                },
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None,
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "workers": workers,
                "batch_size": BATCH_SIZE,
                "class_order_seed": CLASS_ORDER_SEED,
                "torch_seed": TORCH_SEED,
                "python_seed": PYTHON_SEED,
                "data_root": str(Path(data_root).resolve()),
            },
        )
        logging.info("Training started")
        result = _train(config, data_root, device, workers, download)
    except BaseException:
        logging.exception("Training failed")
        raise
    logging.info(
        "Training completed: average=%.4f, final=%.2f, elapsed=%.1fs",
        result["average"],
        result["final"],
        result["elapsed_seconds"],
    )
    return result


def _train(config, data_root, device, workers, download):
    started = time.time()
    set_seed()
    data_manager = DataManager(config, data_root, download)
    logging.info("Class order: %s", data_manager._class_order)
    model = factory.get_model(config, device, workers)
    curve, stages = [], []
    for task in range(data_manager.nb_tasks):
        model.incremental_train(data_manager)
        metrics = model.eval_task()
        model.after_task()

        curve.append(metrics["top1"])
        stages.append(
            {"task": task + 1, "seen_classes": model._total_classes, **metrics}
        )
        result = {
            "complete": task + 1 == data_manager.nb_tasks,
            "curve": curve,
            "average": sum(curve) / len(curve),
            "final": curve[-1],
            "stages": stages,
            "elapsed_seconds": time.time() - started,
        }
        logging.info("Task %d metrics: %s", task + 1, metrics)
        logging.info(
            "Task %d complete: top1=%.2f, average=%.4f; curve=%s",
            task + 1,
            curve[-1],
            result["average"],
            curve,
        )
    return result
