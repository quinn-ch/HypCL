"""Shared runtime constants, seed setup, and parameter counts."""
import random

import numpy as np
import torch

BATCH_SIZE = 128
TORCH_SEED = 1
PYTHON_SEED = 1993
CLASS_ORDER_SEED = 1993


def set_seed():
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(PYTHON_SEED)
    np.random.seed(CLASS_ORDER_SEED)


def count_parameters(model, trainable=False):
    return sum(
        p.numel() for p in model.parameters() if not trainable or p.requires_grad
    )
