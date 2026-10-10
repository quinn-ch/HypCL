"""Construct the HypCL learner used by the main-experiment Trainer."""
from models.hypcl import Learner


def get_model(config, device, workers=16):
    return Learner(config, device, workers)
