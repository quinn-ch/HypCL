"""HypCL learner: train a task adapter, freeze it, and update visual prototypes."""
import logging

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models.base import BaseLearner
from models.geometry import distance_logits
from utils.inc_net import HypCLNet
from utils.toolkit import BATCH_SIZE, count_parameters


class Learner(BaseLearner):
    def __init__(self, config, device, workers=16):
        super().__init__(config, device, workers)
        self._network = HypCLNet(config).to(self._device)

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self.data_manager = data_manager
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        test_dataset = data_manager.get_dataset(
            np.arange(self._total_classes), source="test", mode="test"
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=self.num_workers,
        )
        self._network.add_task()
        logging.info(
            "Task %d/%d: classes %d-%d, trainable parameters %d / %d",
            self._cur_task + 1,
            data_manager.nb_tasks,
            self._known_classes,
            self._total_classes - 1,
            count_parameters(self._network, trainable=True),
            count_parameters(self._network),
        )
        self.train_model(self.train_loader, self.test_loader)
        self._network.freeze_task()
        self._update_prototypes()

    def train_model(self, train_loader, test_loader):
        optimizer = torch.optim.AdamW(
            self._network.current_parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs, eta_min=0
        )
        self.train_function(train_loader, test_loader, optimizer, scheduler)

    def train_function(self, train_loader, test_loader, optimizer, scheduler):
        names = self.data_manager._class_to_label[
            self._known_classes : self._total_classes
        ]
        templates = self.data_manager._data_to_prompt
        for epoch in range(self.config.epochs):
            self._network.train()
            texts = self._network.text_features(names, templates)
            loss_sum = 0.0
            for _, images, targets in train_loader:
                images, targets = images.to(self._device), targets.to(self._device)
                logits = distance_logits(self._network.encode_image(images), texts)
                loss = F.cross_entropy(logits, targets - self._known_classes)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
            scheduler.step()
            # A DataLoader iterator consumes RNG state. Preserve one evaluation
            # traversal per epoch so that subsequent training shuffles match.
            accuracy = self._compute_accuracy(test_loader)
            logging.info(
                "Task %d, epoch %d/%d: loss=%.4f, accuracy=%.2f",
                self._cur_task + 1,
                epoch + 1,
                self.config.epochs,
                loss_sum / len(train_loader),
                accuracy,
            )

    @torch.no_grad()
    def _update_prototypes(self):
        """Update GDA statistics from the current task, after adapter freezing."""
        self._network.eval()
        vectors, labels = [], []
        for _, images, targets in self.train_loader:
            features = self._network.prototype_features(images.to(self._device))
            vectors.append(features / features.norm(dim=-1, keepdim=True))
            labels.append(targets.to(self._device))
        self._network.prototypes.update(
            torch.cat(vectors),
            torch.cat(labels),
            self._known_classes,
            self._total_classes,
        )

    def _compute_accuracy(self, loader):
        predictions, targets = self._eval_cnn(loader, calibrated=False)
        return self._evaluate(predictions, targets)["top1"]

    def eval_task(self):
        predictions, targets = self._eval_cnn(self.test_loader, calibrated=True)
        return self._evaluate(predictions, targets)

    @torch.no_grad()
    def _eval_cnn(self, loader, *, calibrated):
        """Use distance logits during epochs and prototype fusion after training."""
        self._network.eval()
        texts = self._network.text_features(
            self.data_manager._class_to_label[: self._total_classes],
            self.data_manager._data_to_prompt,
        )
        predictions, targets = [], []
        for _, images, labels in loader:
            images = images.to(self._device)
            if calibrated:
                logits = self._network(images, texts)
            else:
                logits = distance_logits(self._network.encode_image(images), texts)
            predictions.append(logits.topk(self.topk, dim=1).indices.cpu().numpy())
            targets.append(labels.numpy())
        return np.concatenate(predictions), np.concatenate(targets)
