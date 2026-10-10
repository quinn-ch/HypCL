"""Frozen CLIP with hyperbolic Q/V adapters and visual-prototype calibration."""
from contextlib import nullcontext

import open_clip
import torch
from torch import nn

from models.adapter_banded import base_features, install_adapters, iter_adapters
from models.geometry import distance_logits
from models.prototypes import PrototypeClassifier, standardize

BACKBONE = "ViT-B-16"
PRETRAINED = "laion400m_e32"


class HypCLNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.clip, _, _ = open_clip.create_model_and_transforms(
            BACKBONE, pretrained=PRETRAINED
        )
        self.tokenizer = open_clip.get_tokenizer(BACKBONE)
        self.clip.requires_grad_(False)
        self.clip.eval()
        install_adapters(self.clip.visual, config.bandwidth)
        self.prototype_space = config.prototype_space
        self.prototype_weight = config.prototype_weight
        self.prototypes = PrototypeClassifier()
        self._text_cache = {}

    def add_task(self):
        for adapter in iter_adapters(self):
            adapter.add_task()

    def freeze_task(self):
        for adapter in iter_adapters(self):
            adapter.freeze_task()

    def current_parameters(self):
        return [adapter.task_weights[-1] for adapter in iter_adapters(self)]

    def encode_image(self, images):
        return self.clip.encode_image(images)

    @torch.no_grad()
    def text_features(self, names, templates):
        # Frozen text features can be cached. Keep train/eval entries separate:
        # PyTorch's attention kernels may differ between these execution modes.
        device = next(self.clip.parameters()).device
        key = (tuple(names), tuple(templates), self.training, str(device))
        if key not in self._text_cache:
            features = []
            for name in names:
                tokens = self.tokenizer(
                    [template.format(name) for template in templates]
                ).to(device)
                vectors = self.clip.encode_text(tokens)
                vectors = vectors / vectors.norm(dim=-1, keepdim=True)
                vector = vectors.mean(dim=0)
                features.append(vector / vector.norm())
            self._text_cache[key] = torch.stack(features)
        return self._text_cache[key]

    def visual_features(self, images):
        """The 768-D pooled visual feature before CLIP's projection."""
        visual = self.clip.visual
        x = visual.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        token = (
            visual.class_embedding.view(1, 1, -1).expand(x.shape[0], -1, -1).to(x.dtype)
        )
        x = torch.cat([token, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(visual.patch_dropout(x))
        x = visual.transformer(x)
        return visual.ln_post(x)[:, 0]

    def prototype_features(self, images):
        scope = base_features(self) if self.prototype_space == "base" else nullcontext()
        with scope:
            return self.visual_features(images)

    def forward(self, images, text_features):
        visual_text_logits = distance_logits(self.encode_image(images), text_features)
        features = self.prototype_features(images)
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        prototype_logits = self.prototypes(features)
        eta = self.prototype_weight
        return eta * standardize(prototype_logits) + (1 - eta) * standardize(
            visual_text_logits
        )
