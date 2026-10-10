"""Task-specific banded updates for the visual encoder's Q/V projections."""
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F

from .geometry import (
    ADAPTER_CURVATURE,
    MAX_UPDATE_NORM,
    expmap0,
    logmap0,
    mobius_add,
    project,
)


class BandedAdapter(nn.Module):
    """Compose frozen task updates sequentially in the Poincare ball."""

    def __init__(self, width, bandwidth):
        super().__init__()
        self.width = width
        offsets = torch.cat(
            [torch.arange(0, bandwidth + 1), torch.arange(-1, -bandwidth - 1, -1)]
        )
        indices = (
            torch.sparse.spdiags(
                torch.ones(2 * bandwidth + 1, width), offsets, (width, width)
            )
            .coalesce()
            .indices()
        )
        self.register_buffer("indices", indices)
        self.task_weights = nn.ParameterList()
        self.updates_enabled = True

    def add_task(self):
        values = torch.zeros(self.indices.shape[1], device=self.indices.device)
        # Store the first task as I + delta, subsequent tasks as delta.
        # Retaining this convention preserves the paper's numerical behavior.
        if not self.task_weights:
            values[self.indices[0] == self.indices[1]] = 1.0
        self.task_weights.append(nn.Parameter(values))

    def freeze_task(self):
        self.task_weights[-1].requires_grad_(False)

    def forward(self, weight):
        with torch.autocast(device_type=weight.device.type, enabled=False):
            tangent = weight.float()
            point = expmap0(tangent, ADAPTER_CURVATURE, 1e-5)
            if self.updates_enabled:
                for task, values in enumerate(self.task_weights):
                    if task == 0:
                        identity = torch.zeros_like(values)
                        identity[self.indices[0] == self.indices[1]] = 1.0
                        values = values - identity
                    matrix = (
                        torch.sparse_coo_tensor(
                            self.indices, values, (self.width, self.width)
                        )
                        .to_dense()
                        .float()
                    )
                    update = tangent @ matrix.T
                    norm = update.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    update = update * (MAX_UPDATE_NORM / norm).clamp(max=1.0)
                    point = project(
                        mobius_add(
                            point,
                            expmap0(update, ADAPTER_CURVATURE, 1e-5),
                            ADAPTER_CURVATURE,
                        )
                    )
            # Base-space prototypes disable learned updates but retain this map.
            return logmap0(point).to(weight.dtype)


class AdaptedAttentionBlock(nn.Module):
    """A CLIP visual block whose query/value weights receive task updates."""

    def __init__(self, block, bandwidth):
        super().__init__()
        self.ln_1, self.attn = block.ln_1, block.attn
        self.ln_2, self.mlp = block.ln_2, block.mlp
        width = self.attn.embed_dim
        self.query_adapter = BandedAdapter(width, bandwidth)
        self.value_adapter = BandedAdapter(width, bandwidth)

    def forward(self, x, attn_mask=None):
        query, key, value = self.attn.in_proj_weight.chunk(3, dim=0)
        weight = torch.cat(
            [self.query_adapter(query), key, self.value_adapter(value)], dim=0
        )
        normalized = self.ln_1(x).transpose(0, 1)
        attended, _ = F.multi_head_attention_forward(
            query=normalized,
            key=normalized,
            value=normalized,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.attn.num_heads,
            in_proj_weight=weight,
            in_proj_bias=self.attn.in_proj_bias,
            bias_k=self.attn.bias_k,
            bias_v=self.attn.bias_v,
            add_zero_attn=False,
            dropout_p=self.attn.dropout if self.training else 0.0,
            out_proj_weight=self.attn.out_proj.weight,
            out_proj_bias=self.attn.out_proj.bias,
            training=self.training,
            key_padding_mask=None,
            need_weights=False,
            attn_mask=attn_mask,
            use_separate_proj_weight=False,
            average_attn_weights=False,
        )
        x = x + attended.transpose(0, 1).contiguous()
        return x + self.mlp(self.ln_2(x))


def install_adapters(visual, bandwidth):
    for i, block in enumerate(visual.transformer.resblocks):
        visual.transformer.resblocks[i] = AdaptedAttentionBlock(block, bandwidth)


def iter_adapters(model):
    return (module for module in model.modules() if isinstance(module, BandedAdapter))


@contextmanager
def base_features(model):
    """Temporarily omit learned task updates when extracting base prototypes."""
    adapters = list(iter_adapters(model))
    previous = [adapter.updates_enabled for adapter in adapters]
    try:
        for adapter in adapters:
            adapter.updates_enabled = False
        yield
    finally:
        for adapter, enabled in zip(adapters, previous):
            adapter.updates_enabled = enabled
