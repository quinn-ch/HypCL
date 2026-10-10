"""Poincare-ball operations used by the adapters and distance classifier.

The adapter and classifier use different curvatures and numerical cutoffs,
matching the implementation used to obtain the paper's main results.
"""
import torch

ADAPTER_CURVATURE = 0.1
DISTANCE_CURVATURE = 1.0
DISTANCE_RADIUS = 1.472
# Approx. atanh(0.9): with curvature 1, unit features map to ball radius ~0.9.
DISTANCE_SCALE = 20.0
MAX_UPDATE_NORM = 2.0


class _Artanh(torch.autograd.Function):
    """Clipped inverse tanh with the adapter's original gradient convention."""

    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        return (torch.log(1 + x) - torch.log(1 - x)) * 0.5

    @staticmethod
    def backward(ctx, gradient):
        (x,) = ctx.saved_tensors
        return gradient / (1 - x**2)


def expmap0(vector, curvature, min_norm):
    norm = vector.norm(dim=-1, keepdim=True).clamp_min(min_norm)
    scaled = curvature**0.5 * norm
    # Compute the denominator separately to preserve the reference backward
    # graph's floating-point accumulation order.
    return scaled.clamp(max=15).tanh() * vector / (curvature**0.5 * norm)


def logmap0(point):
    sqrt_c = ADAPTER_CURVATURE**0.5
    norm = point.norm(dim=-1, keepdim=True).clamp_min(1e-5)
    return point / norm / sqrt_c * _Artanh.apply(sqrt_c * norm)


def mobius_add(x, y, curvature):
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    numerator = (1 + 2 * curvature * xy + curvature * y2) * x + (1 - curvature * x2) * y
    denominator = 1 + 2 * curvature * xy + curvature**2 * x2 * y2
    return numerator / denominator.clamp_min(1e-8)


def project(point):
    norm = point.norm(dim=-1, keepdim=True).clamp_min(1e-5)
    limit = (1 - 1e-3) / ADAPTER_CURVATURE**0.5
    return torch.where(norm > limit, point / norm * limit, point)


def distance_logits(images, texts):
    images = images / images.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    texts = texts / texts.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    image_ball = expmap0(DISTANCE_RADIUS * images, DISTANCE_CURVATURE, 1e-8)
    text_ball = expmap0(DISTANCE_RADIUS * texts, DISTANCE_CURVATURE, 1e-8)
    delta = mobius_add(
        -image_ball[:, None, :], text_ball[None, :, :], DISTANCE_CURVATURE
    )
    norm = delta.norm(dim=-1).clamp_min(1e-8)
    sqrt_c = DISTANCE_CURVATURE**0.5
    scaled = (sqrt_c * norm.clamp(max=(1 - 1e-5) / sqrt_c)).clamp(max=1 - 1e-5)
    distance = 2.0 * torch.atanh(scaled) / sqrt_c
    return -DISTANCE_SCALE * distance
