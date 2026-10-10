"""Visual prototypes and the incremental GDA calibration used by HypCL."""
import torch


def standardize(logits):
    centered = logits - logits.mean(dim=-1, keepdim=True)
    return centered / centered.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)


class PrototypeClassifier:
    """Keep class means and a shared precision estimate, without storing images."""

    def __init__(self):
        self.means = None
        self.precision = None
        self.weight = None
        self.bias = None

    @torch.no_grad()
    def update(self, features, labels, known_classes, total_classes):
        means = torch.cat(
            [
                features[labels == i].mean(dim=0, keepdim=True)
                for i in range(known_classes, total_classes)
            ]
        )
        centered = torch.cat(
            [
                features[labels == i] - means[i - known_classes]
                for i in range(known_classes, total_classes)
            ]
        )
        identity = torch.eye(
            centered.shape[1], device=centered.device, dtype=centered.dtype
        )
        covariance = centered.T.cov()
        precision = centered.shape[1] * torch.linalg.pinv(
            (centered.shape[0] - 1) * covariance + covariance.trace() * identity
        )
        if known_classes == 0:
            self.means, self.precision = means, precision
        else:
            old_ratio = known_classes / total_classes
            new_ratio = (total_classes - known_classes) / total_classes
            mean_delta = self.means.T.mean(dim=1).unsqueeze(1) - means.T.mean(
                dim=1
            ).unsqueeze(1)
            # Preserve the incremental precision update used by the paper.
            correction = (
                old_ratio * (total_classes - known_classes) / total_classes**2
            )
            self.precision = (
                old_ratio * self.precision
                + new_ratio * precision
                + correction * mean_delta @ mean_delta.T
            )
            self.means = torch.cat([self.means, means])
        prior = torch.ones(
            self.means.shape[0], device=features.device, dtype=features.dtype
        ) * (1.0 / self.means.shape[0])
        self.weight = torch.einsum("nd, dc -> cn", self.means, self.precision)
        self.bias = (
            prior.log()
            - torch.einsum("nd, dc, nc -> n", self.means, self.precision, self.means)
            / 2
        )

    def __call__(self, features):
        return features @ self.weight + self.bias
