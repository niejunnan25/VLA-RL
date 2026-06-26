from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int, layer_norm: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev, int(hidden_dim)))
            if layer_norm:
                layers.append(nn.LayerNorm(int(hidden_dim)))
            layers.append(nn.ReLU())
            prev = int(hidden_dim)
        layers.append(nn.Linear(prev, int(output_dim)))
        self.net = nn.Sequential(*layers)
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.normal_(last.weight, std=0.01)
            nn.init.zeros_(last.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SmallImageEncoder(nn.Module):
    def __init__(self, output_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, int(output_dim)),
            nn.ReLU(),
        )

    def forward(self, image: Tensor) -> Tensor:
        return self.net(image)


class SpatialLearnedEmbeddings(nn.Module):
    """Spatial learned pooling used by the SERL/DrQ ResNet encoder."""

    def __init__(self, height: int, width: int, channel: int, num_features: int = 8) -> None:
        super().__init__()
        self.kernel = nn.Parameter(torch.empty(height, width, channel, int(num_features)))
        nn.init.kaiming_normal_(self.kernel)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 4:
            raise ValueError(f"features must be BCHW, got {tuple(features.shape)}")
        x = features.permute(0, 2, 3, 1).contiguous()
        projected = torch.sum(x.unsqueeze(-1) * self.kernel.unsqueeze(0), dim=(1, 2))
        return projected.reshape(features.shape[0], -1)


class HFResNetImageEncoder(nn.Module):
    """HuggingFace ResNet image encoder shared by visual RL agents."""

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    def __init__(
        self,
        backbone: nn.Module,
        output_dim: int = 256,
        freeze_backbone: bool = True,
        pooling_method: str = "spatial_learned_embeddings",
        num_spatial_blocks: int = 8,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.output_dim = int(output_dim)
        self.freeze_backbone = bool(freeze_backbone)
        self.pooling_method = str(pooling_method)
        self.num_spatial_blocks = int(num_spatial_blocks)
        self.spatial_pool: SpatialLearnedEmbeddings | None = None
        self.bottleneck = nn.LazyLinear(self.output_dim)
        self.register_buffer("_mean", torch.tensor(self.imagenet_mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(self.imagenet_std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    @staticmethod
    def create_backbone(model_name: str = "microsoft/resnet-18", pretrained: bool = True) -> nn.Module:
        from transformers import ResNetConfig, ResNetModel

        resolved = _resolve_model_name(model_name)
        if pretrained:
            return ResNetModel.from_pretrained(resolved)
        return ResNetModel(ResNetConfig.from_pretrained(resolved))

    def train(self, mode: bool = True) -> "HFResNetImageEncoder":
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _normalize(self, image: Tensor) -> Tensor:
        x = _to_bchw(image)
        x = x.float()
        if not image.is_floating_point():
            x = x / 255.0
        return (x - self._mean) / self._std

    def _pool(self, features: Tensor) -> Tensor:
        if self.pooling_method == "spatial_learned_embeddings":
            if self.spatial_pool is None:
                _, c, h, w = features.shape
                self.spatial_pool = SpatialLearnedEmbeddings(h, w, c, self.num_spatial_blocks).to(features.device)
            return self.spatial_pool(features)
        if self.pooling_method == "avg":
            return torch.mean(features, dim=(-2, -1))
        if self.pooling_method == "max":
            return torch.amax(features, dim=(-2, -1))
        raise ValueError(f"unsupported ResNet pooling_method={self.pooling_method!r}")

    def forward(self, image: Tensor) -> Tensor:
        x = self._normalize(image)
        if self.freeze_backbone:
            with torch.no_grad():
                out = self.backbone(pixel_values=x, return_dict=True)
            features = out.last_hidden_state.detach()
        else:
            out = self.backbone(pixel_values=x, return_dict=True)
            features = out.last_hidden_state
        pooled = self._pool(features)
        projected = self.bottleneck(pooled)
        projected = torch.layer_norm(projected, projected.shape[-1:])
        return torch.tanh(projected)


def _to_bchw(image: Tensor) -> Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    return image.contiguous()


def _resolve_model_name(model_name: str) -> str:
    path = Path(model_name).expanduser()
    if path.exists():
        return str(path.resolve())
    return model_name
