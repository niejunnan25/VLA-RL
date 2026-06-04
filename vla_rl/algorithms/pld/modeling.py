from __future__ import annotations

import math
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


def _to_bchw(image: Tensor) -> Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    return image.contiguous()


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
    """HuggingFace ResNet encoder matching the original SERL PLD recipe.

    The original PLD task configs in serl_torch use `transformers.ResNetModel`
    with a frozen ImageNet backbone, spatial learned embeddings, and a 256-dim
    bottleneck. This intentionally does not use torchvision.
    """

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
        # PLDFeatureProcessor emits float images in [0, 1]. If callers provide
        # uint8 tensors directly, convert without adding a GPU synchronization.
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


def _resolve_model_name(model_name: str) -> str:
    path = Path(model_name).expanduser()
    if path.exists():
        return str(path.resolve())
    return model_name


class PLDObsEncoder(nn.Module):
    def __init__(
        self,
        image_keys: Sequence[str],
        proprio_dim: int,
        action_dim: int,
        chunk_horizon: int,
        image_encoder_type: str = "small",
        image_feature_dim: int = 64,
        vector_latent_dim: int = 64,
        hidden_dim: int = 256,
        project_obs: bool = True,
        resnet_model_name: str = "microsoft/resnet-18",
        resnet_pretrained: bool = True,
        freeze_image_backbone: bool = True,
        resnet_pooling_method: str = "spatial_learned_embeddings",
        resnet_num_spatial_blocks: int = 8,
    ) -> None:
        super().__init__()
        self.image_keys = tuple(str(key) for key in image_keys)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.chunk_horizon = int(chunk_horizon)
        self.image_encoder_type = str(image_encoder_type)
        self.image_feature_dim = int(image_feature_dim)
        self.vector_latent_dim = int(vector_latent_dim)
        self.project_obs = bool(project_obs)
        self.image_encoders = nn.ModuleDict()
        if self.image_encoder_type == "small":
            self.image_encoders.update({key: SmallImageEncoder(self.image_feature_dim) for key in self.image_keys})
        elif self.image_encoder_type in {"hf_resnet", "resnet"}:
            backbone = HFResNetImageEncoder.create_backbone(resnet_model_name, pretrained=bool(resnet_pretrained))
            self.image_encoders.update(
                {
                    key: HFResNetImageEncoder(
                        backbone=backbone,
                        output_dim=self.image_feature_dim,
                        freeze_backbone=bool(freeze_image_backbone),
                        pooling_method=resnet_pooling_method,
                        num_spatial_blocks=int(resnet_num_spatial_blocks),
                    )
                    for key in self.image_keys
                }
            )
        else:
            raise ValueError(f"unknown image_encoder_type={self.image_encoder_type!r}")

        vector_input_dim = self.proprio_dim + self.action_dim * self.chunk_horizon + 1
        self.vector_proj = nn.Linear(vector_input_dim, self.vector_latent_dim)
        concat_dim = len(self.image_keys) * self.image_feature_dim + self.vector_latent_dim
        self.proj = MLP(concat_dim, [hidden_dim], hidden_dim, layer_norm=True) if self.project_obs else None
        self.output_dim = int(hidden_dim if self.project_obs else concat_dim)

    def forward(self, obs: dict[str, Tensor]) -> Tensor:
        pieces: list[Tensor] = []
        for key in self.image_keys:
            image = obs[f"image_{key}"]
            if image.ndim != 4:
                raise ValueError(f"image_{key} must be [B,C,H,W], got {tuple(image.shape)}")
            pieces.append(self.image_encoders[key](image.float()))
        batch_size = pieces[0].shape[0] if pieces else obs["base_action_chunk"].shape[0]
        device = pieces[0].device if pieces else obs["base_action_chunk"].device
        proprio = obs.get("proprio")
        if proprio is None:
            proprio = torch.zeros(batch_size, self.proprio_dim, device=device)
        vector = torch.cat(
            [
                proprio.reshape(batch_size, -1).float(),
                obs["base_action_chunk"].reshape(batch_size, -1).float(),
                obs["alpha"].reshape(batch_size, -1).float(),
            ],
            dim=-1,
        )
        vector_features = self.vector_proj(vector)
        vector_features = torch.layer_norm(vector_features, vector_features.shape[-1:])
        pieces.append(torch.tanh(vector_features))
        encoded = torch.cat(pieces, dim=-1)
        return self.proj(encoded) if self.proj is not None else encoded


class GaussianResidualActor(nn.Module):
    def __init__(
        self,
        obs_encoder: PLDObsEncoder,
        residual_action_dim: int,
        hidden_dims: Sequence[int],
        log_std_min: float = -11.5,
        log_std_max: float = 0.0,
    ) -> None:
        super().__init__()
        self.obs_encoder = obs_encoder
        self.residual_action_dim = int(residual_action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.head = MLP(obs_encoder.output_dim, list(hidden_dims), 2 * self.residual_action_dim, layer_norm=True)

    def forward(self, obs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        out = self.head(self.obs_encoder(obs))
        mean, log_std = torch.chunk(out, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def deterministic(self, obs: dict[str, Tensor]) -> Tensor:
        mean, _ = self.forward(obs)
        return torch.tanh(mean)

    def sample(self, obs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        raw = normal.rsample()
        action = torch.tanh(raw)
        log_prob = normal.log_prob(raw) - torch.log(torch.clamp(1.0 - action.pow(2), min=1e-6))
        return action, log_prob.sum(dim=-1, keepdim=True)


class PLDCritic(nn.Module):
    def __init__(
        self,
        obs_encoder: PLDObsEncoder,
        final_action_dim: int,
        hidden_dims: Sequence[int],
    ) -> None:
        super().__init__()
        self.obs_encoder = obs_encoder
        self.q = MLP(obs_encoder.output_dim + int(final_action_dim), list(hidden_dims), 1, layer_norm=True)

    def forward(self, obs: dict[str, Tensor], final_action: Tensor) -> Tensor:
        encoded = self.obs_encoder(obs)
        return self.q(torch.cat([encoded, final_action.reshape(final_action.shape[0], -1).float()], dim=-1))


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - float(tau)).add_(param.data, alpha=float(tau))


def normal_log_prob_correction(action: Tensor) -> Tensor:
    return torch.log(torch.clamp(1.0 - action.pow(2), min=1e-6)).sum(dim=-1, keepdim=True)


def fanin_uniform(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            bound = 1.0 / math.sqrt(max(layer.weight.shape[1], 1))
            nn.init.uniform_(layer.weight, -bound, bound)
            nn.init.zeros_(layer.bias)
