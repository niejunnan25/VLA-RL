from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from vla_rl.nn import HFResNetImageEncoder, MLP, SmallImageEncoder


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
        resnet_spatial_dropout_rate: float = 0.0,
        fuse_views: bool = True,
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
        self.fuse_views = bool(fuse_views)
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
                        spatial_dropout_rate=float(resnet_spatial_dropout_rate),
                    )
                    for key in self.image_keys
                }
            )
        else:
            raise ValueError(f"unknown image_encoder_type={self.image_encoder_type!r}")

        vector_input_dim = self.proprio_dim + self.action_dim * self.chunk_horizon + 1
        self.vector_proj = nn.Linear(vector_input_dim, self.vector_latent_dim)
        concat_dim = len(self.image_keys) * self.image_feature_dim + self.vector_latent_dim
        self.proj = MLP(concat_dim, [hidden_dim], hidden_dim, layer_norm=True, activation="tanh") if self.project_obs else None
        self.output_dim = int(hidden_dim if self.project_obs else concat_dim)

    def forward(self, obs: dict[str, Tensor]) -> Tensor:
        pieces = self._encode_images(obs)
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

    def _encode_images(self, obs: dict[str, Tensor]) -> list[Tensor]:
        if self._can_fuse_views(obs):
            return self._encode_images_fused(obs)
        pieces: list[Tensor] = []
        for key in self.image_keys:
            image = obs[f"image_{key}"]
            if image.ndim != 4:
                raise ValueError(f"image_{key} must be [B,C,H,W], got {tuple(image.shape)}")
            pieces.append(self.image_encoders[key](image.float()))
        return pieces

    def _can_fuse_views(self, obs: dict[str, Tensor]) -> bool:
        if not self.fuse_views or self.image_encoder_type not in {"hf_resnet", "resnet"}:
            return False
        if len(self.image_keys) < 2:
            return False
        first_encoder = self.image_encoders[self.image_keys[0]]
        if not isinstance(first_encoder, HFResNetImageEncoder):
            return False
        first_image = obs.get(f"image_{self.image_keys[0]}")
        if first_image is None or first_image.ndim != 4:
            return False
        for key in self.image_keys[1:]:
            encoder = self.image_encoders[key]
            image = obs.get(f"image_{key}")
            if not isinstance(encoder, HFResNetImageEncoder):
                return False
            if encoder.backbone is not first_encoder.backbone:
                return False
            if image is None or image.ndim != 4 or tuple(image.shape) != tuple(first_image.shape):
                return False
        return True

    def _encode_images_fused(self, obs: dict[str, Tensor]) -> list[Tensor]:
        first_encoder = self.image_encoders[self.image_keys[0]]
        assert isinstance(first_encoder, HFResNetImageEncoder)
        images = [obs[f"image_{key}"].float() for key in self.image_keys]
        batch_size = int(images[0].shape[0])
        fused_images = torch.cat(images, dim=0)
        fused_features = first_encoder.encode_backbone_bchw(fused_images)
        features_by_view = fused_features.split(batch_size, dim=0)
        pieces: list[Tensor] = []
        for key, features in zip(self.image_keys, features_by_view):
            encoder = self.image_encoders[key]
            assert isinstance(encoder, HFResNetImageEncoder)
            pieces.append(encoder.pool_features(features))
        return pieces


class GaussianResidualActor(nn.Module):
    def __init__(
        self,
        obs_encoder: PLDObsEncoder,
        residual_action_dim: int,
        hidden_dims: Sequence[int],
        log_std_min: float = -11.5,
        log_std_max: float = 0.0,
        activation: str = "relu",
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.obs_encoder = obs_encoder
        self.residual_action_dim = int(residual_action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.head = MLP(
            obs_encoder.output_dim,
            list(hidden_dims),
            2 * self.residual_action_dim,
            layer_norm=bool(layer_norm),
            activation=activation,
        )

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
        activation: str = "relu",
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.obs_encoder = obs_encoder
        self.q = MLP(
            obs_encoder.output_dim + int(final_action_dim),
            list(hidden_dims),
            1,
            layer_norm=bool(layer_norm),
            activation=activation,
        )

    def forward(self, obs: dict[str, Tensor], final_action: Tensor) -> Tensor:
        encoded = self.obs_encoder(obs)
        return self.q(torch.cat([encoded, final_action.reshape(final_action.shape[0], -1).float()], dim=-1))


def normal_log_prob_correction(action: Tensor) -> Tensor:
    return torch.log(torch.clamp(1.0 - action.pow(2), min=1e-6)).sum(dim=-1, keepdim=True)


def fanin_uniform(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            bound = 1.0 / math.sqrt(max(layer.weight.shape[1], 1))
            nn.init.uniform_(layer.weight, -bound, bound)
            nn.init.zeros_(layer.bias)
