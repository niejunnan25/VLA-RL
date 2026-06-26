from __future__ import annotations

import copy
from contextlib import nullcontext
import threading
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vla_rl.algorithms.base import Algorithm
from vla_rl.nn import HFResNetImageEncoder, MLP, SmallImageEncoder, soft_update
from vla_rl.data import RolloutBatch


_COMPILE_KWARGS = {
    "backend": "inductor",
    "mode": "default",
    "fullgraph": True,
    "dynamic": False,
}


class ObsEncoder(nn.Module):
    def __init__(
        self,
        image_keys: Sequence[str] = ("image_rgb_0", "image_rgb_1"),
        proprio_dim: int = 8,
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

        self.vector_proj = nn.Linear(self.proprio_dim, self.vector_latent_dim)
        concat_dim = len(self.image_keys) * self.image_feature_dim + self.vector_latent_dim
        self.proj = MLP(concat_dim, [hidden_dim], hidden_dim, layer_norm=True) if self.project_obs else None
        self.output_dim = int(hidden_dim if self.project_obs else concat_dim)
        self._fuse_resnet_views = self._can_fuse_resnet_views()

    def forward(self, obs: dict[str, torch.Tensor], *, stop_gradient: bool = False) -> torch.Tensor:
        pieces = self._encode_image_features(obs)
        if pieces:
            batch_size = pieces[0].shape[0]
            device = pieces[0].device
        else:
            proprio = obs["proprio"]
            batch_size = proprio.shape[0]
            device = proprio.device
        proprio = obs.get("proprio")
        if proprio is None:
            proprio = torch.zeros(batch_size, self.proprio_dim, device=device)
        vector_features = self.vector_proj(proprio.reshape(batch_size, -1).float())
        vector_features = torch.layer_norm(vector_features, vector_features.shape[-1:])
        pieces.append(torch.tanh(vector_features))
        encoded = torch.cat(pieces, dim=-1)
        encoded = self.proj(encoded) if self.proj is not None else encoded
        return encoded.detach() if stop_gradient else encoded

    def _can_fuse_resnet_views(self) -> bool:
        if self.image_encoder_type not in {"hf_resnet", "resnet"} or len(self.image_keys) <= 1:
            return False
        encoders = [self.image_encoders[key] for key in self.image_keys]
        if not all(isinstance(encoder, HFResNetImageEncoder) for encoder in encoders):
            return False
        return len({id(encoder.backbone) for encoder in encoders}) == 1

    def _encode_image_features(self, obs: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        if self._fuse_resnet_views:
            fused = self._encode_image_features_fused(obs)
            if fused is not None:
                return fused
        return self._encode_image_features_loop(obs)

    def _encode_image_features_loop(self, obs: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        pieces: list[torch.Tensor] = []
        for key in self.image_keys:
            image = obs[f"image_{key}"]
            if image.ndim != 4:
                raise ValueError(f"image_{key} must be [B,C,H,W], got {tuple(image.shape)}")
            pieces.append(self.image_encoders[key](image.float()))
        return pieces

    def _encode_image_features_fused(self, obs: dict[str, torch.Tensor]) -> list[torch.Tensor] | None:
        images = []
        for key in self.image_keys:
            image = obs[f"image_{key}"]
            if image.ndim != 4:
                raise ValueError(f"image_{key} must be [B,C,H,W], got {tuple(image.shape)}")
            images.append(image.float())
        first = images[0]
        if any(image.device != first.device or image.dtype != first.dtype or tuple(image.shape) != tuple(first.shape) for image in images[1:]):
            return None
        batch_size = int(first.shape[0])
        encoders = [self.image_encoders[key] for key in self.image_keys]
        fused_features = encoders[0].encode_backbone_bchw(torch.cat(images, dim=0))
        return [
            encoder.pool_features(features)
            for encoder, features in zip(encoders, fused_features.split(batch_size, dim=0))
        ]


class GaussianActor(nn.Module):
    def __init__(
        self,
        obs_encoder: ObsEncoder,
        action_dim: int,
        hidden_dims: Sequence[int],
        log_std_min: float = -11.5,
        log_std_max: float = 0.0,
    ) -> None:
        super().__init__()
        self.obs_encoder = obs_encoder
        self.action_dim = int(action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.head = MLP(obs_encoder.output_dim, list(hidden_dims), 2 * self.action_dim, layer_norm=True)

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        *,
        stop_encoder_gradient: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.head(self.obs_encoder(obs, stop_gradient=stop_encoder_gradient))
        mean, log_std = torch.chunk(out, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def deterministic(self, obs: dict[str, torch.Tensor], *, stop_encoder_gradient: bool = False) -> torch.Tensor:
        mean, _ = self.forward(obs, stop_encoder_gradient=stop_encoder_gradient)
        return torch.tanh(mean)

    def sample(
        self,
        obs: dict[str, torch.Tensor],
        *,
        stop_encoder_gradient: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs, stop_encoder_gradient=stop_encoder_gradient)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        raw = normal.rsample()
        action = torch.tanh(raw)
        log_prob = normal.log_prob(raw) - torch.log(torch.clamp(1.0 - action.pow(2), min=1e-6))
        return action, log_prob.sum(dim=-1, keepdim=True)


class Critic(nn.Module):
    def __init__(self, obs_encoder: ObsEncoder, action_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        self.obs_encoder = obs_encoder
        self.q = MLP(obs_encoder.output_dim + int(action_dim), list(hidden_dims), 1, layer_norm=True)

    def forward_encoded(self, encoded: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([encoded, action.reshape(action.shape[0], -1).float()], dim=-1))

    def forward(self, obs: dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        return self.forward_encoded(self.obs_encoder(obs), action)


class CriticEnsemble(nn.Module):
    def __init__(self, critics: torch.nn.ModuleList) -> None:
        super().__init__()
        self.critics = critics

    def forward(self, obs: dict[str, torch.Tensor], action: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if len(self.critics) > 1 and all(critic.obs_encoder is self.critics[0].obs_encoder for critic in self.critics):
            encoded = self.critics[0].obs_encoder(obs)
            return tuple(critic.forward_encoded(encoded, action) for critic in self.critics)
        return tuple(critic(obs, action) for critic in self.critics)


class SACAgent(Algorithm):
    """Torch SAC agent used for standard single-step RLPD."""

    def __init__(
        self,
        image_keys: Sequence[str] = ("image_rgb_0", "image_rgb_1"),
        proprio_dim: int = 8,
        action_dim: int = 7,
        action_limits: Sequence[float] | None = None,
        clip_gripper: bool = True,
        image_encoder_type: str = "small",
        image_feature_dim: int = 64,
        vector_latent_dim: int = 64,
        encoder_hidden_dim: int = 256,
        project_obs: bool = True,
        shared_encoder: bool = True,
        drq_random_crop: bool = True,
        drq_padding: int = 4,
        resnet_model_name: str = "microsoft/resnet-18",
        resnet_pretrained: bool = True,
        freeze_image_backbone: bool = True,
        resnet_pooling_method: str = "spatial_learned_embeddings",
        resnet_num_spatial_blocks: int = 8,
        actor_hidden_dims: Sequence[int] = (256, 256, 256),
        critic_hidden_dims: Sequence[int] = (256, 256, 256),
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temperature_lr: float = 3e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        init_temperature: float = 1.0,
        target_entropy: float | None = None,
        backup_entropy: bool = False,
        utd_ratio: int = 1,
        critic_actor_ratio: int = 2,
        clip_grad_norm: float = 1.0,
        device: str = "cpu",
        enable_compile: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.enable_compile = bool(enable_compile)
        self._network_lock = threading.RLock()
        self.image_keys = tuple(str(key) for key in image_keys)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.clip_gripper = bool(clip_gripper)
        self.shared_encoder = bool(shared_encoder)
        self.drq_random_crop = bool(drq_random_crop)
        self.drq_padding = int(drq_padding)
        limits = np.ones((self.action_dim,), dtype=np.float32) if action_limits is None else np.asarray(list(action_limits), dtype=np.float32)
        if limits.shape != (self.action_dim,):
            raise ValueError(f"action_limits shape {limits.shape} does not match action_dim={self.action_dim}")
        if np.any(~np.isfinite(limits)) or np.any(limits <= 0.0):
            raise ValueError(f"action_limits must be finite and positive, got {limits.tolist()}")
        self.action_limits_np = limits.astype(np.float32, copy=False)
        self.registered_action_limits = torch.as_tensor(self.action_limits_np, dtype=torch.float32, device=self.device)
        self.actor_lr = float(actor_lr)
        self.critic_lr = float(critic_lr)
        self.temperature_lr = float(temperature_lr)
        self.discount = float(discount)
        self.tau = float(tau)
        self.backup_entropy = bool(backup_entropy)
        self.utd_ratio = int(utd_ratio)
        self.critic_actor_ratio = int(critic_actor_ratio)
        self.clip_grad_norm = float(clip_grad_norm)
        self.update_count = 0
        self._critic_step_count = 0

        encoder_kwargs = dict(
            image_keys=self.image_keys,
            proprio_dim=self.proprio_dim,
            image_encoder_type=image_encoder_type,
            image_feature_dim=image_feature_dim,
            vector_latent_dim=vector_latent_dim,
            hidden_dim=encoder_hidden_dim,
            project_obs=project_obs,
            resnet_model_name=resnet_model_name,
            resnet_pretrained=resnet_pretrained,
            freeze_image_backbone=freeze_image_backbone,
            resnet_pooling_method=resnet_pooling_method,
            resnet_num_spatial_blocks=resnet_num_spatial_blocks,
        )
        actor_encoder = ObsEncoder(**encoder_kwargs)
        critic_encoder = actor_encoder if self.shared_encoder else ObsEncoder(**encoder_kwargs)
        self.actor = GaussianActor(actor_encoder, self.action_dim, list(actor_hidden_dims)).to(self.device)
        self.critics = torch.nn.ModuleList(
            [
                Critic(
                    critic_encoder,
                    action_dim=self.action_dim,
                    hidden_dims=list(critic_hidden_dims),
                )
                for _ in range(2)
            ]
        ).to(self.device)
        self.critic_targets = copy.deepcopy(self.critics).to(self.device)
        actor_params = self.actor.head.parameters() if self.shared_encoder else self.actor.parameters()
        self.actor_optimizer = torch.optim.Adam(self._unique_parameters(actor_params), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self._unique_parameters(self.critics.parameters()), lr=self.critic_lr)
        init_log_temp = float(np.log(max(float(init_temperature), 1e-6)))
        self.log_temperature = torch.nn.Parameter(torch.tensor(init_log_temp, dtype=torch.float32, device=self.device))
        self.temperature_optimizer = torch.optim.Adam([self.log_temperature], lr=self.temperature_lr)
        self.target_entropy = float(target_entropy) if target_entropy is not None else -float(self.action_dim)
        self._compiled_actor: nn.Module | None = None
        self._compiled_critics: nn.Module | None = None
        self._compiled_critic_targets: nn.Module | None = None

    @torch.no_grad()
    def sample_action(self, obs: dict[str, Any], deterministic: bool = False) -> np.ndarray:
        with self._network_lock:
            batch = self._obs_to_torch([obs])
            with self._autocast_context():
                normalized = self._actor_deterministic(batch) if deterministic else self._actor_sample(batch)[0]
            action = self._scale_action_torch(normalized)
            return action.detach().cpu().numpy().reshape(1, self.action_dim).astype(np.float32)

    def set_compile_enabled(self, enabled: bool) -> None:
        self.enable_compile = bool(enabled)
        if not self.enable_compile:
            self._compiled_actor = None
            self._compiled_critics = None
            self._compiled_critic_targets = None

    def random_action(self) -> np.ndarray:
        action = np.random.uniform(-1.0, 1.0, size=(1, self.action_dim)).astype(np.float32)
        action = action * self.action_limits_np.reshape(1, -1)
        if self.clip_gripper and action.shape[-1] > 0:
            action[:, -1] = np.clip(action[:, -1], -1.0, 1.0)
        return action.astype(np.float32, copy=False)

    def update(self, batch: RolloutBatch) -> dict:
        batch.validate()
        fb = self._convert_batch(batch)
        drq_enabled = bool(self.drq_random_crop and self.drq_padding > 0)
        info: dict[str, float] = {
            "drq_random_crop": float(drq_enabled),
            "shared_encoder": float(self.shared_encoder),
        }
        critic_updates = 0
        drq_augmentations = 0
        critic_only_steps = max(0, int(self.critic_actor_ratio) - 1)
        for _ in range(critic_only_steps):
            critic_fb = self._apply_drq_augmentation(fb)
            drq_augmentations += int(drq_enabled)
            info.update(self._critic_step(critic_fb))
            self._critic_step_count += 1
            critic_updates += 1
            self._update_targets()

        high_utd_fb = self._apply_drq_augmentation(fb)
        drq_augmentations += int(drq_enabled)
        utd_ratio = max(1, int(self.utd_ratio))
        critic_batches = self._split_batch(high_utd_fb, utd_ratio)
        critic_infos: list[dict[str, float]] = []
        for idx in range(utd_ratio):
            minibatch = high_utd_fb if utd_ratio == 1 else self._index_batch(critic_batches, idx)
            critic_info = self._critic_step(minibatch)
            critic_infos.append(critic_info)
            self._critic_step_count += 1
            critic_updates += 1
            self._update_targets()
        if critic_infos:
            info.update(self._mean_infos(critic_infos))

        info.update(self._actor_step(high_utd_fb))
        info.update(self._temperature_step(info.get("actor_log_prob_mean", 0.0)))
        self.update_count += 1
        info["updates"] = float(self.update_count)
        info["critic_updates"] = float(critic_updates)
        info["actor_updates"] = 1.0
        info["drq_augmentations"] = float(drq_augmentations)
        return info

    def _critic_step(self, fb: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> dict[str, float]:
        obs = fb["obs"]
        next_obs = fb["next_obs"]
        action = fb["action"]
        reward = fb["reward"]
        done = fb["done"]
        discount = fb["discount"]
        with torch.no_grad(), self._autocast_context():
            next_norm, next_log_prob = self._actor_sample(next_obs)
            next_action = self._scale_action_torch(next_norm)
            target_qs = self._critic_values(self.critic_targets, next_obs, next_action)
            min_target_q = torch.min(torch.cat(target_qs, dim=-1), dim=-1, keepdim=True).values
            if self.backup_entropy:
                min_target_q = min_target_q - self.temperature.detach() * next_log_prob
            target = reward + (1.0 - done) * discount * min_target_q

        with self._autocast_context():
            q_preds = self._critic_values(self.critics, obs, action)
            loss = sum(F.mse_loss(q, target) for q in q_preds)
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critics.parameters(), self.clip_grad_norm)
        self.critic_optimizer.step()
        return {
            "loss_critic": loss.detach(),
            "target_q_mean": target.mean().detach(),
            "predicted_q_mean": torch.stack([q.mean() for q in q_preds]).mean().detach(),
        }

    def _actor_step(self, fb: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> dict[str, float]:
        obs = fb["obs"]
        critic_params = self._unique_parameters(self.critics.parameters())
        previous_requires_grad = [param.requires_grad for param in critic_params]
        for param in critic_params:
            param.requires_grad_(False)
        try:
            with self._autocast_context():
                normalized, log_prob = self._actor_sample(obs, stop_encoder_gradient=self.shared_encoder)
                action = self._scale_action_torch(normalized)
                q_values = self._critic_values(self.critics, obs, action)
                min_q = torch.min(torch.cat(q_values, dim=-1), dim=-1, keepdim=True).values
                loss = (self.temperature.detach() * log_prob - min_q).mean()
            self.actor_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._unique_parameters(self.actor_optimizer.param_groups[0]["params"]), self.clip_grad_norm)
            self.actor_optimizer.step()
        finally:
            for param, requires_grad in zip(critic_params, previous_requires_grad):
                param.requires_grad_(requires_grad)
        return {
            "loss_actor": loss.detach(),
            "actor_q_mean": min_q.mean().detach(),
            "actor_log_prob_mean": log_prob.mean().detach(),
        }

    def _temperature_step(self, log_prob_mean: float | torch.Tensor) -> dict[str, torch.Tensor]:
        if isinstance(log_prob_mean, torch.Tensor):
            log_prob = log_prob_mean.detach().to(device=self.device, dtype=torch.float32)
        else:
            log_prob = torch.tensor(float(log_prob_mean), device=self.device)
        loss = -(self.log_temperature * (log_prob + self.target_entropy).detach())
        self.temperature_optimizer.zero_grad()
        loss.backward()
        self.temperature_optimizer.step()
        return {"temperature": self.temperature.detach(), "loss_temperature": loss.detach()}

    def _apply_drq_augmentation(
        self,
        fb: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if not self.drq_random_crop or self.drq_padding <= 0:
            return fb
        augmented = dict(fb)
        augmented["obs"] = self._augment_observations(fb["obs"])
        augmented["next_obs"] = self._augment_observations(fb["next_obs"])
        return augmented

    def _augment_observations(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        result = dict(obs)
        for key in self.image_keys:
            image_key = f"image_{key}"
            result[image_key] = self._batched_random_crop(result[image_key], padding=self.drq_padding)
        return result

    @staticmethod
    def _batched_random_crop(image: torch.Tensor, *, padding: int) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"DrQ crop expects BCHW image tensors, got {tuple(image.shape)}")
        if padding <= 0:
            return image
        batch_size, channels, height, width = image.shape
        padded = F.pad(image, (padding, padding, padding, padding), mode="replicate")
        padded = padded.permute(0, 2, 3, 1).contiguous()
        top = torch.randint(0, 2 * padding + 1, (batch_size,), device=image.device)
        left = torch.randint(0, 2 * padding + 1, (batch_size,), device=image.device)
        rows = top.view(batch_size, 1, 1) + torch.arange(height, device=image.device).view(1, height, 1)
        cols = left.view(batch_size, 1, 1) + torch.arange(width, device=image.device).view(1, 1, width)
        rows = rows.expand(batch_size, height, width)
        cols = cols.expand(batch_size, height, width)
        batch = torch.arange(batch_size, device=image.device).view(batch_size, 1, 1).expand(batch_size, height, width)
        cropped = padded[batch, rows, cols]
        return cropped.permute(0, 3, 1, 2).contiguous().view(batch_size, channels, height, width)

    def _critic_values(
        self,
        critics: torch.nn.ModuleList,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor,
    ) -> list[torch.Tensor]:
        if critics is self.critics:
            compiled = self._ensure_critics_compiled(target=False)
            if compiled is not None:
                return list(compiled(obs, action))
        if critics is self.critic_targets:
            compiled = self._ensure_critics_compiled(target=True)
            if compiled is not None:
                return list(compiled(obs, action))
        return list(CriticEnsemble(critics)(obs, action))

    def _actor_forward(
        self,
        obs: dict[str, torch.Tensor],
        *,
        stop_encoder_gradient: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actor = self._ensure_actor_compiled() or self.actor
        return actor(obs, stop_encoder_gradient=stop_encoder_gradient)

    def _actor_deterministic(self, obs: dict[str, torch.Tensor], *, stop_encoder_gradient: bool = False) -> torch.Tensor:
        mean, _ = self._actor_forward(obs, stop_encoder_gradient=stop_encoder_gradient)
        return torch.tanh(mean)

    def _actor_sample(
        self,
        obs: dict[str, torch.Tensor],
        *,
        stop_encoder_gradient: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self._actor_forward(obs, stop_encoder_gradient=stop_encoder_gradient)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        raw = normal.rsample()
        action = torch.tanh(raw)
        log_prob = normal.log_prob(raw) - torch.log(torch.clamp(1.0 - action.pow(2), min=1e-6))
        return action, log_prob.sum(dim=-1, keepdim=True)

    def _autocast_context(self):
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    def _compile_available(self) -> bool:
        return bool(self.enable_compile) and self.device.type == "cuda" and torch.cuda.is_available() and hasattr(torch, "compile")

    def _ensure_actor_compiled(self) -> nn.Module | None:
        if self._compiled_actor is not None:
            return self._compiled_actor
        if not self._compile_available():
            return None
        with torch.no_grad():
            self.actor.deterministic(self._dummy_obs_torch())
        self._compiled_actor = torch.compile(self.actor, **_COMPILE_KWARGS)
        return self._compiled_actor

    def _ensure_critics_compiled(self, *, target: bool) -> nn.Module | None:
        if not self._compile_available():
            return None
        if target:
            if self._compiled_critic_targets is not None:
                return self._compiled_critic_targets
            critics = self.critic_targets
        else:
            if self._compiled_critics is not None:
                return self._compiled_critics
            critics = self.critics
        dummy = self._dummy_obs_torch()
        action = torch.zeros((1, self.action_dim), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            CriticEnsemble(critics).to(self.device)(dummy, action)
        compiled = torch.compile(CriticEnsemble(critics).to(self.device), **_COMPILE_KWARGS)
        if target:
            self._compiled_critic_targets = compiled
        else:
            self._compiled_critics = compiled
        return compiled

    def _split_batch(
        self,
        fb: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        parts: int,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        parts = int(parts)
        if parts <= 1:
            return fb
        return {key: self._split_value(value, parts) for key, value in fb.items()}

    def _index_batch(
        self,
        fb: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        index: int,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        return {key: self._index_value(value, index) for key, value in fb.items()}

    def _split_value(self, value: torch.Tensor | dict[str, torch.Tensor], parts: int):
        if isinstance(value, dict):
            return {key: self._split_value(child, parts) for key, child in value.items()}
        batch_size = int(value.shape[0])
        if batch_size % int(parts) != 0:
            raise ValueError(f"batch size {batch_size} must be divisible by utd_ratio={parts}")
        mini_batch = batch_size // int(parts)
        return value.reshape(int(parts), mini_batch, *value.shape[1:])

    def _index_value(self, value: torch.Tensor | dict[str, torch.Tensor], index: int):
        if isinstance(value, dict):
            return {key: self._index_value(child, index) for key, child in value.items()}
        return value[int(index)]

    def _mean_infos(self, infos: list[dict[str, Any]]) -> dict[str, Any]:
        if not infos:
            return {}
        keys = set().union(*(info.keys() for info in infos))
        averaged: dict[str, Any] = {}
        for key in keys:
            values = [info[key] for info in infos if key in info]
            if not values:
                continue
            if any(isinstance(value, torch.Tensor) for value in values):
                tensors = [
                    value if isinstance(value, torch.Tensor) else torch.tensor(float(value), device=self.device)
                    for value in values
                ]
                averaged[key] = sum(tensors) / len(tensors)
            else:
                averaged[key] = float(sum(float(value) for value in values) / len(values))
        return averaged

    def _convert_batch(self, batch: RolloutBatch) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        transitions = batch.transitions
        obs = self._obs_to_torch([t.obs for t in transitions])
        next_obs = self._obs_to_torch([(t.next_obs if t.next_obs is not None else t.obs) for t in transitions])
        actions = np.stack([np.asarray(t.action, dtype=np.float32).reshape(-1) for t in transitions])
        return {
            "obs": obs,
            "next_obs": next_obs,
            "action": torch.as_tensor(actions, dtype=torch.float32, device=self.device),
            "reward": torch.tensor([[t.reward] for t in transitions], dtype=torch.float32, device=self.device),
            "done": torch.tensor(
                [[bool(t.info.get("critic_terminal", t.done or t.truncated))] for t in transitions],
                dtype=torch.float32,
                device=self.device,
            ),
            "discount": torch.tensor([[t.discount] for t in transitions], dtype=torch.float32, device=self.device),
        }

    def _obs_to_torch(self, obs_list: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        required_keys = [*(f"image_{key}" for key in self.image_keys), "proprio"]
        for key in required_keys:
            values = [np.asarray(obs[key], dtype=np.float32) for obs in obs_list]
            result[key] = torch.as_tensor(np.stack(values), dtype=torch.float32, device=self.device)
        return result

    def _scale_action_torch(self, normalized_action: torch.Tensor) -> torch.Tensor:
        limits = self.registered_action_limits.to(device=normalized_action.device, dtype=normalized_action.dtype)
        action = torch.clamp(normalized_action, -1.0, 1.0) * limits.view(1, -1)
        if self.clip_gripper and action.shape[-1] > 0:
            action = torch.cat([action[..., :-1], torch.clamp(action[..., -1:], -1.0, 1.0)], dim=-1)
        return action

    @staticmethod
    def _unique_parameters(parameters) -> list[torch.nn.Parameter]:
        unique: list[torch.nn.Parameter] = []
        seen: set[int] = set()
        for param in parameters:
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            unique.append(param)
        return unique

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def _update_targets(self) -> None:
        soft_update(self.critics, self.critic_targets, self.tau)

    def state_dict(self) -> dict:
        self._ensure_networks_initialized()
        return {
            "actor": self.actor.state_dict(),
            "critics": self.critics.state_dict(),
            "critic_targets": self.critic_targets.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "temperature_optimizer": self.temperature_optimizer.state_dict(),
            "log_temperature": self.log_temperature.detach().cpu(),
            "update_count": self.update_count,
            "critic_step_count": self._critic_step_count,
        }

    def load_state_dict(self, state: dict) -> None:
        self._ensure_networks_initialized()
        self.actor.load_state_dict(state["actor"])
        self.critics.load_state_dict(state["critics"])
        self.critic_targets.load_state_dict(state["critic_targets"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        if "temperature_optimizer" in state:
            self.temperature_optimizer.load_state_dict(state["temperature_optimizer"])
        if "log_temperature" in state:
            self.log_temperature.data.copy_(torch.as_tensor(state["log_temperature"], device=self.device))
        self.update_count = int(state.get("update_count", 0))
        self._critic_step_count = int(state.get("critic_step_count", 0))
        self._move_optimizer_state_to_device(self.actor_optimizer)
        self._move_optimizer_state_to_device(self.critic_optimizer)
        self._move_optimizer_state_to_device(self.temperature_optimizer)

    def policy_state_dict(self) -> dict:
        with self._network_lock:
            self._ensure_networks_initialized(actor_only=True)
            actor_state = {key: value.detach().cpu().clone() for key, value in self.actor.state_dict().items()}
            return {"actor": actor_state, "update_count": self.update_count}

    def load_policy_state_dict(self, state: dict) -> None:
        with self._network_lock:
            self._ensure_networks_initialized(actor_only=True)
            self.actor.load_state_dict(state["actor"])
            self.update_count = int(state.get("update_count", self.update_count))

    def _ensure_networks_initialized(self, actor_only: bool = False) -> None:
        dummy = self._dummy_obs_torch()
        with torch.no_grad():
            self.actor.deterministic(dummy)
            if actor_only:
                return
            action = torch.zeros((1, self.action_dim), dtype=torch.float32, device=self.device)
            for critic in self.critics:
                critic(dummy, action)
            for target in self.critic_targets:
                target(dummy, action)

    def _dummy_obs_torch(self) -> dict[str, torch.Tensor]:
        dummy: dict[str, torch.Tensor] = {
            "proprio": torch.zeros((1, self.proprio_dim), dtype=torch.float32, device=self.device),
        }
        for key in self.image_keys:
            dummy[f"image_{key}"] = torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=self.device)
        return dummy

    def _move_optimizer_state_to_device(self, optimizer: torch.optim.Optimizer) -> None:
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self.device)
