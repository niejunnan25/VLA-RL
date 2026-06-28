from __future__ import annotations

import copy
import math
from contextlib import nullcontext
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch._subclasses.functional_tensor import FunctionalTensor
except Exception:  # pragma: no cover - older torch builds do not expose it.
    FunctionalTensor = ()  # type: ignore[assignment]

from vla_rl.algorithms.base import Algorithm
from vla_rl.algorithms.pld.action import ResidualActionSpec
from vla_rl.algorithms.pld.modeling import GaussianResidualActor, PLDCritic, PLDObsEncoder
from vla_rl.nn import soft_update
from vla_rl.data import Observation, PolicyFeatures, RolloutBatch, Transition

_COMPILE_KWARGS = {
    "backend": "inductor",
    "mode": "default",
    "fullgraph": True,
    "dynamic": False,
}


class PLDCriticEnsemble(nn.Module):
    def __init__(self, critics: torch.nn.ModuleList) -> None:
        super().__init__()
        self.critics = critics

    def forward(self, obs: dict[str, torch.Tensor], action: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(critic(obs, action) for critic in self.critics)


class PLDSACAgent(Algorithm):
    """PLD Stage-1 residual SAC agent."""

    def __init__(
        self,
        image_keys: Sequence[str] = ("image_rgb_0", "image_rgb_1"),
        proprio_dim: int = 8,
        action_dim: int = 7,
        chunk_horizon: int = 1,
        alpha: float = 0.5,
        action_mask: Sequence[bool] | None = None,
        action_limits: Sequence[float] | None = None,
        clip_gripper: bool = True,
        image_encoder_type: str = "small",
        image_feature_dim: int = 64,
        vector_latent_dim: int = 64,
        encoder_hidden_dim: int = 256,
        project_obs: bool = True,
        resnet_model_name: str = "microsoft/resnet-18",
        resnet_pretrained: bool = True,
        freeze_image_backbone: bool = True,
        resnet_pooling_method: str = "spatial_learned_embeddings",
        resnet_num_spatial_blocks: int = 8,
        resnet_spatial_dropout_rate: float = 0.0,
        fuse_views: bool = True,
        actor_hidden_dims: Sequence[int] = (256, 256, 256),
        critic_hidden_dims: Sequence[int] = (256, 256, 256),
        actor_activation: str = "relu",
        critic_activation: str = "relu",
        actor_layer_norm: bool = True,
        critic_layer_norm: bool = True,
        std_min: float = 1e-5,
        std_max: float = 1.0,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temperature_lr: float = 3e-4,
        optimizer_type: str = "adam",
        weight_decay: float = 0.0,
        temperature_weight_decay: float = 0.0,
        mixed_precision_enabled: bool = False,
        mixed_precision_dtype: str = "bfloat16",
        torch_compile_enabled: bool = False,
        torch_compile_target: str = "actor_critic",
        torch_compile_backend: str = "inductor",
        torch_compile_mode: str = "default",
        torch_compile_fullgraph: bool = True,
        torch_compile_dynamic: bool = False,
        discount: float = 0.99,
        tau: float = 0.005,
        init_temperature: float = 1.0,
        target_entropy: float | None = None,
        backup_entropy: bool = False,
        utd_ratio: int = 1,
        critic_actor_ratio: int = 2,
        clip_grad_norm: float = 1.0,
        cql_n_actions: int = 10,
        cql_temperature: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.image_keys = tuple(str(key) for key in image_keys)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.chunk_horizon = int(chunk_horizon)
        self.residual_spec = ResidualActionSpec(
            full_action_dim=self.action_dim,
            action_mask=None if action_mask is None else tuple(bool(x) for x in action_mask),
            action_limits=None if action_limits is None else tuple(float(x) for x in action_limits),
            alpha=float(alpha),
            clip_gripper=bool(clip_gripper),
            chunk_horizon=self.chunk_horizon,
        )
        self.actor_lr = float(actor_lr)
        self.critic_lr = float(critic_lr)
        self.temperature_lr = float(temperature_lr)
        self.optimizer_type = str(optimizer_type).lower()
        self.weight_decay = float(weight_decay)
        self.temperature_weight_decay = float(temperature_weight_decay)
        self.mixed_precision_enabled = bool(mixed_precision_enabled)
        self.mixed_precision_dtype = str(mixed_precision_dtype)
        self.torch_compile_enabled = bool(torch_compile_enabled)
        self.torch_compile_target = str(torch_compile_target)
        self.torch_compile_backend = str(torch_compile_backend)
        self.torch_compile_mode = str(torch_compile_mode)
        self.torch_compile_fullgraph = bool(torch_compile_fullgraph)
        self.torch_compile_dynamic = bool(torch_compile_dynamic)
        self.discount = float(discount)
        self.tau = float(tau)
        self.backup_entropy = bool(backup_entropy)
        self.utd_ratio = int(utd_ratio)
        self.critic_actor_ratio = int(critic_actor_ratio)
        self.clip_grad_norm = float(clip_grad_norm)
        self.cql_n_actions = int(cql_n_actions)
        self.cql_temperature = float(cql_temperature)
        self.update_count = 0
        self._critic_step_count = 0

        actor_encoder = PLDObsEncoder(
            self.image_keys,
            self.proprio_dim,
            self.action_dim,
            self.chunk_horizon,
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
            resnet_spatial_dropout_rate=resnet_spatial_dropout_rate,
            fuse_views=fuse_views,
        )
        self.actor = GaussianResidualActor(
            actor_encoder,
            residual_action_dim=self.residual_spec.policy_action_dim,
            hidden_dims=list(actor_hidden_dims),
            log_std_min=math.log(max(float(std_min), 1e-12)),
            log_std_max=math.log(max(float(std_max), 1e-12)),
            activation=str(actor_activation),
            layer_norm=bool(actor_layer_norm),
        ).to(self.device)
        self.critics = torch.nn.ModuleList(
            [
                PLDCritic(
                    PLDObsEncoder(
                        self.image_keys,
                        self.proprio_dim,
                        self.action_dim,
                        self.chunk_horizon,
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
                        resnet_spatial_dropout_rate=resnet_spatial_dropout_rate,
                        fuse_views=fuse_views,
                    ),
                    final_action_dim=self.residual_spec.critic_action_dim,
                    hidden_dims=list(critic_hidden_dims),
                    activation=str(critic_activation),
                    layer_norm=bool(critic_layer_norm),
                )
                for _ in range(2)
            ]
        ).to(self.device)
        self.critic_targets = torch.nn.ModuleList([copy.deepcopy(critic) for critic in self.critics]).to(self.device)
        self.actor_optimizer = self._make_optimizer(self.actor.parameters(), lr=self.actor_lr, weight_decay=self.weight_decay)
        self.critic_optimizer = self._make_optimizer(self.critics.parameters(), lr=self.critic_lr, weight_decay=self.weight_decay)
        init_log_temp = float(np.log(max(float(init_temperature), 1e-6)))
        self.log_temperature = torch.nn.Parameter(torch.tensor(init_log_temp, dtype=torch.float32, device=self.device))
        self.temperature_optimizer = self._make_optimizer(
            [self.log_temperature],
            lr=self.temperature_lr,
            weight_decay=self.temperature_weight_decay,
        )
        self.target_entropy = (
            float(target_entropy)
            if target_entropy is not None
            else -0.5 * float(self.residual_spec.policy_action_dim)
        )
        self._compiled_actor: nn.Module | None = None
        self._compiled_critics: nn.Module | None = None
        self._compiled_critic_targets: nn.Module | None = None

    @torch.no_grad()
    def sample_action(
        self,
        pld_obs: dict[str, Any],
        deterministic: bool = False,
    ) -> np.ndarray:
        """Return the composed final action chunk executed by the environment."""
        batch = self._obs_to_torch([pld_obs])
        residual = self._sample_residual(batch, deterministic=deterministic)
        final_action = self._compose_final_action(batch, residual)
        return final_action.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def _sample_residual(self, batch: dict[str, torch.Tensor], *, deterministic: bool) -> torch.Tensor:
        return self._actor_deterministic(batch) if deterministic else self._actor_sample(batch)[0]

    def _compose_final_action(self, batch: dict[str, torch.Tensor], residual: torch.Tensor) -> torch.Tensor:
        return self.residual_spec.compose_chunk_torch(batch["base_action_chunk"], residual)

    def update(self, batch: RolloutBatch) -> dict:
        return self.update_high_utd(batch)

    def update_critics(self, batch: RolloutBatch) -> dict:
        batch.validate()
        fb = self._convert_batch(batch)
        info = self._critic_step(fb)
        self._critic_step_count += 1
        self._update_targets()
        info["critic_steps"] = float(self._critic_step_count)
        return info

    def update_high_utd(self, batch: RolloutBatch, utd_ratio: int | None = None) -> dict:
        batch.validate()
        fb = self._convert_batch(batch)
        info: dict[str, float] = {}
        high_utd = max(1, int(self.utd_ratio if utd_ratio is None else utd_ratio))
        # Mirror serl_torch update_high_utd: split one sampled batch into
        # distinct critic mini-batches, then run one actor/temperature update on
        # the full batch.
        critic_batches = self._split_batch(fb, high_utd)
        critic_infos: list[dict[str, float]] = []
        for idx in range(high_utd):
            minibatch = fb if high_utd == 1 else self._index_batch(critic_batches, idx)
            critic_infos.append(self._critic_step(minibatch))
            self._critic_step_count += 1
            self._update_targets()
        actor_info = self._actor_step(fb)
        info.update(actor_info)
        info.update(self._temperature_step(actor_info.get("actor_log_prob_mean", 0.0)))
        info.update(self._mean_infos(critic_infos))
        self.update_count += 1
        info["updates"] = float(self.update_count)
        info["critic_steps"] = float(self._critic_step_count)
        return info

    def update_critics_calql(
        self,
        batch: RolloutBatch,
        calql_alpha: float = 1.0,
        calql_n_actions: int | None = None,
        calql_temperature: float | None = None,
    ) -> dict:
        batch.validate()
        fb = self._convert_batch(batch)
        return self._critic_step(
            fb,
            calql_alpha=float(calql_alpha),
            calql_n_actions=int(calql_n_actions or self.cql_n_actions),
            calql_temperature=float(calql_temperature or self.cql_temperature),
        )

    def _critic_step(
        self,
        fb: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        calql_alpha: float = 0.0,
        calql_n_actions: int | None = None,
        calql_temperature: float | None = None,
    ) -> dict[str, float]:
        obs = fb["obs"]
        next_obs = fb["next_obs"]
        action = fb["action"]
        reward = fb["reward"]
        done = fb["done"]
        discount = fb["discount"]
        with torch.no_grad(), self._autocast_context():
            next_residual, next_log_prob = self._actor_sample(next_obs)
            next_final = self.residual_spec.compose_chunk_torch(next_obs["base_action_chunk"], next_residual).reshape(action.shape[0], -1)
            target_qs = self._critic_values(self.critic_targets, next_obs, next_final)
            min_target_q = torch.min(torch.cat(target_qs, dim=-1), dim=-1, keepdim=True).values
            if self.backup_entropy:
                min_target_q = min_target_q - self.temperature.detach() * next_log_prob
            target = reward + (1.0 - done) * discount * min_target_q

        with self._autocast_context():
            q_preds = self._critic_values(self.critics, obs, action)
            td_loss = sum(torch.nn.functional.mse_loss(q, target) for q in q_preds)
        cql_penalty = torch.zeros((), dtype=torch.float32, device=self.device)
        bound_applied = torch.zeros((), dtype=torch.float32, device=self.device)
        if calql_alpha > 0.0:
            cql_penalty, bound_applied = self._cql_penalty(
                obs,
                data_action=action,
                q_preds=q_preds,
                mc_returns=fb["mc_returns"],
                mc_returns_valid=fb["mc_returns_valid"],
                n_actions=int(calql_n_actions or self.cql_n_actions),
                temperature=float(calql_temperature or self.cql_temperature),
            )
        loss = td_loss + float(calql_alpha) * cql_penalty
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critics.parameters(), self.clip_grad_norm)
        self.critic_optimizer.step()
        return {
            "loss_critic": float(loss.detach().cpu()),
            "loss_td": float(td_loss.detach().cpu()),
            "critic_cql_penalty": float(cql_penalty.detach().cpu()),
            "calql_bound_applied": float(bound_applied.detach().cpu()),
            "target_q_mean": float(target.mean().detach().cpu()),
            "predicted_q_mean": float(torch.stack([q.mean() for q in q_preds]).mean().detach().cpu()),
        }

    def _actor_step(self, fb: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> dict[str, float]:
        obs = fb["obs"]
        with self._autocast_context():
            residual, log_prob = self._actor_sample(obs)
            final = self.residual_spec.compose_chunk_torch(obs["base_action_chunk"], residual).reshape(residual.shape[0], -1)
            q_values = self._critic_values(self.critics, obs, final)
            min_q = torch.min(torch.cat(q_values, dim=-1), dim=-1, keepdim=True).values
            loss = (self.temperature.detach() * log_prob - min_q).mean()
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad_norm)
        self.actor_optimizer.step()
        return {
            "loss_actor": float(loss.detach().cpu()),
            "actor_q_mean": float(min_q.mean().detach().cpu()),
            "actor_log_prob_mean": float(log_prob.mean().detach().cpu()),
        }

    def _temperature_step(self, log_prob_mean: float) -> dict[str, float]:
        log_prob = torch.tensor(float(log_prob_mean), device=self.device)
        loss = -(self.log_temperature * (log_prob + self.target_entropy).detach())
        self.temperature_optimizer.zero_grad()
        loss.backward()
        self.temperature_optimizer.step()
        return {"temperature": float(self.temperature.detach().cpu()), "loss_temperature": float(loss.detach().cpu())}

    def _cql_penalty(
        self,
        obs: dict[str, torch.Tensor],
        data_action: torch.Tensor,
        q_preds: list[torch.Tensor],
        mc_returns: torch.Tensor,
        mc_returns_valid: torch.Tensor,
        n_actions: int,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = data_action.shape[0]
        candidate_qs = []
        for _ in range(max(1, int(n_actions))):
            random_residual = torch.empty(
                batch_size,
                self.residual_spec.policy_action_dim,
                device=self.device,
                dtype=data_action.dtype,
            ).uniform_(-1.0, 1.0)
            random_final = self.residual_spec.compose_chunk_torch(obs["base_action_chunk"], random_residual).reshape(batch_size, -1)
            policy_residual, _ = self._actor_sample(obs)
            policy_final = self.residual_spec.compose_chunk_torch(obs["base_action_chunk"], policy_residual).reshape(batch_size, -1)
            random_qs = self._critic_values(self.critics, obs, random_final)
            policy_qs = self._critic_values(self.critics, obs, policy_final)
            for q_random, q_pi in zip(random_qs, policy_qs):
                candidate_qs.append(q_random)
                valid = mc_returns_valid.reshape(batch_size, 1).bool()
                bounded_q = torch.where(valid, torch.maximum(q_pi, mc_returns.reshape(batch_size, 1)), q_pi)
                candidate_qs.append(bounded_q)
        q_cat = torch.stack(candidate_qs, dim=0).squeeze(-1)
        lse = torch.logsumexp(q_cat / max(float(temperature), 1e-6), dim=0) * max(float(temperature), 1e-6)
        data_q = torch.cat(q_preds, dim=-1).mean(dim=-1)
        return (lse - data_q).mean(), mc_returns_valid.float().mean()

    def _convert_batch(self, batch: RolloutBatch) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        transitions = batch.transitions
        obs = self._obs_to_torch([t.obs for t in transitions])
        next_obs = self._obs_to_torch([(t.next_obs if t.next_obs is not None else t.obs) for t in transitions])
        action = np.stack([np.asarray(t.action, dtype=np.float32).reshape(-1) for t in transitions])
        return {
            "obs": obs,
            "next_obs": next_obs,
            "action": torch.as_tensor(action, dtype=torch.float32, device=self.device),
            "reward": torch.tensor([[t.reward] for t in transitions], dtype=torch.float32, device=self.device),
            "done": torch.tensor(
                [[bool(t.info.get("critic_terminal", t.done or t.truncated))] for t in transitions],
                dtype=torch.float32,
                device=self.device,
            ),
            "discount": torch.tensor([[t.discount] for t in transitions], dtype=torch.float32, device=self.device),
            "mc_returns": torch.tensor([float(t.info.get("mc_returns", 0.0)) for t in transitions], dtype=torch.float32, device=self.device),
            "mc_returns_valid": torch.tensor([bool(t.info.get("mc_returns_valid", False)) for t in transitions], dtype=torch.float32, device=self.device),
        }

    def _obs_to_torch(self, obs_list: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        required_keys = [*(f"image_{key}" for key in self.image_keys), "proprio", "base_action_chunk", "alpha"]
        for key in required_keys:
            values = [np.asarray(obs[key]) for obs in obs_list]
            tensor = torch.as_tensor(np.stack(values), device=self.device)
            if key.startswith("image_") and not tensor.is_floating_point():
                tensor = tensor.to(dtype=torch.float32).div_(255.0)
            else:
                tensor = tensor.to(dtype=torch.float32)
            result[key] = tensor
        return result

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

    @staticmethod
    def _mean_infos(infos: list[dict[str, float]]) -> dict[str, float]:
        if not infos:
            return {}
        keys = set().union(*(info.keys() for info in infos))
        averaged: dict[str, float] = {}
        for key in keys:
            values = [float(info[key]) for info in infos if key in info]
            if values:
                averaged[key] = float(sum(values) / len(values))
        return averaged

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def set_compile_enabled(self, enabled: bool) -> None:
        self.torch_compile_enabled = bool(enabled)
        if not self.torch_compile_enabled:
            self._compiled_actor = None
            self._compiled_critics = None
            self._compiled_critic_targets = None

    def _actor_forward(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        actor = self._ensure_actor_compiled() or self.actor
        return actor(obs)

    def _actor_deterministic(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        mean, _ = self._actor_forward(obs)
        return torch.tanh(mean)

    def _actor_sample(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self._actor_forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        raw = normal.rsample()
        action = torch.tanh(raw)
        log_prob = normal.log_prob(raw) - torch.log(torch.clamp(1.0 - action.pow(2), min=1e-6))
        return action, log_prob.sum(dim=-1, keepdim=True)

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
        return [critic(obs, action) for critic in critics]

    def _make_optimizer(
        self,
        params,
        *,
        lr: float,
        weight_decay: float,
    ) -> torch.optim.Optimizer:
        if self.optimizer_type == "adamw":
            return torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay))
        if self.optimizer_type == "adam":
            return torch.optim.Adam(params, lr=float(lr))
        raise ValueError(f"unsupported optimizer_type={self.optimizer_type!r}")

    def _autocast_context(self):
        if not self.mixed_precision_enabled or self.device.type != "cuda":
            return nullcontext()
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
        }
        dtype = dtype_map.get(self.mixed_precision_dtype.lower())
        if dtype is None:
            raise ValueError(f"unsupported mixed_precision_dtype={self.mixed_precision_dtype!r}")
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    def _compile_available(self) -> bool:
        if not bool(self.torch_compile_enabled):
            return False
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return False
        if not hasattr(torch, "compile"):
            return False
        if self.torch_compile_target not in {"critic", "actor_critic"}:
            raise ValueError(
                "torch_compile_target must be one of {'critic', 'actor_critic'}, "
                f"got {self.torch_compile_target!r}"
            )
        return True

    def _compile_kwargs(self) -> dict[str, object]:
        return {
            **_COMPILE_KWARGS,
            "backend": self.torch_compile_backend,
            "mode": self.torch_compile_mode,
            "fullgraph": self.torch_compile_fullgraph,
            "dynamic": self.torch_compile_dynamic,
        }

    def _ensure_actor_compiled(self) -> nn.Module | None:
        if self._compiled_actor is not None:
            return self._compiled_actor
        if self.torch_compile_target != "actor_critic" or not self._compile_available():
            return None
        with torch.no_grad():
            self.actor.deterministic(self._dummy_obs_torch())
        self._compiled_actor = torch.compile(self.actor, **self._compile_kwargs())
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
        final_action = torch.zeros((1, self.residual_spec.critic_action_dim), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            PLDCriticEnsemble(critics).to(self.device)(dummy, final_action)
        compiled = torch.compile(PLDCriticEnsemble(critics).to(self.device), **self._compile_kwargs())
        if target:
            self._compiled_critic_targets = compiled
        else:
            self._compiled_critics = compiled
        return compiled

    def _update_targets(self) -> None:
        for critic, target in zip(self.critics, self.critic_targets):
            soft_update(critic, target, self.tau)

    def state_dict(self) -> dict:
        self._ensure_networks_initialized()
        return {
            "actor": self._plain_cpu_state_dict(self.actor.state_dict()),
            "critics": self._plain_cpu_state_dict(self.critics.state_dict()),
            "critic_targets": self._plain_cpu_state_dict(self.critic_targets.state_dict()),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "temperature_optimizer": self.temperature_optimizer.state_dict(),
            "log_temperature": self.log_temperature.detach().cpu().clone(),
            "update_count": self.update_count,
            "critic_step_count": self._critic_step_count,
        }

    def load_state_dict(self, state: dict) -> None:
        self._compiled_actor = None
        self._compiled_critics = None
        self._compiled_critic_targets = None
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
        self._ensure_networks_initialized(actor_only=True)
        return {"actor": self._plain_cpu_state_dict(self.actor.state_dict()), "update_count": self.update_count}

    def load_policy_state_dict(self, state: dict) -> None:
        if self._compiled_actor is None:
            self._ensure_networks_initialized(actor_only=True)
        self.actor.load_state_dict(state["actor"])
        self.update_count = int(state.get("update_count", self.update_count))

    @staticmethod
    def _plain_cpu_state_dict(state: dict[str, Any]) -> dict[str, Any]:
        plain: dict[str, Any] = {}
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                if hasattr(torch, "_is_functional_tensor") and torch._is_functional_tensor(value):
                    torch._sync(value)
                    value = torch._from_functional_tensor(value)
                elif FunctionalTensor and isinstance(value, FunctionalTensor):
                    value = value.from_functional()
                plain[key] = value.detach().to("cpu").clone()
            else:
                plain[key] = copy.deepcopy(value)
        return plain

    def _ensure_networks_initialized(self, actor_only: bool = False) -> None:
        dummy = self._dummy_obs_torch()
        with torch.no_grad():
            self.actor.deterministic(dummy)
            if actor_only:
                return
            final_action = torch.zeros((1, self.residual_spec.critic_action_dim), dtype=torch.float32, device=self.device)
            for critic in self.critics:
                critic(dummy, final_action)
            for target in self.critic_targets:
                target(dummy, final_action)

    def _dummy_obs_torch(self) -> dict[str, torch.Tensor]:
        dummy: dict[str, torch.Tensor] = {
            "proprio": torch.zeros((1, self.proprio_dim), dtype=torch.float32, device=self.device),
            "base_action_chunk": torch.zeros(
                (1, self.chunk_horizon, self.action_dim),
                dtype=torch.float32,
                device=self.device,
            ),
            "alpha": torch.full((1, 1), float(self.residual_spec.alpha), dtype=torch.float32, device=self.device),
        }
        for key in self.image_keys:
            dummy[f"image_{key}"] = torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=self.device)
        return dummy

    def _move_optimizer_state_to_device(self, optimizer: torch.optim.Optimizer) -> None:
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self.device)
