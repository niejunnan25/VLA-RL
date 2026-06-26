from __future__ import annotations

import copy
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from vla_rl.algorithms.base import Algorithm
from vla_rl.algorithms.pld.action import ResidualActionSpec
from vla_rl.algorithms.pld.modeling import GaussianResidualActor, PLDCritic, PLDObsEncoder
from vla_rl.nn import soft_update
from vla_rl.data import Observation, PolicyFeatures, RolloutBatch, Transition


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
        )
        self.actor = GaussianResidualActor(
            actor_encoder,
            residual_action_dim=self.residual_spec.policy_action_dim,
            hidden_dims=list(actor_hidden_dims),
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
                    ),
                    final_action_dim=self.residual_spec.critic_action_dim,
                    hidden_dims=list(critic_hidden_dims),
                )
                for _ in range(2)
            ]
        ).to(self.device)
        self.critic_targets = torch.nn.ModuleList([copy.deepcopy(critic) for critic in self.critics]).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=self.critic_lr)
        init_log_temp = float(np.log(max(float(init_temperature), 1e-6)))
        self.log_temperature = torch.nn.Parameter(torch.tensor(init_log_temp, dtype=torch.float32, device=self.device))
        self.temperature_optimizer = torch.optim.Adam([self.log_temperature], lr=self.temperature_lr)
        self.target_entropy = (
            float(target_entropy)
            if target_entropy is not None
            else -0.5 * float(self.residual_spec.policy_action_dim)
        )

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
        return self.actor.deterministic(batch) if deterministic else self.actor.sample(batch)[0]

    def _compose_final_action(self, batch: dict[str, torch.Tensor], residual: torch.Tensor) -> torch.Tensor:
        return self.residual_spec.compose_chunk_torch(batch["base_action_chunk"], residual)

    def update(self, batch: RolloutBatch) -> dict:
        batch.validate()
        fb = self._convert_batch(batch)
        info: dict[str, float] = {}
        utd_ratio = max(1, int(self.utd_ratio))
        actor_update_ratio = max(1, int(self.critic_actor_ratio))
        # High UTD: do `utd_ratio` critic gradient steps, each on a DISTINCT
        # mini-batch sliced from the sampled batch, instead of reusing one batch.
        critic_batches = self._split_batch(fb, utd_ratio)
        critic_infos: list[dict[str, float]] = []
        for idx in range(utd_ratio):
            minibatch = fb if utd_ratio == 1 else self._index_batch(critic_batches, idx)
            critic_infos.append(self._critic_step(minibatch))
            self._critic_step_count += 1
            if self._critic_step_count % actor_update_ratio == 0:
                actor_info = self._actor_step(minibatch)
                info.update(actor_info)
                info.update(self._temperature_step(actor_info.get("actor_log_prob_mean", 0.0)))
            self._update_targets()
        info.update(self._mean_infos(critic_infos))
        self.update_count += 1
        info["updates"] = float(self.update_count)
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
        with torch.no_grad():
            next_residual, next_log_prob = self.actor.sample(next_obs)
            next_final = self.residual_spec.compose_chunk_torch(next_obs["base_action_chunk"], next_residual).reshape(action.shape[0], -1)
            target_qs = [target(next_obs, next_final) for target in self.critic_targets]
            min_target_q = torch.min(torch.cat(target_qs, dim=-1), dim=-1, keepdim=True).values
            if self.backup_entropy:
                min_target_q = min_target_q - self.temperature.detach() * next_log_prob
            target = reward + (1.0 - done) * discount * min_target_q

        q_preds = [critic(obs, action) for critic in self.critics]
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
        residual, log_prob = self.actor.sample(obs)
        final = self.residual_spec.compose_chunk_torch(obs["base_action_chunk"], residual).reshape(residual.shape[0], -1)
        q_values = [critic(obs, final) for critic in self.critics]
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
            policy_residual, _ = self.actor.sample(obs)
            policy_final = self.residual_spec.compose_chunk_torch(obs["base_action_chunk"], policy_residual).reshape(batch_size, -1)
            for critic in self.critics:
                candidate_qs.append(critic(obs, random_final))
                q_pi = critic(obs, policy_final)
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
            values = [np.asarray(obs[key], dtype=np.float32) for obs in obs_list]
            result[key] = torch.as_tensor(np.stack(values), dtype=torch.float32, device=self.device)
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

    def _update_targets(self) -> None:
        for critic, target in zip(self.critics, self.critic_targets):
            soft_update(critic, target, self.tau)

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
        self._ensure_networks_initialized(actor_only=True)
        return {"actor": self.actor.state_dict(), "update_count": self.update_count}

    def load_policy_state_dict(self, state: dict) -> None:
        self._ensure_networks_initialized(actor_only=True)
        self.actor.load_state_dict(state["actor"])
        self.update_count = int(state.get("update_count", self.update_count))

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
