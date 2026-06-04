from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from vla_rl.algorithms.base import Algorithm
from vla_rl.algorithms.rlt.modeling import RLTActor, RLTCritic
from vla_rl.data import RolloutBatch


class RLTAgent(Algorithm):
    def __init__(
        self,
        z_rl_dim: int = 2048,
        proprio_dim: int = 8,
        action_dim: int = 7,
        chunk_size: int = 10,
        actor_hidden_dims: tuple[int, ...] = (512, 512, 512),
        critic_hidden_dims: tuple[int, ...] = (512, 512, 512),
        actor_std: float = 0.01,
        num_critics: int = 2,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        bc_reg_coeff: float = 3.0,
        ref_dropout: float = 0.5,
        clip_grad_norm: float = 10.0,
        policy_update_freq: int = 2,
        utd_ratio: int = 1,
        device: str = "cpu",
    ) -> None:
        del proprio_dim
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.device = torch.device(device)
        self.z_rl_dim = int(z_rl_dim)
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.actor_hidden_dims = tuple(int(dim) for dim in actor_hidden_dims)
        self.critic_hidden_dims = tuple(int(dim) for dim in critic_hidden_dims)
        self.actor_std = float(actor_std)
        self.num_critics = int(num_critics)
        self.actor_lr = float(actor_lr)
        self.critic_lr = float(critic_lr)
        self.discount = float(discount)
        self.tau = float(tau)
        self.bc_reg_coeff = float(bc_reg_coeff)
        self.ref_dropout = float(ref_dropout)
        self.clip_grad_norm = float(clip_grad_norm)
        self.policy_update_freq = int(policy_update_freq)
        self.utd_ratio = int(utd_ratio)
        self.update_count = 0
        self._critic_step_count = 0

        action_chunk_dim = self.chunk_size * self.action_dim
        self.actor = RLTActor(
            state_dim=self.z_rl_dim,
            action_chunk_dim=action_chunk_dim,
            hidden_dims=list(self.actor_hidden_dims),
            std=self.actor_std,
        ).to(self.device)
        self.critics = torch.nn.ModuleList(
            [RLTCritic(self.z_rl_dim, action_chunk_dim, list(self.critic_hidden_dims)) for _ in range(self.num_critics)]
        ).to(self.device)
        self.critic_targets = torch.nn.ModuleList([copy.deepcopy(critic) for critic in self.critics]).to(self.device)
        for target in self.critic_targets:
            target.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(actor_lr))
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=float(critic_lr))

    @torch.no_grad()
    def sample_action(
        self,
        rlt_state: dict[str, Any],
        deterministic: bool = False,
    ) -> np.ndarray:
        z_rl = torch.as_tensor(rlt_state["z_rl"], dtype=torch.float32, device=self.device).reshape(1, -1)
        ref = torch.as_tensor(rlt_state["reference_action"], dtype=torch.float32, device=self.device).reshape(1, -1)
        if deterministic:
            action = self.actor(z_rl, ref)
        else:
            action, _ = self.actor.sample(z_rl, ref)
        action_np = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return action_np.reshape(self.chunk_size, self.action_dim)

    def update(self, batch: RolloutBatch) -> dict:
        batch.validate()
        fb = self._convert_batch(batch)
        info: dict[str, float] = {}
        for _ in range(max(1, self.utd_ratio)):
            critic_loss, target_q_mean, predicted_q_mean = self._critic_step(fb)
            info.update(
                {
                    "loss_critic": critic_loss,
                    "target_q_mean": target_q_mean,
                    "predicted_q_mean": predicted_q_mean,
                }
            )
            self._update_target_networks()
            self._critic_step_count += 1
            if self._critic_step_count % self.policy_update_freq == 0:
                info.update(self._actor_step(fb))
        self.update_count += 1
        info["updates"] = self.update_count
        return info

    def _convert_batch(self, batch: RolloutBatch) -> dict[str, torch.Tensor]:
        transitions = batch.transitions
        z_rl = self._stack_obs_key(transitions, "z_rl")
        next_z_rl = self._stack_next_key(transitions, "z_rl")
        ref_action = self._stack_obs_key(transitions, "reference_action")
        next_ref_action = self._stack_next_key(transitions, "reference_action")
        action_mask = self._stack_action_mask(transitions)
        actions = np.stack([transition.action for transition in transitions]).astype(np.float32)
        actions = actions * action_mask
        rewards = np.asarray([transition.reward for transition in transitions], dtype=np.float32)[:, None]
        dones = np.asarray([transition.done or transition.truncated for transition in transitions], dtype=np.float32)[:, None]
        discounts = np.asarray([transition.discount for transition in transitions], dtype=np.float32)[:, None]
        return {
            "state": torch.as_tensor(z_rl, dtype=torch.float32, device=self.device),
            "next_state": torch.as_tensor(next_z_rl, dtype=torch.float32, device=self.device),
            "action": torch.as_tensor(actions, dtype=torch.float32, device=self.device),
            "reward": torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
            "done": torch.as_tensor(dones, dtype=torch.float32, device=self.device),
            "discount": torch.as_tensor(discounts, dtype=torch.float32, device=self.device),
            "reference_action": torch.as_tensor(ref_action, dtype=torch.float32, device=self.device),
            "next_reference_action": torch.as_tensor(next_ref_action, dtype=torch.float32, device=self.device),
            "action_mask": torch.as_tensor(action_mask, dtype=torch.float32, device=self.device),
        }

    def _critic_step(self, fb: dict[str, torch.Tensor]) -> tuple[float, float, float]:
        state = fb["state"]
        next_state = fb["next_state"]
        action = fb["action"]
        reward = fb["reward"]
        done = fb["done"]
        discount = fb["discount"]
        with torch.no_grad():
            next_action = self.actor(next_state, fb["next_reference_action"])
            target_qs = [target(next_state, next_action) for target in self.critic_targets]
            min_target_q = torch.min(torch.cat(target_qs, dim=-1), dim=-1, keepdim=True).values
            td_target = reward + (1.0 - done) * discount * min_target_q
        q_preds = [critic(state, action) for critic in self.critics]
        loss = sum(F.mse_loss(q, td_target) for q in q_preds)
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critics.parameters(), max_norm=self.clip_grad_norm)
        self.critic_optimizer.step()
        predicted_q_mean = torch.stack([q.mean() for q in q_preds]).mean().item()
        return loss.item(), td_target.mean().item(), predicted_q_mean

    def _actor_step(self, fb: dict[str, torch.Tensor]) -> dict[str, float]:
        state = fb["state"]
        ref = fb["reference_action"]
        action_mask = fb["action_mask"]
        mask = (torch.rand(ref.shape[0], 1, device=self.device) > self.ref_dropout).float()
        ref_input = ref * mask
        action = self.actor(state, ref_input)
        action_for_q = action * action_mask
        q_value = self.critics[0](state, action_for_q)
        bc_loss = ((action - ref).pow(2) * action_mask).sum() / action_mask.sum().clamp_min(1.0)
        loss = -q_value.mean() + self.bc_reg_coeff * bc_loss
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.clip_grad_norm)
        self.actor_optimizer.step()
        return {"loss_actor": loss.item(), "bc_loss": bc_loss.item(), "q_value_mean": q_value.mean().item()}

    def _update_target_networks(self) -> None:
        for critic, target in zip(self.critics, self.critic_targets):
            for param, target_param in zip(critic.parameters(), target.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)

    @staticmethod
    def _stack_obs_key(transitions, key: str) -> np.ndarray:
        values = []
        for transition in transitions:
            values.append(np.asarray(transition.obs[key], dtype=np.float32).reshape(-1))
        return np.stack(values)

    def _stack_next_key(self, transitions, key: str) -> np.ndarray:
        values = []
        for transition in transitions:
            source = transition.next_obs if transition.next_obs is not None else transition.obs
            values.append(np.asarray(source[key], dtype=np.float32).reshape(-1))
        return np.stack(values)

    def _stack_action_mask(self, transitions) -> np.ndarray:
        values = []
        for transition in transitions:
            values.append(np.asarray(transition.obs["action_mask"], dtype=np.float32).reshape(-1))
        return np.stack(values)

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critics": self.critics.state_dict(),
            "critic_targets": self.critic_targets.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "update_count": self.update_count,
            "critic_step_count": self._critic_step_count,
            "config": {
                "z_rl_dim": self.z_rl_dim,
                "action_dim": self.action_dim,
                "chunk_size": self.chunk_size,
                "actor_hidden_dims": self.actor_hidden_dims,
                "critic_hidden_dims": self.critic_hidden_dims,
                "actor_std": self.actor_std,
                "num_critics": self.num_critics,
                "actor_lr": self.actor_lr,
                "critic_lr": self.critic_lr,
                "discount": self.discount,
                "tau": self.tau,
                "bc_reg_coeff": self.bc_reg_coeff,
                "ref_dropout": self.ref_dropout,
                "clip_grad_norm": self.clip_grad_norm,
                "policy_update_freq": self.policy_update_freq,
                "utd_ratio": self.utd_ratio,
            },
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critics.load_state_dict(state["critics"])
        self.critic_targets.load_state_dict(state["critic_targets"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.update_count = int(state.get("update_count", 0))
        self._critic_step_count = int(state.get("critic_step_count", 0))
        self._move_optimizer_state_to_device(self.actor_optimizer)
        self._move_optimizer_state_to_device(self.critic_optimizer)

    def policy_state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "update_count": self.update_count,
            "critic_step_count": self._critic_step_count,
        }

    def load_policy_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.update_count = int(state.get("update_count", self.update_count))
        self._critic_step_count = int(state.get("critic_step_count", self._critic_step_count))

    def _move_optimizer_state_to_device(self, optimizer: torch.optim.Optimizer) -> None:
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)
