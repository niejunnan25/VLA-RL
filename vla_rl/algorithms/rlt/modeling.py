from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev, int(hidden_dim)))
            if use_layer_norm:
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


class RLTokenEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        rl_token_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.rl_token_dim = int(rl_token_dim)
        self.e_rl = nn.Parameter(torch.randn(1, 1, int(input_dim)) * 0.02)
        if int(input_dim) != int(rl_token_dim):
            self.input_proj = nn.Linear(int(input_dim), int(rl_token_dim))
        else:
            self.input_proj = nn.Identity()
        layer = nn.TransformerEncoderLayer(
            d_model=int(rl_token_dim),
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=int(num_layers))

    def forward(self, z_vla: Tensor) -> Tensor:
        batch_size = z_vla.shape[0]
        e_rl = self.e_rl.expand(batch_size, -1, -1)
        seq = torch.cat([z_vla, e_rl], dim=1)
        seq = self.input_proj(seq)
        out = self.transformer(seq)
        return out[:, -1, :]


class RLTokenDecoder(nn.Module):
    def __init__(
        self,
        rl_token_dim: int,
        output_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        if int(rl_token_dim) != int(output_dim):
            self.rl_proj = nn.Linear(int(rl_token_dim), int(output_dim))
        else:
            self.rl_proj = nn.Identity()
        layer = nn.TransformerDecoderLayer(
            d_model=int(output_dim),
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=int(num_layers))
        self.output_head = nn.Linear(int(output_dim), int(output_dim))

    def forward(self, z_rl: Tensor, z_vla_stopped: Tensor) -> Tensor:
        seq_len = z_vla_stopped.shape[1]
        z_rl_proj = self.rl_proj(z_rl).unsqueeze(1)
        target = torch.cat([z_rl_proj, z_vla_stopped[:, :-1, :]], dim=1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=z_rl.device)
        decoded = self.transformer(tgt=target, memory=z_rl_proj, tgt_mask=causal_mask)
        return self.output_head(decoded)


class RLTActor(nn.Module):
    def __init__(self, state_dim: int, action_chunk_dim: int, hidden_dims: list[int], std: float = 0.01) -> None:
        super().__init__()
        self.net = MLP(int(state_dim) + int(action_chunk_dim), list(hidden_dims), int(action_chunk_dim))
        self.log_std = math.log(float(std))

    def forward(self, state: Tensor, ref_action_chunk: Tensor) -> Tensor:
        return self.net(torch.cat([state, ref_action_chunk], dim=-1))

    def sample(self, state: Tensor, ref_action_chunk: Tensor) -> tuple[Tensor, Tensor]:
        mean = self.forward(state, ref_action_chunk)
        std = math.exp(self.log_std)
        noise = torch.randn_like(mean) * std
        action = mean + noise
        log_prob = -0.5 * (noise / std).pow(2).sum(dim=-1)
        log_prob = log_prob - mean.shape[-1] * math.log(std * math.sqrt(2.0 * math.pi))
        return action, log_prob


class RLTCritic(nn.Module):
    def __init__(self, state_dim: int, action_chunk_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        self.net = MLP(int(state_dim) + int(action_chunk_dim), list(hidden_dims), output_dim=1)

    def forward(self, state: Tensor, action_chunk: Tensor) -> Tensor:
        return self.net(torch.cat([state, action_chunk], dim=-1))
