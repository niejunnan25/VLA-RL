from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class HandoverClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def predict_prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


def save_handover_classifier(classifier: HandoverClassifier, path: str | Path, **extra: Any) -> None:
    torch.save(
        {
            "input_dim": classifier.input_dim,
            "hidden_dim": classifier.hidden_dim,
            "state_dict": classifier.state_dict(),
            **extra,
        },
        path,
    )


def load_handover_classifier(path: str | Path, device: str = "cpu") -> HandoverClassifier:
    payload = torch.load(path, map_location="cpu")
    model = HandoverClassifier(input_dim=int(payload["input_dim"]), hidden_dim=int(payload.get("hidden_dim", 512)))
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()


def predict_handover_prob(classifier: HandoverClassifier, hidden: torch.Tensor) -> float:
    if hidden.dim() == 1:
        pooled = hidden.unsqueeze(0)
    elif hidden.dim() == 2:
        pooled = hidden.mean(dim=0, keepdim=True)
    else:
        pooled = hidden.mean(dim=1)
    pooled = pooled.to(next(classifier.parameters()).device)
    return float(classifier.predict_prob(pooled).item())


@dataclass(slots=True)
class HandoverRecord:
    episode_id: int | str
    step: int
    mean_hidden: np.ndarray
    handover: int


class HandoverLabelCollector:
    """Synchronous label collector for operator handover events."""

    def __init__(self, output_path: str | Path, lookahead: int) -> None:
        self.output_path = Path(output_path)
        self.lookahead = max(0, int(lookahead))
        self._buffers: defaultdict[int | str, deque[tuple[int, np.ndarray]]] = defaultdict(deque)

    def submit_hidden(self, episode_id: int | str, step: int, mean_hidden: np.ndarray) -> None:
        buffer = self._buffers[episode_id]
        buffer.append((int(step), np.asarray(mean_hidden, dtype=np.float32).reshape(-1)))
        while len(buffer) > self.lookahead:
            old_step, old_hidden = buffer.popleft()
            self._write(HandoverRecord(episode_id, old_step, old_hidden, 0))

    def mark_handover(self, episode_id: int | str) -> None:
        self._flush_episode(episode_id, handover=1)

    def end_episode(self, episode_id: int | str) -> None:
        self._flush_episode(episode_id, handover=0)

    def close(self) -> None:
        for episode_id in list(self._buffers):
            self._flush_episode(episode_id, handover=0)

    def _flush_episode(self, episode_id: int | str, *, handover: int) -> None:
        records = self._buffers.pop(episode_id, ())
        for step, hidden in records:
            self._write(HandoverRecord(episode_id, step, hidden, int(handover)))

    def _write(self, record: HandoverRecord) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "episode_id": record.episode_id,
                        "step": int(record.step),
                        "mean_hidden": record.mean_hidden.tolist(),
                        "handover": int(record.handover),
                    }
                )
                + "\n"
            )
