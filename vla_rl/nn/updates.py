from __future__ import annotations

import torch
import torch.nn as nn


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - float(tau)).add_(param.data, alpha=float(tau))
