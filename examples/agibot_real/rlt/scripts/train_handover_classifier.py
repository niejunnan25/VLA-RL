#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.agibot_real.rlt.handover import HandoverClassifier, save_handover_classifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vectors, labels = [], []
    with Path(args.labels).open() as f:
        for line in f:
            record = json.loads(line)
            vectors.append(np.asarray(record["mean_hidden"], dtype=np.float32))
            labels.append(float(record["handover"]))
    x = torch.as_tensor(np.stack(vectors), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(labels, dtype=np.float32))
    loader = DataLoader(TensorDataset(x, y), batch_size=int(args.batch_size), shuffle=True)
    device = torch.device(args.device if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu")
    model = HandoverClassifier(input_dim=x.shape[-1], hidden_dim=int(args.hidden_dim)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    for _ in range(int(args.epochs)):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            loss = F.binary_cross_entropy_with_logits(model(bx), by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    save_handover_classifier(model.cpu(), args.output, epochs=int(args.epochs))


if __name__ == "__main__":
    main()
