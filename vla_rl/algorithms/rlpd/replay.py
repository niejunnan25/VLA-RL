from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Iterable

from vla_rl.data import ReplayBuffer, Transition

EPISODE_GLOB = "episode_*.pkl"
MANIFEST_NAME = "manifest.json"


def write_offline_episode(
    output_dir: str | Path,
    episode_index: int,
    transitions: Iterable[Transition | dict[str, Any]],
    *,
    manifest_stats: dict[str, Any] | None = None,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payloads = [Transition.from_payload(t).to_payload() for t in transitions]
    path = output / f"episode_{int(episode_index):06d}.pkl"
    with path.open("wb") as fp:
        pickle.dump(payloads, fp, protocol=pickle.HIGHEST_PROTOCOL)
    stats = dict(manifest_stats or {})
    stats.setdefault("episodes_written", int(episode_index) + 1)
    stats.setdefault("steps_written", len(payloads))
    manifest = {"format": "vla-rl-rlpd-transition-v1", "stats": stats}
    (output / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def load_offline_replay(
    path: str | Path,
    *,
    capacity: int = 250_000,
    seed: int = 0,
    max_episodes: int | None = None,
    max_transitions: int | None = None,
) -> tuple[ReplayBuffer, dict[str, int]]:
    root = Path(path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"offline replay path does not exist: {root}")
    replay = ReplayBuffer(capacity=capacity, seed=seed)
    episode_files = sorted(root.glob(EPISODE_GLOB))
    if max_episodes is not None:
        episode_files = episode_files[: int(max_episodes)]
    transitions_loaded = 0
    episodes_loaded = 0
    for episode_file in episode_files:
        with episode_file.open("rb") as fp:
            episode = pickle.load(fp)
        episodes_loaded += 1
        for raw in episode:
            replay.add(raw)
            transitions_loaded += 1
            if max_transitions is not None and transitions_loaded >= int(max_transitions):
                return replay, {"episodes_loaded": episodes_loaded, "transitions_loaded": transitions_loaded}
    return replay, {"episodes_loaded": episodes_loaded, "transitions_loaded": transitions_loaded}
