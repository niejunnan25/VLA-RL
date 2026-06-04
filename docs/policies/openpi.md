# OpenPI Policy Backend

`OpenPIBackend` adapts a Torch OpenPI policy to the VLA-RL `PolicyBackend`
interface.

## Responsibilities

- Dynamically add `openpi_root/src` to `sys.path`.
- Load an OpenPI trained policy from `config_name` and `checkpoint_path`.
- Convert `Observation` into an OpenPI-style observation dict.
- Return base policy action chunks as `np.ndarray` with shape `[chunk, action_dim]`.
- Return OpenPI reference actions and prefix hidden states as `PolicyFeatures`.

## Observation Bridge

The backend first checks:

```python
Observation.raw["openpi_observation"]
```

If present, that dict is passed directly to OpenPI. Otherwise the backend builds
a minimal dict from:

- `Observation.images` -> `image/<key>`
- `Observation.proprio` -> `state`
- `Observation.task` or explicit `task` -> `prompt`

## Feature Keys

`extract_features()` returns embeddings with stable keys:

- `prefix`: VLA prefix hidden states.

RLT-specific encoded features should use `z_rl` later, outside this backend.

## Requirements

The OpenPI model must expose:

```python
predict_action_with_features(...)
```

The method must return both normalized reference actions and prefix features
from the same model pass. Use the `niejunnan25/openpi` RLT branch. The first supported real
checkpoints are:

```text
/vla/users/niejunnan/assets/openpi-assets/checkpoints/pi05_libero_pytorch
/vla/users/niejunnan/assets/openpi-assets/checkpoints/pi0_libero_pytorch
```

OpenPI policy loading expects checkpoint-local assets, including
`assets/physical-intelligence/libero/norm_stats.json`. A converted Torch
checkpoint without this assets directory is not sufficient for runtime policy
inference.
