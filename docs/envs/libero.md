# LIBERO Env Backend

`LiberoRemoteEnvBackend` connects to a LIBERO environment server through
pickle-over-HTTP RPC. The backend converts raw LIBERO observations into the
shared `Observation` schema and stores an OpenPI-compatible observation dict in:

```python
Observation.raw["openpi_observation"]
```

This keeps OpenPI-specific input formatting outside `OpenPIBackend`.

## Server

Milestone 1 does not vendor the LIBERO simulator implementation. Start the
validated server through the compatibility wrapper:

```bash
python scripts/serve_libero_env.py \
  --serl-torch-root /vla/users/niejunnan/codebase/serl_torch-rlt-merge \
  --host 127.0.0.1 \
  --port 23000 \
  --gpu-id 0
```

The wrapper forwards all unknown arguments to
`examples/libero/scripts/serve_env.py` in the external `serl_torch` checkout.

## Client Recipe

```yaml
env:
  _target_: vla_rl.envs.libero.LiberoRemoteEnvBackend
  url: http://127.0.0.1:23000
  task_suite_name: libero_spatial
  task_id: 4
  seed: 0
```

`step_chunk()` is supported when the remote server exposes it. The local
actor-learner runtime uses `executed_steps` returned in `info` to set transition
discounts consistently with the number of environment steps actually executed.
