# AgiBot RLT

AgiBot RLT follows the same frozen-OpenPI + RLToken path as LIBERO RLT:

1. `scripts/serve_reference_policy.py` runs OpenPI in a separate process and returns `reference_actions` plus OpenPI prefix tokens.
2. The actor loads the frozen RLTokenEncoder locally and encodes prefix tokens into `z_rl`.
3. Before critical phase, the robot executes the OpenPI reference action. After manual `c` handover, or a metadata handover signal, only critical-phase transitions are sent to the learner.


## Launch

```bash
cd /vla/users/niejunnan/codebase/VLA-RL

bash examples/agibot_real/rlt/launch_agentlace.sh   runtime.max_env_steps=1000   runtime.max_update_steps=1000
```

Useful environment overrides:

```bash
POLICY_ROOT=/vla/users/niejunnan/codebase/openpi-rlt-github
POLICY_PYTHON=/vla/users/niejunnan/codebase/openpi-modified/.venv/bin/python3
POLICY_CHECKPOINT=/vla/users/niejunnan/assets/openpi-assets/checkpoints/pi0_libero_pytorch
POLICY_PORT=8899
ACTOR_GPU=0
LEARNER_GPU=1
```

Set `env.backend=local` for the real robot. The checked-in config uses `fake` for interface tests.
