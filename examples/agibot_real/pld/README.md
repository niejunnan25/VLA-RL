# AgiBot PLD / Residual Stage 1

This example runs the PLD Stage-1 residual RL path on AgiBot:

1. `scripts/serve_reference_policy.py` runs the frozen OpenPI base policy in a separate process.
2. The actor requests OpenPI `reference_actions` and builds a residual observation.
3. `PLDSACAgent.sample_action()` returns the composed final action chunk, which is executed on the robot.

No PLD Stage-2 hybrid data collection or Stage-3 distillation is implemented here.

## Launch

```bash
cd /vla/users/niejunnan/codebase/VLA-RL

bash examples/agibot_real/pld/launch_agentlace.sh   runtime.max_env_steps=1000   runtime.max_update_steps=1000
```

Useful environment overrides:

```bash
POLICY_ROOT=/vla/users/niejunnan/codebase/openpi-rlt-github
POLICY_PYTHON=/vla/users/niejunnan/codebase/openpi-modified/.venv/bin/python3
POLICY_CHECKPOINT=/vla/users/niejunnan/assets/openpi-assets/checkpoints/pi0_libero_pytorch
POLICY_PORT=8898
ACTOR_GPU=0
LEARNER_GPU=1
```

Set `env.backend=local` for the real robot. The checked-in config uses `fake` for interface tests.
