# Fake Debug

This example family is only for lightweight interface tests and local smoke
checks. It is not a training framework and it is not the main entrypoint for
RLT or PLD experiments.

The real LIBERO training loops live in:

- `examples/libero/rlt/scripts/train.py`
- `examples/libero/pld/scripts/train.py`

The local debug loops here intentionally stay under `examples/` so that
`vla_rl.runtime` remains limited to transport, checkpointing, timing, and
logging utilities.
