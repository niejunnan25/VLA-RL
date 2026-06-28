# RLPD Output Layout Migration

## New layout rule

All LIBERO experiment YAML files should write outputs under the corresponding example package instead of the repository root `outputs/` directory.

For RLPD reward-model experiments, the canonical layout is:

```text
examples/libero/rlpd/outputs/reward_model/<campaign>/taskXX/
```

For the current LIBERO spatial sparse RLPD campaign, the campaign directory is:

```text
examples/libero/rlpd/outputs/reward_model/libero_spatial_rlpd_sparse_600k_s0/
```

The task directories are:

```text
task00
task01
...
task09
```

The output directories are ignored by git through the repository-level `outputs/` ignore rule, so experiment artifacts do not become source files.

## Config changes

The RLPD sparse YAML files now point directly to the new layout. Example:

```yaml
runtime:
  run_dir: examples/libero/rlpd/outputs/reward_model/libero_spatial_rlpd_sparse_600k_s0/task00
```

The launchers now preserve YAML-defined `runtime.run_dir` by default. `--run-dir` still exists, but it is an explicit override only.

This applies to the LIBERO launchers under:

```text
examples/libero/rlpd/tools/launch_rlpd.sh
examples/libero/pld/tools/launch_pld.sh
examples/libero/pld/tools/launch_pld_after_collect.sh
examples/libero/pld/tools/launch_residual_sac.sh
examples/libero/rlt/tools/launch_rlt.sh
```

## Migrated completed runs

The following completed RLPD sparse runs were moved from the old root-output layout:

```text
outputs/reward_model/239_localenv/libero_spatial_task0_rlpd_sparse
outputs/reward_model/239_localenv/libero_spatial_task1_rlpd_sparse
outputs/reward_model/239_localenv/libero_spatial_task2_rlpd_sparse
outputs/reward_model/239_localenv/libero_spatial_task3_rlpd_sparse
outputs/reward_model/239_localenv/libero_spatial_task4_rlpd_sparse
outputs/reward_model/239_localenv/libero_spatial_task5_rlpd_sparse
outputs/reward_model/239_localenv/libero_spatial_task6_rlpd_sparse
outputs/reward_model/239_localenv/libero_spatial_task7_rlpd_sparse
```

They now live at:

```text
examples/libero/rlpd/outputs/reward_model/libero_spatial_rlpd_sparse_600k_s0/task00
examples/libero/rlpd/outputs/reward_model/libero_spatial_rlpd_sparse_600k_s0/task01
examples/libero/rlpd/outputs/reward_model/libero_spatial_rlpd_sparse_600k_s0/task02
examples/libero/rlpd/outputs/reward_model/libero_spatial_rlpd_sparse_600k_s0/task03
examples/libero/rlpd/outputs/reward_model/libero_spatial_rlpd_sparse_600k_s0/task04
examples/libero/rlpd/outputs/reward_model/libero_spatial_rlpd_sparse_600k_s0/task05
examples/libero/rlpd/outputs/reward_model/libero_spatial_rlpd_sparse_600k_s0/task06
examples/libero/rlpd/outputs/reward_model/libero_spatial_rlpd_sparse_600k_s0/task07
```

These runs had already reached `600000` actor env steps before migration.

## Runs not moved yet

The following runs were not moved because they were still actively writing logs on 239 at the time of this migration:

```text
outputs/reward_model/239_localenv/libero_spatial_task8_rlpd_sparse
outputs/reward_model/239_localenv/libero_spatial_task9_rlpd_sparse
```

At the check time, task8 and task9 still had live learner, actor, and async-eval worker processes. Their latest observed progress was approximately:

| run | env_steps | latest eval |
|---|---:|---:|
| task8 | 146636 / 600000 | 0.0 @ ep600 |
| task9 | 144561 / 600000 | 0.0 @ ep600 |

## Finish migration after task8/task9 complete

After task8 and task9 finish, run this from the repository root:

```bash
cd /vla/users/niejunnan/codebase/VLA-RL

for i in 8 9; do
  src="outputs/reward_model/239_localenv/libero_spatial_task${i}_rlpd_sparse"
  dst="examples/libero/rlpd/outputs/reward_model/libero_spatial_rlpd_sparse_600k_s0/task$(printf "%02d" "$i")"

  if pgrep -af "${src}|libero_spatial_task${i}_rlpd_sparse" | grep -E "train_rlpd.py|examples.libero.common.eval_queue" >/dev/null; then
    echo "SKIP task${i}: run is still active"
    continue
  fi

  if [ ! -d "$src" ]; then
    echo "SKIP task${i}: source missing: $src"
    continue
  fi

  if [ -e "$dst" ]; then
    echo "SKIP task${i}: destination exists: $dst"
    continue
  fi

  mkdir -p "$(dirname "$dst")"
  mv "$src" "$dst"
  echo "moved task${i}: $src -> $dst"
done
```

Verify the final layout with:

```bash
find examples/libero/rlpd/outputs/reward_model/libero_spatial_rlpd_sparse_600k_s0 -maxdepth 1 -type d -name 'task*' | sort
```

## Notes

Historical files inside migrated directories, such as `config.yaml`, `eval_summary.jsonl`, and `eval_queue.jsonl`, may still contain the original absolute `outputs/reward_model/239_localenv/...` paths because they record how the run was launched. The directory migration intentionally does not rewrite those historical records.
