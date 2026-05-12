# Go2W CusRL RSL-Aligned Notes

## What Changed

Added a new CusRL trainer config for Go2W:

- `UnitreeGo2WRoughTrainerRslAlignedCfg`
- `UnitreeGo2WFlatTrainerRslAlignedCfg`

Files:

- `agents/cusrl_ppo_cfg.py`
- `__init__.py`

Main changes in the new rough config:

- actor/critic backbone: `Lstm(2x256)` -> `Mlp([512, 256, 128], ELU)`
- optimizer: `AdamW` -> `Adam`
- entropy weight: `0.008` -> `0.01`

Kept unchanged:

- `num_steps_per_update=24`
- `num_epochs=5`
- `num_mini_batches=4`
- `lr=1e-3`
- `gamma=0.99`
- `lamda=0.95`
- `desired_kl_divergence=0.01`
- `max_grad_norm=1.0`
- `max_iterations=20000`

## Why

The successful Go2W `rslrl` config uses:

- MLP policy
- hidden dims `[512, 256, 128]`
- ELU activation
- entropy `0.01`

The previous Go2W `cusrl` config was a special case using a 2-layer LSTM, while nearby successful locomotion configs are mostly MLP-based. This new config is meant to isolate algorithm differences from architecture differences.

## How To Use

Train with the new config:

```bash
python scripts/reinforcement_learning/cusrl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 \
  --agent cusrl_rsl_aligned_cfg_entry_point \
  --headless
```

## Why `--agent` Works

`cusrl/train.py` defines:

- `--agent`, defaulting to `cusrl_cfg_entry_point`

Then it passes that string into:

- `@hydra_task_config(args_cli.task, args_cli.agent)`

The task registry for Go2W now contains:

- `cusrl_cfg_entry_point`
- `cusrl_rsl_aligned_cfg_entry_point`

So `--agent cusrl_rsl_aligned_cfg_entry_point` tells the loader to fetch the new trainer config instead of the default one.

## If `--agent` Is Not Given

The script uses the default:

```text
cusrl_cfg_entry_point
```

That means it will still load the original Go2W CusRL config, not the new RSL-aligned one.

4.6 Modify
  - lin_vel_z_l2: -2.0 -> -0.8
  - joint_power: -2e-5 -> -8e-6
  - joint_mirror: -0.05 -> -0.01
  - action_rate_l2: -0.01 -> -0.003
  - contact_forces: -1.5e-4 -> -7e-5

## Radar Observation + StepIt Alignment

- `rough_env_cfg.py` now anchors `scene.height_scanner` to `Robot/radar` instead of `Robot/base`.
- `go2w_description.urdf` sets `radar_joint` as `dont_collapse="true"` so the `radar` frame is preserved when fixed joints are merged.
- `scene.height_scanner_base` remains on `Robot/base` for base-height related rewards.

StepIt alignment note:

- StepIt does not consume a URDF radar link directly; it consumes terrain perception as `heightmap` field (`policy_neuro_ros/heightmap_subscriber` from `grid_map_msgs/GridMap`).
- For a perception-enabled actor exported from this env, append `heightmap` in StepIt `actor.yml` observation fields after proprioceptive terms, with size matching the training `height_scan` sample count.
