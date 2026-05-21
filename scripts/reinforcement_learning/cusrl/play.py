# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Script to play a checkpoint if an RL agent from CusRL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate an RL agent with CusRL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="cusrl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to load for playing.")
parser.add_argument(
    "--stochastic",
    action="store_true",
    default=False,
    help="Whether to run the agent in stochastic mode.",
)
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument(
    "--debug_obs",
    action="store_true",
    default=False,
    help="Print policy observation config/runtime stats (including height_scan) before playing.",
)
parser.add_argument(
    "--debug_obs_interval",
    type=int,
    default=100,
    help="Print runtime height_scan stats every N play steps when --debug_obs is enabled.",
)
parser.add_argument(
    "--debug_obs_values",
    action="store_true",
    default=False,
    help="Print the full flattened policy height_scan values when --debug_obs is enabled.",
)
parser.add_argument(
    "--debug_obs_grid",
    action="store_true",
    default=False,
    help="Print the policy height_scan values reshaped as an 11x17 grid when --debug_obs is enabled.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import cusrl
import gymnasium as gym
import torch
from cusrl.environment.isaaclab import TrainerCfg

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,  # noqa: F401
    DirectRLEnvCfg,  # noqa: F401
    ManagerBasedRLEnvCfg,  # noqa: F401
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.dict import print_dict

from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: F401

import robot_lab.tasks  # noqa: F401  # isort: skip

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl_utils import camera_follow


class CameraFollowPlayerHook(cusrl.Player.Hook):
    def step(self, step: int, transition: dict, metrics: dict):
        camera_follow(self.player.environment)


class HeightScanDebugPlayerHook(cusrl.Player.Hook):
    def __init__(self, interval: int = 100, print_values: bool = False, print_grid: bool = False):
        self.interval = max(1, int(interval))
        self.print_values = print_values
        self.print_grid = print_grid
        self._obs_mgr = None
        self._height_scan_start = None
        self._height_scan_size = None

    def init(self, player):
        super().init(player)
        self._obs_mgr = player.environment.unwrapped.observation_manager
        term_names = self._obs_mgr.active_terms.get("policy", [])
        if not self._obs_mgr.group_obs_concatenate.get("policy", False) or "height_scan" not in term_names:
            return

        term_dims = self._obs_mgr.group_obs_term_dim["policy"]
        hs_idx = term_names.index("height_scan")
        self._height_scan_start = sum(int(dim[-1]) for dim in term_dims[:hs_idx])
        self._height_scan_size = int(term_dims[hs_idx][-1])

    def step(self, step: int, transition: dict, metrics: dict):
        if step % self.interval != 0:
            return
        if self._height_scan_start is None or self._height_scan_size is None:
            return

        sensor = self.player.environment.unwrapped.scene.sensors["height_scanner"]
        sensor_z = sensor.data.pos_w[:, 2]
        hit_z = sensor.data.ray_hits_w[..., 2]
        raw_height_scan = sensor_z.unsqueeze(1) - hit_z - 0.5
        print(
            "[OBS-DEBUG] raw sensor_z shape/min/max/mean:",
            tuple(sensor_z.shape),
            float(sensor_z.min().item()),
            float(sensor_z.max().item()),
            float(sensor_z.mean().item()),
        )
        print(
            "[OBS-DEBUG] raw hit_z shape/min/max/mean/std:",
            tuple(hit_z.shape),
            float(hit_z.min().item()),
            float(hit_z.max().item()),
            float(hit_z.mean().item()),
            float(hit_z.std().item()),
        )
        print(
            "[OBS-DEBUG] raw height_scan=sensor_z-hit_z-0.5 shape/min/max/mean/std:",
            tuple(raw_height_scan.shape),
            float(raw_height_scan.min().item()),
            float(raw_height_scan.max().item()),
            float(raw_height_scan.mean().item()),
            float(raw_height_scan.std().item()),
        )

        policy_obs = transition.get("observation")
        if not isinstance(policy_obs, torch.Tensor):
            return

        hs = policy_obs[..., self._height_scan_start : self._height_scan_start + self._height_scan_size]
        print(
            "[OBS-DEBUG] height_scan slice shape/min/max/mean/std:",
            tuple(hs.shape),
            float(hs.min().item()),
            float(hs.max().item()),
            float(hs.mean().item()),
            float(hs.std().item()),
        )
        _print_height_scan_values(hs, self.print_values, self.print_grid)


def _format_float_list(values: torch.Tensor) -> str:
    return "[" + ", ".join(f"{float(value):.6f}" for value in values.detach().cpu().flatten()) + "]"


def _print_height_scan_values(height_scan: torch.Tensor, print_values: bool, print_grid: bool):
    if not print_values and not print_grid:
        return

    first_env_hs = height_scan[0].detach().cpu().flatten()
    if print_values:
        print("[OBS-DEBUG] height_scan flat values:")
        print(_format_float_list(first_env_hs))
    if print_grid:
        if first_env_hs.numel() != 187:
            print(f"[OBS-DEBUG] height_scan grid skipped: expected 187 values, got {first_env_hs.numel()}")
            return
        print("[OBS-DEBUG] height_scan grid values (11 rows x 17 cols, x changes fastest):")
        for row in first_env_hs.reshape(11, 17):
            print(_format_float_list(row))


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: TrainerCfg):
    """Play with CusRL-RL agent."""
    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    cusrl.set_global_seed(args_cli.seed)

    # modify environment configurations based on CLI args
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 50
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # spawn the robot randomly in the grid (instead of their terrain levels)
    env_cfg.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    env_cfg.events.randomize_apply_external_force_torque = None
    # NOTE: the event term in this task is named `randomize_push_robot`, not `push_robot`.
    if hasattr(env_cfg.events, "randomize_push_robot"):
        env_cfg.events.randomize_push_robot = None
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.command_levels_ang_vel = None

    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        config = Se2KeyboardCfg(
            v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
            v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
            omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
        )
        controller = Se2Keyboard(config)
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device),
        )

    if args_cli.checkpoint is None:
        args_cli.checkpoint = os.path.join("logs", "cusrl", agent_cfg.experiment_name)
    trial = cusrl.Trial(args_cli.checkpoint)
    if trial is not None:
        log_dir = trial.home
    else:
        # specify directory for logging videos
        log_dir = os.path.join("logs", "cusrl", agent_cfg.experiment_name)
        log_dir = os.path.abspath(log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.debug_obs:
        print("[OBS-DEBUG] policy.height_scan is None:", env_cfg.observations.policy.height_scan is None)
        if env_cfg.scene.height_scanner is not None:
            print("[OBS-DEBUG] height_scanner.prim_path:", env_cfg.scene.height_scanner.prim_path)
            print("[OBS-DEBUG] height_scanner.mesh_prim_paths:", env_cfg.scene.height_scanner.mesh_prim_paths)
        else:
            print("[OBS-DEBUG] height_scanner is None")

        obs_mgr = env.unwrapped.observation_manager
        if "policy" in obs_mgr.active_terms:
            print("[OBS-DEBUG] policy terms:", obs_mgr.active_terms["policy"])
            print("[OBS-DEBUG] policy term dims:", obs_mgr.group_obs_term_dim["policy"])
            print("[OBS-DEBUG] policy concatenated dim:", obs_mgr.group_obs_dim["policy"])

            reset_out = env.reset()
            # Gymnasium reset returns (obs, info)
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            policy_obs = obs.get("policy", obs) if isinstance(obs, dict) else obs

            if isinstance(policy_obs, torch.Tensor):
                print(
                    "[OBS-DEBUG] policy obs shape/min/max:",
                    tuple(policy_obs.shape),
                    float(policy_obs.min().item()),
                    float(policy_obs.max().item()),
                )

                # Slice out height_scan from concatenated policy obs for quick sanity check.
                if (
                    obs_mgr.group_obs_concatenate.get("policy", False)
                    and "height_scan" in obs_mgr.active_terms["policy"]
                ):
                    term_names = obs_mgr.active_terms["policy"]
                    term_dims = obs_mgr.group_obs_term_dim["policy"]
                    hs_idx = term_names.index("height_scan")
                    hs_start = sum(int(d[-1]) for d in term_dims[:hs_idx])
                    hs_size = int(term_dims[hs_idx][-1])
                    hs = policy_obs[..., hs_start : hs_start + hs_size]
                    print(
                        "[OBS-DEBUG] height_scan slice shape/min/max/mean/std:",
                        tuple(hs.shape),
                        float(hs.min().item()),
                        float(hs.max().item()),
                        float(hs.mean().item()),
                        float(hs.std().item()),
                    )
                    _print_height_scan_values(hs, args_cli.debug_obs_values, args_cli.debug_obs_grid)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # create player from cusrl
    player = cusrl.Player(
        environment=cusrl.environment.IsaacLabEnvAdapter(env),
        agent=agent_cfg.agent_factory.override(device=args_cli.device),
        checkpoint_path=trial,
        deterministic=not args_cli.stochastic,
    )

    export_model_dir = os.path.join(log_dir, "exported")
    player.agent.export(output_dir=export_model_dir, target_format="onnx", verbose=args_cli.verbose)
    player.agent.export(output_dir=export_model_dir, target_format="jit", verbose=args_cli.verbose)

    if args_cli.keyboard:
        player.register_hook(CameraFollowPlayerHook())
    if args_cli.debug_obs:
        player.register_hook(
            HeightScanDebugPlayerHook(
                interval=args_cli.debug_obs_interval,
                print_values=args_cli.debug_obs_values,
                print_grid=args_cli.debug_obs_grid,
            )
        )

    # run playing loop
    player.run_playing_loop()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
