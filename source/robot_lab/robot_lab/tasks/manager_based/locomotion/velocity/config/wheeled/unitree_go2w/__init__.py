# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Unitree-Go2W-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2WFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2WFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2WFlatTrainerCfg",
        "cusrl_rsl_aligned_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2WFlatTrainerRslAlignedCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2WRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2WRoughPPORunnerCfg",
        "rsl_rl_heightmap_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2WHeightMapEncoderPPORunnerCfg"
        ),
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2WRoughTrainerCfg",
        "cusrl_rsl_aligned_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2WRoughTrainerRslAlignedCfg",
        # 新 CusRL 高程图编码入口：不覆盖默认 cusrl 配置，方便单独跑对照实验。
        "cusrl_heightmap_cfg_entry_point": (
            f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2WRoughTrainerHeightMapEncoderCfg"
        ),
    },
)
