# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""My Anymal quadruped locomotion environment."""

import gymnasium as gym
from . import agents
from .my_anymal_env_cfg import MyAnymalFlatEnvCfg, MyAnymalRoughEnvCfg

##
# Register Gym environments
##

gym.register(
    id="Isaac-MyAnymal-Flat-v0",
    entry_point="isaaclab_tasks.direct.my_anymal_quadruped.my_anymal_env:MyAnymalEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MyAnymalFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyAnymalFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-MyAnymal-Rough-v0",
    entry_point="isaaclab_tasks.direct.my_anymal_quadruped.my_anymal_env:MyAnymalEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MyAnymalRoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyAnymalRoughPPORunnerCfg",
    },
)