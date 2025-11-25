# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Anymal C quadruped robot."""

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG

##
# Custom Anymal C Configuration for PPO Training
##

ANYMAL_C_CUSTOM_CFG = ANYMAL_C_CFG.copy()
ANYMAL_C_CUSTOM_CFG.prim_path = "/World/envs/env_.*/Robot"

# Customize actuators for better control
ANYMAL_C_CUSTOM_CFG.actuators = {
    "legs": ImplicitActuatorCfg(
        joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
        effort_limit=80.0,
        velocity_limit=7.5,
        stiffness=80.0,
        damping=2.0,
    ),
}

# Optional: Add spawn configuration customizations
# ANYMAL_C_CUSTOM_CFG.spawn.rigid_props.disable_gravity = False
# ANYMAL_C_CUSTOM_CFG.spawn.articulation_props.enabled_self_collisions = False
