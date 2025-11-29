"""
Keyboard Control Script for Anymal-C Quadruped Robot.

Provides interactive keyboard control for trained PPO models with
real-time velocity command visualization and smooth ramping.

Usage:
    ./isaaclab.bat -p scripts/play_keyboard.py --task <TASK> --checkpoint <PATH>

Examples:
    # My PPO
    ./isaaclab.bat -p scripts/play_keyboard.py --task Isaac-MyAnymal-Flat-v0 \
        --checkpoint logs/rsl_rl/custom_ppo_v2/2025-11-28_20-39-37/model_best.pt

    # RSL-RL
    ./isaaclab.bat -p scripts/play_keyboard.py --task Isaac-Velocity-Flat-Anymal-C-Direct-v0 \
        --checkpoint logs/rsl_rl/anymal_c_flat_direct/2025-11-28_12-15-24/model_9999.pt
"""

import argparse
import math
import torch
import torch.nn as nn
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Keyboard control for quadruped robots")
parser.add_argument("--task", type=str, default="Isaac-MyAnymal-Flat-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.num_envs = 1
args_cli.disable_fabric = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import omni.appwindow
import gymnasium as gym
import isaaclab_tasks


class EmpiricalNormalization(nn.Module):
    """Online observation normalization using running statistics."""

    def __init__(self, input_shape: tuple, epsilon: float = 1e-8):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_var", torch.ones(input_shape))
        self.register_buffer("count", torch.tensor(epsilon))
        self.epsilon = epsilon

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network compatible with both custom PPO and RSL-RL checkpoints."""

    def __init__(self, num_obs: int, num_actions: int, hidden_dims: list = [256, 256, 256]):
        super().__init__()

        actor_layers = []
        prev_dim = num_obs
        for dim in hidden_dims:
            actor_layers.extend([nn.Linear(prev_dim, dim), nn.ELU()])
            prev_dim = dim
        actor_layers.append(nn.Linear(prev_dim, num_actions))
        self.actor = nn.Sequential(*actor_layers)

        critic_layers = []
        prev_dim = num_obs
        for dim in hidden_dims:
            critic_layers.extend([nn.Linear(prev_dim, dim), nn.ELU()])
            prev_dim = dim
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        self.log_std = nn.Parameter(torch.zeros(num_actions))

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


class QuadrupedKeyboardController:
    """Keyboard controller with smooth velocity ramping."""

    def __init__(self):
        self._target_command = np.array([0.0, 0.0, 0.0])
        self._current_command = np.array([0.0, 0.0, 0.0])

        self._max_lin_vel = 0.5
        self._max_ang_vel = 0.5
        self._ramp_rate = 0.05

        self._input_keyboard_mapping = {
            "W": [self._max_lin_vel, 0.0, 0.0],
            "UP": [self._max_lin_vel, 0.0, 0.0],
            "NUMPAD_8": [self._max_lin_vel, 0.0, 0.0],
            "S": [-self._max_lin_vel, 0.0, 0.0],
            "DOWN": [-self._max_lin_vel, 0.0, 0.0],
            "NUMPAD_2": [-self._max_lin_vel, 0.0, 0.0],
            "A": [0.0, self._max_lin_vel, 0.0],
            "LEFT": [0.0, self._max_lin_vel, 0.0],
            "NUMPAD_4": [0.0, self._max_lin_vel, 0.0],
            "D": [0.0, -self._max_lin_vel, 0.0],
            "RIGHT": [0.0, -self._max_lin_vel, 0.0],
            "NUMPAD_6": [0.0, -self._max_lin_vel, 0.0],
            "Q": [0.0, 0.0, self._max_ang_vel],
            "N": [0.0, 0.0, self._max_ang_vel],
            "NUMPAD_7": [0.0, 0.0, self._max_ang_vel],
            "E": [0.0, 0.0, -self._max_ang_vel],
            "M": [0.0, 0.0, -self._max_ang_vel],
            "NUMPAD_9": [0.0, 0.0, -self._max_ang_vel],
        }

        self._active_keys = set()
        self._reset_requested = False
        self._quit_requested = False

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_keyboard_event
        )

        print("[KEYBOARD] Controller initialized")

    def _on_keyboard_event(self, event, *args, **kwargs) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key_name = event.input.name
            if key_name in self._input_keyboard_mapping:
                self._active_keys.add(key_name)
            elif key_name == "R":
                self._reset_requested = True
            elif key_name == "ESCAPE":
                self._quit_requested = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            key_name = event.input.name
            self._active_keys.discard(key_name)
        return True

    def get_command(self, device) -> torch.Tensor:
        self._target_command = np.array([0.0, 0.0, 0.0])
        for key in self._active_keys:
            if key in self._input_keyboard_mapping:
                self._target_command += np.array(self._input_keyboard_mapping[key])

        self._target_command = np.clip(
            self._target_command,
            [-self._max_lin_vel, -self._max_lin_vel, -self._max_ang_vel],
            [self._max_lin_vel, self._max_lin_vel, self._max_ang_vel]
        )

        diff = self._target_command - self._current_command
        self._current_command += np.clip(diff, -self._ramp_rate, self._ramp_rate)

        return torch.tensor([self._current_command], device=device, dtype=torch.float32)

    def reset_command(self):
        self._target_command = np.array([0.0, 0.0, 0.0])
        self._current_command = np.array([0.0, 0.0, 0.0])
        self._active_keys.clear()

    @property
    def reset_requested(self) -> bool:
        flag = self._reset_requested
        self._reset_requested = False
        return flag

    @property
    def quit_requested(self) -> bool:
        return self._quit_requested

    def print_status(self):
        vx, vy, yaw = self._current_command
        tx, ty, tyaw = self._target_command
        print(f"\r[CMD] Vx: {vx:+.2f}/{tx:+.2f} | Vy: {vy:+.2f}/{ty:+.2f} | Yaw: {yaw:+.2f}/{tyaw:+.2f}    ", end="", flush=True)


def main():
    env = gym.make(args_cli.task, cfg=None, render_mode="rgb_array")
    unwrapped_env = env.unwrapped

    num_obs = unwrapped_env.observation_space["policy"].shape[1]
    num_actions = unwrapped_env.action_space.shape[1]
    device = unwrapped_env.device

    print(f"[INFO] Observation dim: {num_obs}")
    print(f"[INFO] Action dim: {num_actions}")
    print(f"[INFO] Device: {device}")

    print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
    checkpoint = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)

    has_obs_normalizer = "obs_normalizer" in checkpoint

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    first_layer_key = None
    for key in state_dict.keys():
        if "actor" in key and "weight" in key:
            first_layer_key = key
            break

    if first_layer_key:
        first_layer_shape = state_dict[first_layer_key].shape
        input_dim = first_layer_shape[1]

        hidden_dims = []
        for i in range(10):
            key = f"actor.{i*2}.weight"
            if key in state_dict:
                hidden_dims.append(state_dict[key].shape[0])

        if hidden_dims:
            hidden_dims = hidden_dims[:-1]

        if not hidden_dims:
            hidden_dims = [128, 128, 128]

        print(f"[INFO] Detected hidden dims: {hidden_dims}")
    else:
        hidden_dims = [128, 128, 128]

    actor_critic = ActorCriticNetwork(num_obs, num_actions, hidden_dims).to(device)
    actor_critic.load_state_dict(state_dict, strict=False)
    actor_critic.eval()
    print("[INFO] Model loaded successfully!")

    obs_normalizer = None
    if has_obs_normalizer:
        obs_normalizer = EmpiricalNormalization((num_obs,)).to(device)
        obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])
        print("[INFO] Observation normalizer loaded!")
        print(f"       Running mean range: [{obs_normalizer.running_mean.min():.3f}, {obs_normalizer.running_mean.max():.3f}]")
        print(f"       Running var range:  [{obs_normalizer.running_var.min():.3f}, {obs_normalizer.running_var.max():.3f}]")
    else:
        print("[INFO] No observation normalizer in checkpoint (RSL-RL style)")

    keyboard_ctrl = QuadrupedKeyboardController()

    print("\n" + "=" * 60)
    print("            KEYBOARD CONTROL ACTIVE")
    print("=" * 60)
    print("  Movement:")
    print("    W / UP / NUM8       - Forward")
    print("    S / DOWN / NUM2     - Backward")
    print("    A / LEFT / NUM4     - Strafe Left")
    print("    D / RIGHT / NUM6    - Strafe Right")
    print("  Rotation:")
    print("    Q / N / NUM7        - Turn Left")
    print("    E / M / NUM9        - Turn Right")
    print("  Control:")
    print("    R                   - Reset Robot")
    print("    ESC                 - Quit")
    print("=" * 60 + "\n")

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

    step_count = 0

    while simulation_app.is_running() and not keyboard_ctrl.quit_requested:
        cmd = keyboard_ctrl.get_command(device)

        try:
            if hasattr(unwrapped_env, "_commands"):
                unwrapped_env._commands[:] = cmd
        except Exception:
            pass

        with torch.no_grad():
            if obs_normalizer is not None:
                obs_norm = obs_normalizer.normalize(obs)
            else:
                obs_norm = obs
            actions = actor_critic.act_inference(obs_norm)

        obs_dict, rewards, terminated, truncated, info = env.step(actions)
        dones = terminated | truncated
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

        if keyboard_ctrl.reset_requested or dones.any():
            obs_dict, _ = env.reset()
            obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
            keyboard_ctrl.reset_command()
            print("\n[RESET] Robot reset")

        if step_count % 5 == 0:
            keyboard_ctrl.print_status()

        step_count += 1

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()