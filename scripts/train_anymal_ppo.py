#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train Anymal C quadruped with PPO using RSL-RL.

Usage:
    # Train on flat terrain
    python train_anymal_ppo.py --terrain flat

    # Train on rough terrain
    python train_anymal_ppo.py --terrain rough

    # Train with custom number of environments
    python train_anymal_ppo.py --terrain flat --num_envs 128

    # Continue training from checkpoint
    python train_anymal_ppo.py --terrain flat --resume
"""

import argparse
import os
import sys


def main():
    """Main function to launch PPO training for Anymal C."""
    parser = argparse.ArgumentParser(description="Train Anymal C with PPO")
    parser.add_argument(
        "--terrain",
        type=str,
        default="flat",
        choices=["flat", "rough"],
        help="Terrain type: flat or rough (default: flat)",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Number of parallel environments (default: use config default)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no GUI)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file to load",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # Construct task name
    task_name = f"Isaac-MyAnymal-{args.terrain.capitalize()}-v0"

    # Build command for Isaac Lab RSL-RL training
    print(f"\n{'='*60}")
    print(f"  Training Anymal C with PPO")
    print(f"{'='*60}")
    print(f"  Task: {task_name}")
    print(f"  Terrain: {args.terrain}")
    if args.num_envs:
        print(f"  Num Envs: {args.num_envs}")
    print(f"  Headless: {args.headless}")
    print(f"  Resume: {args.resume}")
    print(f"  Seed: {args.seed}")
    print(f"{'='*60}\n")

    # Try to import and run directly
    try:
        from isaaclab_rl.rsl_rl.train import main as train_main
        import sys

        # Build arguments for the training script
        train_args = [
            "--task", task_name,
            "--seed", str(args.seed),
        ]

        if args.num_envs:
            train_args.extend(["--num_envs", str(args.num_envs)])

        if args.headless:
            train_args.append("--headless")

        if args.resume:
            train_args.append("--resume")

        if args.checkpoint:
            train_args.extend(["--load_run", args.checkpoint])

        # Override sys.argv for the training script
        sys.argv = ["train.py"] + train_args

        # Launch training
        train_main()

    except ImportError as e:
        print(f"Error: Could not import Isaac Lab training module.")
        print(f"Details: {e}")
        print("\nPlease ensure Isaac Lab is properly installed.")
        print("\nAlternatively, run training using Isaac Lab's training script:")
        print(f"  python <ISAAC_LAB_PATH>/source/isaaclab_rl/rsl_rl/train.py --task {task_name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
