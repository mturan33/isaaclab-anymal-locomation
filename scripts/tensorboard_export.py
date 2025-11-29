"""
TensorBoard Log Export and Comparison Script for PPO Training Analysis.

Exports training curves from RSL-RL and Custom PPO implementations,
generates comparison plots for documentation and presentations.

Usage:
    python tensorboard_export.py --rsl_rl_log <path> --custom_ppo_log <path> --output <dir>
"""

import argparse
import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("TensorBoard not found. Install with: pip install tensorboard")
    exit(1)


CUSTOM_PPO_TAG_MAPPING = {
    'Train/mean_episode_reward': 'Train/mean_reward',
    'Train/action_std': 'Policy/mean_noise_std',
    'Loss/actor': 'Loss/surrogate',
    'Loss/critic': 'Loss/value_function',
    'Perf/steps_per_sec': 'Perf/total_fps',
}


def load_tensorboard_scalars(log_dir, tag_mapping=None):
    """Load scalar data from TensorBoard logs."""
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    available_tags = ea.Tags().get('scalars', [])
    print(f"[INFO] Found {len(available_tags)} scalar tags in {log_dir}")

    data = {}
    for tag in available_tags:
        try:
            events = ea.Scalars(tag)
            df = pd.DataFrame([
                {'step': e.step, 'value': e.value, 'wall_time': e.wall_time}
                for e in events
            ])
            mapped_tag = tag_mapping.get(tag, tag) if tag_mapping else tag
            data[mapped_tag] = df
        except Exception as e:
            print(f"[WARN] Could not load tag {tag}: {e}")

    return data


def plot_comparison(rsl_data, custom_data, tag, output_dir, max_iter=None):
    """Plot comparison of a single metric."""
    fig, ax = plt.subplots(figsize=(12, 6))

    if tag in rsl_data:
        df = rsl_data[tag]
        if max_iter:
            df = df[df['step'] <= max_iter]
        ax.plot(df['step'], df['value'], label='RSL-RL PPO', color='#FF8C00', linewidth=2, alpha=0.9)

    if tag in custom_data:
        df = custom_data[tag]
        if max_iter:
            df = df[df['step'] <= max_iter]
        ax.plot(df['step'], df['value'], label='My PPO', color='#32CD32', linewidth=2, alpha=0.9)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel(tag.replace('/', ' - '), fontsize=12)
    ax.set_title(f'Comparison: {tag}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    ax.set_facecolor('#1e1e1e')
    fig.patch.set_facecolor('#1e1e1e')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_color('white')

    safe_tag = tag.replace('/', '_').replace(' ', '_')
    output_path = os.path.join(output_dir, f'{safe_tag}_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1e1e1e')
    plt.close()
    print(f"[SAVED] {output_path}")


def create_summary_plot(rsl_data, custom_data, output_dir, max_iter=None):
    """Create a summary plot with multiple metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('#1e1e1e')

    metrics = [
        ('Train/mean_reward', 'Mean Reward'),
        ('Policy/mean_noise_std', 'Action STD'),
        ('Loss/surrogate', 'Surrogate Loss'),
        ('Train/mean_episode_length', 'Episode Length'),
    ]

    for ax, (tag, title) in zip(axes.flat, metrics):
        ax.set_facecolor('#1e1e1e')

        if tag in rsl_data:
            df = rsl_data[tag]
            if max_iter:
                df = df[df['step'] <= max_iter]
            ax.plot(df['step'], df['value'], label='RSL-RL', color='#FF8C00', linewidth=2, alpha=0.9)

        if tag in custom_data:
            df = custom_data[tag]
            if max_iter:
                df = df[df['step'] <= max_iter]
            ax.plot(df['step'], df['value'], label='My PPO', color='#32CD32', linewidth=2, alpha=0.9)

        ax.set_title(title, fontsize=12, fontweight='bold', color='white')
        ax.set_xlabel('Iteration', fontsize=10, color='white')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')

    plt.suptitle('PPO Comparison: My Implementation vs RSL-RL', fontsize=16, fontweight='bold', color='white', y=1.02)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'summary_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1e1e1e')
    plt.close()
    print(f"[SAVED] {output_path}")


def create_reward_plot(rsl_data, custom_data, output_dir, max_iter, filename):
    """Create a professional reward comparison plot."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    tag = 'Train/mean_reward'
    rsl_final, custom_final = 0, 0

    if tag in rsl_data:
        df = rsl_data[tag]
        if max_iter:
            df = df[df['step'] <= max_iter]
        ax.plot(df['step'], df['value'], label='RSL-RL PPO', color='#FF6B35', linewidth=3, alpha=0.9)
        rsl_final = df['value'].iloc[-1] if len(df) > 0 else 0

    if tag in custom_data:
        df = custom_data[tag]
        if max_iter:
            df = df[df['step'] <= max_iter]
        ax.plot(df['step'], df['value'], label='My PPO (From Scratch)', color='#004E89', linewidth=3, alpha=0.9)
        custom_final = df['value'].iloc[-1] if len(df) > 0 else 0

    ax.set_xlabel('Training Iteration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Episode Reward', fontsize=14, fontweight='bold')
    ax.set_title(f'From-Scratch PPO vs RSL-RL ({max_iter} iterations)\nIsaac Lab Anymal-C Quadruped', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    if rsl_final > 0 and custom_final > 0:
        ratio = (custom_final / rsl_final) * 100
        y_min, y_max = ax.get_ylim()
        ax.annotate(f'{ratio:.0f}% Performance Match!',
                    xy=(max_iter * 0.05, y_min + (y_max - y_min) * 0.1),
                    fontsize=14, fontweight='bold', color='#2E7D32',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9', edgecolor='#2E7D32'))

    plt.tight_layout()
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[SAVED] {output_path}")


def print_final_stats(rsl_data, custom_data):
    """Print final statistics comparison."""
    print("\n" + "=" * 60)
    print("FINAL STATISTICS COMPARISON")
    print("=" * 60)

    metrics = [
        ('Train/mean_reward', 'Mean Reward'),
        ('Policy/mean_noise_std', 'Action STD'),
        ('Train/mean_episode_length', 'Episode Length'),
    ]

    print(f"{'Metric':<25} {'RSL-RL':>12} {'My PPO':>12} {'Ratio':>10}")
    print("-" * 60)

    for tag, name in metrics:
        rsl_val = rsl_data[tag]['value'].iloc[-1] if tag in rsl_data else 0
        custom_val = custom_data[tag]['value'].iloc[-1] if tag in custom_data else 0
        ratio = (custom_val / rsl_val * 100) if rsl_val != 0 else 0
        print(f"{name:<25} {rsl_val:>12.2f} {custom_val:>12.2f} {ratio:>9.1f}%")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="TensorBoard Log Comparison")
    parser.add_argument("--rsl_rl_log", type=str, required=True)
    parser.add_argument("--custom_ppo_log", type=str, required=True)
    parser.add_argument("--output", type=str, default="comparison_plots")
    parser.add_argument("--max_iter", type=int, default=None)

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    print(f"\n[INFO] Loading RSL-RL logs from: {args.rsl_rl_log}")
    rsl_data = load_tensorboard_scalars(args.rsl_rl_log)

    print(f"\n[INFO] Loading Custom PPO logs from: {args.custom_ppo_log}")
    custom_data = load_tensorboard_scalars(args.custom_ppo_log, tag_mapping=CUSTOM_PPO_TAG_MAPPING)

    print(f"\n[INFO] Creating comparison plots in: {args.output}")

    create_summary_plot(rsl_data, custom_data, args.output, args.max_iter)
    create_reward_plot(rsl_data, custom_data, args.output, 1000, 'reward_comparison_1k.png')

    if args.max_iter and args.max_iter > 1000:
        create_reward_plot(rsl_data, custom_data, args.output, args.max_iter, f'reward_comparison_{args.max_iter}.png')

    for tag in ['Train/mean_reward', 'Policy/mean_noise_std', 'Loss/entropy', 'Train/mean_episode_length']:
        if tag in rsl_data or tag in custom_data:
            plot_comparison(rsl_data, custom_data, tag, args.output, args.max_iter)

    print_final_stats(rsl_data, custom_data)
    print(f"\n[DONE] All plots saved to: {args.output}")


if __name__ == "__main__":
    main()