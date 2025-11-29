# tensorboard_export.py
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib.pyplot as plt

# RSL-RL logunu yükle
rsl_ea = event_accumulator.EventAccumulator('logs/rsl_rl/anymal_c_flat_direct/2025-11-27_11-57-46')
rsl_ea.Reload()

# Custom PPO logunu yükle
custom_ea = event_accumulator.EventAccumulator('logs/rsl_rl/custom_ppo_v2/2025-11-28_20-39-37')
custom_ea.Reload()

# Mean reward verilerini al
rsl_reward = pd.DataFrame(rsl_ea.Scalars('Train/mean_episode_reward'))
custom_reward = pd.DataFrame(custom_ea.Scalars('Train/mean_episode_reward'))

# Grafik çiz
plt.figure(figsize=(12, 6))
plt.plot(rsl_reward['step'], rsl_reward['value'], label='RSL-RL PPO', color='orange')
plt.plot(custom_reward['step'], custom_reward['value'], label='Custom PPO', color='green')
plt.xlabel('Iteration')
plt.ylabel('Mean Episode Reward')
plt.title('PPO Comparison: Custom vs RSL-RL')
plt.legend()
plt.grid(True)
plt.savefig('ppo_comparison.png', dpi=150)
plt.show()