"""
STEP 4 — Algorithm Comparison: PPO vs A2C on CartPole-v1
=========================================================
Portfolio goal: Benchmark two algorithms on the same environment —
the kind of output that signals ML engineering maturity.

Run:
    python step4_algorithm_comparison.py

Outputs:
    logs/compare_ppo/              ← TensorBoard logs
    logs/compare_a2c/
    results/algorithm_comparison.png  ← Best portfolio chart
"""

import os
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.results_plotter import load_results, ts2xy

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

TIMESTEPS = 100_000
SEED      = 42

def train_and_collect(AlgoClass, algo_name, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    env   = Monitor(gym.make("CartPole-v1"), log_dir)
    model = AlgoClass("MlpPolicy", env, verbose=0, seed=SEED)
    print(f"  Training {algo_name} for {TIMESTEPS:,} timesteps...")
    model.learn(total_timesteps=TIMESTEPS)

    eval_env = gym.make("CartPole-v1")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
    eval_env.close()
    env.close()

    x, y = ts2xy(load_results(log_dir), "timesteps")
    return x, y, mean_reward, std_reward

configs = [
    (PPO, "PPO", "logs/compare_ppo", "steelblue"),
    (A2C, "A2C", "logs/compare_a2c", "darkorange"),
]

print(f"\n🔬 Benchmarking PPO vs A2C on CartPole-v1\n")

results = {}
for AlgoClass, name, log_dir, color in configs:
    x, y, mean_r, std_r = train_and_collect(AlgoClass, name, log_dir)
    results[name] = dict(x=x, y=y, mean_r=mean_r, std_r=std_r, color=color)
    print(f"  ✅ {name}: Mean reward = {mean_r:.1f} ± {std_r:.1f}\n")

# ── Summary table ─────────────────────────────────────────────────────────────
print("=" * 45)
print(f"{'Algorithm':<12} {'Mean Reward':>14} {'Std':>10}")
print("-" * 45)
for name, r in results.items():
    print(f"{name:<12} {r['mean_r']:>14.1f} {r['std_r']:>10.1f}")
print("=" * 45)
print("\n💡 PPO typically converges more stably than A2C due to its")
print("   clipped surrogate objective, limiting policy update size.")

# ── Plot ──────────────────────────────────────────────────────────────────────
WINDOW = 10
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Algorithm Comparison: PPO vs A2C on CartPole-v1", fontsize=14, fontweight="bold")

for ax, (name, r) in zip(axes, results.items()):
    x, y, color = r["x"], r["y"], r["color"]
    ax.plot(x, y, alpha=0.2, color=color)
    if len(y) >= WINDOW:
        y_s = np.convolve(y, np.ones(WINDOW) / WINDOW, mode="valid")
        x_s = x[WINDOW - 1:]
        ax.plot(x_s, y_s, color=color, linewidth=2, label=name)
    ax.axhline(y=500, color="green", linestyle="--", label="Max (500)", alpha=0.7)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Reward")
    ax.set_title(f"{name}  |  Mean: {r['mean_r']:.0f} ± {r['std_r']:.0f}")
    ax.legend()

plt.tight_layout()

png_path = os.path.abspath(f"{RESULT_DIR}/algorithm_comparison.png")
plt.savefig(png_path, dpi=150)
print(f"\n📊 Chart saved → {png_path}")

# ── Open results folder in Finder ─────────────────────────────────────────────
results_abs = os.path.abspath(RESULT_DIR)
print(f"📁 Opening results folder → {results_abs}")
subprocess.run(["open", results_abs])
