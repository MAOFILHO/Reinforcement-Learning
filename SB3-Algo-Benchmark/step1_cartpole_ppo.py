"""
STEP 1 — CartPole-v1 with PPO + TensorBoard + Reward Plot
==========================================================
Portfolio goal: Show a trained agent solving CartPole, with a reward
learning curve saved as a PNG.

Run:
    python step1_cartpole_ppo.py

Outputs:
    logs/cartpole_ppo/          ← TensorBoard logs
    models/ppo_cartpole.zip     ← Saved model
    models/best_cartpole_ppo/   ← Best model checkpoint
    results/cartpole_reward.png ← Reward curve PNG
"""

import os
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Save to file without needing a display window
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import EvalCallback

# ── Directories ───────────────────────────────────────────────────────────────
LOG_DIR    = "logs/cartpole_ppo"
MODEL_DIR  = "models"
RESULT_DIR = "results"
for d in [LOG_DIR, MODEL_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Environment ───────────────────────────────────────────────────────────────
env      = Monitor(gym.make("CartPole-v1"), LOG_DIR)
eval_env = Monitor(gym.make("CartPole-v1"))

# ── Model ─────────────────────────────────────────────────────────────────────
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    seed=42,
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"{MODEL_DIR}/best_cartpole_ppo",
    log_path=LOG_DIR,
    eval_freq=5_000,
    deterministic=True,
    render=False,
)

print("\n🚀 Training PPO on CartPole-v1 for 100,000 timesteps...\n")
model.learn(total_timesteps=100_000, callback=eval_callback)

# ── Save model ────────────────────────────────────────────────────────────────
model_path = os.path.abspath(f"{MODEL_DIR}/ppo_cartpole.zip")
model.save(f"{MODEL_DIR}/ppo_cartpole")
print(f"\n✅ Model saved → {model_path}")

# ── Evaluate ──────────────────────────────────────────────────────────────────
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
print(f"\n📈 Evaluation over 20 episodes:")
print(f"   Mean reward : {mean_reward:.1f} ± {std_reward:.1f}  (max possible: 500)")

# ── Plot reward curve ─────────────────────────────────────────────────────────
x, y = ts2xy(load_results(LOG_DIR), "timesteps")
window = 10
y_smooth = np.convolve(y, np.ones(window) / window, mode="valid")
x_smooth = x[window - 1:]

plt.figure(figsize=(10, 5))
plt.plot(x, y, alpha=0.3, color="steelblue", label="Episode reward")
plt.plot(x_smooth, y_smooth, color="steelblue", linewidth=2, label=f"Smoothed (window={window})")
plt.axhline(y=500, color="green", linestyle="--", label="Max reward (500)")
plt.xlabel("Timesteps")
plt.ylabel("Episode Reward")
plt.title("PPO on CartPole-v1 — Learning Curve")
plt.legend()
plt.tight_layout()

png_path = os.path.abspath(f"{RESULT_DIR}/cartpole_reward.png")
plt.savefig(png_path, dpi=150)
print(f"\n📊 Chart saved → {png_path}")

# ── Open results folder in Finder ─────────────────────────────────────────────
results_abs = os.path.abspath(RESULT_DIR)
print(f"\n📁 Opening results folder → {results_abs}")
subprocess.run(["open", results_abs])

env.close()
eval_env.close()
