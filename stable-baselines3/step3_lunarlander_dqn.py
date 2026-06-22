"""
STEP 3 — LunarLander-v3 with DQN: Train → Save → Load → Evaluate
=================================================================
Portfolio goal: Full ML lifecycle — training, persistence, reloading,
and quantitative evaluation with mean ± std reward.

Prerequisites:
    pip install 'gymnasium[box2d]'

Run:
    python step3_lunarlander_dqn.py

Outputs:
    logs/lunarlander_dqn/          ← TensorBoard logs
    models/dqn_lunarlander.zip     ← Saved model
    models/best_lunarlander_dqn/   ← Best model checkpoint
    results/lunarlander_reward.png ← Reward curve PNG
"""

import os
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import EvalCallback

# ── Directories ───────────────────────────────────────────────────────────────
LOG_DIR    = "logs/lunarlander_dqn"
MODEL_DIR  = "models"
RESULT_DIR = "results"
for d in [LOG_DIR, MODEL_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Check box2d ───────────────────────────────────────────────────────────────
try:
    env_test = gym.make("LunarLander-v3")
    env_test.close()
except Exception:
    print("❌ box2d not installed. Run: pip install 'gymnasium[box2d]'")
    raise

# ── Environments ──────────────────────────────────────────────────────────────
env      = Monitor(gym.make("LunarLander-v3"), LOG_DIR)
eval_env = Monitor(gym.make("LunarLander-v3"))

# ── Model ─────────────────────────────────────────────────────────────────────
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=1_000,
    batch_size=64,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1_000,
    exploration_fraction=0.2,
    exploration_final_eps=0.05,
    seed=42,
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"{MODEL_DIR}/best_lunarlander_dqn",
    log_path=LOG_DIR,
    eval_freq=10_000,
    deterministic=True,
    render=False,
)

print("\n🚀 Training DQN on LunarLander-v3 for 200,000 timesteps...")
print("   ~5–10 min on Mac M-series. Good time for a coffee ☕\n")
model.learn(total_timesteps=200_000, callback=eval_callback, progress_bar=True)

# ── Save ──────────────────────────────────────────────────────────────────────
model_path = os.path.abspath(f"{MODEL_DIR}/dqn_lunarlander.zip")
model.save(f"{MODEL_DIR}/dqn_lunarlander")
print(f"\n✅ Model saved → {model_path}")

# ── Delete and reload (demonstrates persistence) ──────────────────────────────
del model
print("🔄 Reloading model from disk...")
model = DQN.load(f"{MODEL_DIR}/dqn_lunarlander", env=eval_env)
print("✅ Model reloaded.\n")

# ── Evaluate ──────────────────────────────────────────────────────────────────
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
print("=" * 50)
print(f"📈 Final Evaluation (20 episodes):")
print(f"   Mean reward : {mean_reward:.1f}")
print(f"   Std  reward : ± {std_reward:.1f}")
print(f"   Solved threshold: 200+")
print("=" * 50)

# ── Plot ──────────────────────────────────────────────────────────────────────
x, y = ts2xy(load_results(LOG_DIR), "timesteps")
window = 20
if len(y) >= window:
    y_smooth = np.convolve(y, np.ones(window) / window, mode="valid")
    x_smooth = x[window - 1:]
else:
    y_smooth, x_smooth = y, x

plt.figure(figsize=(10, 5))
plt.plot(x, y, alpha=0.2, color="darkorange", label="Episode reward")
plt.plot(x_smooth, y_smooth, color="darkorange", linewidth=2, label=f"Smoothed (window={window})")
plt.axhline(y=200, color="green", linestyle="--", label="Solved threshold (200)")
plt.axhline(y=0, color="red", linestyle=":", alpha=0.5, label="Zero reward")
plt.xlabel("Timesteps")
plt.ylabel("Episode Reward")
plt.title("DQN on LunarLander-v3 — Learning Curve")
plt.legend()
plt.tight_layout()

png_path = os.path.abspath(f"{RESULT_DIR}/lunarlander_reward.png")
plt.savefig(png_path, dpi=150)
print(f"\n📊 Chart saved → {png_path}")

# ── Open results folder in Finder ─────────────────────────────────────────────
results_abs = os.path.abspath(RESULT_DIR)
print(f"📁 Opening results folder → {results_abs}")
subprocess.run(["open", results_abs])

env.close()
eval_env.close()
