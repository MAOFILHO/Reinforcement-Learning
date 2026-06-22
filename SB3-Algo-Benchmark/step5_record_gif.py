"""
STEP 5 — Record GIFs of trained agents in action
=================================================
Portfolio goal: Visual proof of the agent working. A GIF of the
LunarLander landing or CartPole balancing is perfect for:
  - GitHub README
  - LinkedIn post
  - Portfolio website

Prerequisites:
    pip install imageio imageio-ffmpeg

Run (after step1 and step3):
    python step5_record_gif.py

Outputs:
    results/cartpole_agent.gif
    results/lunarlander_agent.gif
"""

import os
import subprocess
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO, DQN

RESULT_DIR = "results"
MODEL_DIR  = "models"
os.makedirs(RESULT_DIR, exist_ok=True)

# ── Check imageio ─────────────────────────────────────────────────────────────
try:
    import imageio
except ImportError:
    print("❌ imageio not installed. Run: pip install imageio imageio-ffmpeg")
    raise

def record_gif(model, env_id, gif_path, n_steps=300, fps=30):
    """Record a GIF of a trained agent."""
    env = gym.make(env_id, render_mode="rgb_array")
    obs, _ = env.reset()
    frames = []

    print(f"  Recording {n_steps} frames for {env_id}...")
    for _ in range(n_steps):
        frame = env.render()
        frames.append(frame)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    abs_path = os.path.abspath(gif_path)
    imageio.mimsave(abs_path, [np.array(f) for f in frames], fps=fps)
    print(f"  ✅ GIF saved → {abs_path}  ({len(frames)} frames @ {fps}fps)")
    return abs_path

saved = []

# ── CartPole GIF ──────────────────────────────────────────────────────────────
cartpole_zip = f"{MODEL_DIR}/ppo_cartpole.zip"
if os.path.exists(cartpole_zip):
    print("\n🎬 Recording CartPole agent...")
    model_cp = PPO.load(cartpole_zip)
    path = record_gif(model_cp, "CartPole-v1", f"{RESULT_DIR}/cartpole_agent.gif", n_steps=400)
    saved.append(path)
else:
    print(f"⚠️  CartPole model not found at {cartpole_zip}. Run step1 first.")

# ── LunarLander GIF ───────────────────────────────────────────────────────────
lunar_zip = f"{MODEL_DIR}/dqn_lunarlander.zip"
if os.path.exists(lunar_zip):
    print("\n🎬 Recording LunarLander agent...")
    try:
        model_ll = DQN.load(lunar_zip)
        path = record_gif(model_ll, "LunarLander-v3", f"{RESULT_DIR}/lunarlander_agent.gif", n_steps=500)
        saved.append(path)
    except Exception as e:
        print(f"⚠️  LunarLander GIF failed: {e}")
        print("   Make sure box2d is installed: pip install 'gymnasium[box2d]'")
else:
    print(f"⚠️  LunarLander model not found at {lunar_zip}. Run step3 first.")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print(f"🎯 Done! {len(saved)} GIF(s) saved:")
for p in saved:
    print(f"   {p}")

# ── Open results folder in Finder ─────────────────────────────────────────────
results_abs = os.path.abspath(RESULT_DIR)
print(f"\n📁 Opening results folder → {results_abs}")
subprocess.run(["open", results_abs])
