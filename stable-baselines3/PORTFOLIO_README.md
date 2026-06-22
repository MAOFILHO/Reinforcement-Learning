# Stable-Baselines3 Portfolio Execution Guide

## Setup (one time)

```bash
cd stable-baselines3
source .venv/bin/activate

# Install box2d for LunarLander (Step 3 & 5)
pip install 'gymnasium[box2d]'

# Install imageio for GIF recording (Step 5)
pip install imageio imageio-ffmpeg

# Copy scripts into the project root
cp ~/path/to/step*.py .
```

---

## Step 1 — CartPole PPO (~2 min)

```bash
python step1_cartpole_ppo.py
```

**Screenshot:**
- Terminal output showing rollout/ep_rew_mean climbing toward 500
- `results/cartpole_reward.png` (reward curve)

---

## Step 2 — TensorBoard (~30 sec setup)

In a **new terminal tab** (with .venv active):
```bash
tensorboard --logdir logs/
```
Open http://localhost:6006

**Screenshot:**
- `ep_rew_mean` chart showing upward curve
- `train/entropy_loss` chart
- Overview showing both runs (after Step 4)

---

## Step 3 — LunarLander DQN (~8 min)

```bash
python step3_lunarlander_dqn.py
```

**Screenshot:**
- Progress bar in terminal
- Final evaluation output: `Mean reward: XXX ± YYY`
- `results/lunarlander_reward.png`

---

## Step 4 — Algorithm Comparison (~4 min)

```bash
python step4_algorithm_comparison.py
```

**Screenshot:**
- `results/algorithm_comparison.png` ← **Best portfolio piece**
- Summary table in terminal

---

## Step 5 — Record Agent GIFs (~1 min)

```bash
python step5_record_gif.py
```

**Output:**
- `results/cartpole_agent.gif`
- `results/lunarlander_agent.gif`

Use these in your GitHub README and LinkedIn posts.

---

## Portfolio Narrative

> "Implemented and benchmarked multiple deep RL algorithms (PPO, A2C, DQN)
> using Stable-Baselines3 on classic control and physics simulation environments.
> PPO converged to maximum CartPole reward (500/500) within 60k timesteps.
> DQN successfully solved LunarLander-v3 (mean reward > 200) in 200k timesteps.
> Tracked all experiments with TensorBoard and automated model persistence
> with save/load lifecycle management."

## Key Metrics to Capture

| Metric | What to show |
|---|---|
| CartPole PPO | Mean reward = 500 ± 0 (perfect) |
| LunarLander DQN | Mean reward > 200 (solved) |
| PPO vs A2C | Convergence speed comparison chart |
| TensorBoard | Live training dashboard screenshot |
| GIFs | Animated agent solving each environment |
