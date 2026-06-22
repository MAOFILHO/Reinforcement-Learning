# Deep Reinforcement Learning with Stable-Baselines3

## Benchmarking PPO, A2C, and DQN on Control & Physics Environments

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.9.0-00A98F?style=flat)](https://stable-baselines3.readthedocs.io)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Farama-0081A5?style=flat)](https://gymnasium.farama.org)
[![TensorBoard](https://img.shields.io/badge/TensorBoard-Tracking-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/tensorboard)
[![Tests](https://img.shields.io/badge/tests-873%20passed-success?style=flat)](https://stable-baselines3.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Trained, benchmarked, and evaluated multiple deep reinforcement learning agents using **Stable-Baselines3 (SB3)** — the industry-standard PyTorch RL library. This project implements a full experimental workflow across three algorithm families (on-policy, off-policy, and actor-critic) on classic control and physics-simulation environments, with **TensorBoard experiment tracking**, **automated model persistence**, and **reproducible evaluation metrics**.

**Key differentiator:** This is not a single-model demo. It's a structured benchmark comparing algorithm convergence behaviour, with quantitative evaluation (mean ± std reward over 20 episodes), live training telemetry, and recorded agent rollouts — the same workflow used to validate RL systems before production deployment.

---

## The Problem

Reinforcement learning is one of the hardest branches of machine learning to operationalize. Unlike supervised learning, where you have a fixed labelled dataset, RL agents learn by trial and error through millions of interactions with an environment — and that process is notoriously unstable, sample-inefficient, and hard to reproduce.

For teams adopting RL, the practical barriers are steep:

| Pain Point | Operational Impact |
|---|---|
| Algorithm implementations differ subtly across papers | A "PPO" in one repo behaves differently from another — results don't replicate |
| Training is unstable and seed-sensitive | The same code can converge brilliantly or collapse, run to run |
| No standard way to track or compare experiments | Tuning becomes guesswork without reward curves and telemetry |
| Choosing the right algorithm per problem is non-obvious | On-policy vs off-policy vs actor-critic, each fits different action spaces |
| Models are hard to persist, reload, and evaluate consistently | Hard to move from "it trained" to "it works, measurably" |

This matters commercially. The reinforcement learning market was valued at over **$52 billion in 2024**, with large enterprises representing roughly **70% of demand** and the BFSI sector alone accounting for about **20% of the market** — driven by algorithmic trading, fraud detection, and risk management use cases. RL adoption is accelerating across robotics, autonomous vehicles, supply chain optimization, and healthcare, but the field's reproducibility and tooling gap remain the primary obstacle to moving prototypes into production.

**The result:** practitioners spend more time fighting unstable training and non-reproducible baselines than solving their actual problem.

---

## The Solution: A Reproducible Multi-Algorithm RL Benchmark

A structured, end-to-end experimentation workflow built on Stable-Baselines3 that turns RL from a research gamble into a measurable, repeatable engineering process. Every experiment is seeded, tracked, persisted, and evaluated the same way.

- **Three algorithm families, one interface** — PPO (on-policy), A2C (actor-critic), and DQN (off-policy value-based), all using SB3's sklearn-like common API so they're directly comparable.
- **Reproducible by design** — fixed random seeds, deterministic evaluation, and pinned dependencies mean results replicate run to run.
- **Live experiment tracking** — every training run streams reward, loss, and entropy metrics to TensorBoard in real time, so convergence is observable, not assumed.
- **Automated model lifecycle** — `EvalCallback` checkpoints the best model automatically; train → save → reload → evaluate demonstrates the full persistence cycle.
- **Quantitative evaluation** — agents are scored as mean ± std reward over 20 independent episodes against documented "solved" thresholds, not eyeballed.
- **Algorithm benchmarking** — a controlled head-to-head (PPO vs A2C on identical environment and seed) isolates convergence-speed and stability differences.
- **Visual verification** — trained agents are recorded as GIFs, providing immediate visual proof that the policy works.
- **Validated foundation** — the underlying SB3 library passes 873 unit tests at 95% code coverage, confirming the implementations are trustworthy baselines.

---

## Results & Impact

→ **PPO achieved a perfect score on CartPole-v1 — mean reward 500/500** — converging to the environment's maximum within ~60k timesteps and holding it with zero variance across 20 evaluation episodes.

→ **DQN solved LunarLander-v3 (mean reward > 200)** — the documented "solved" threshold — in 200k timesteps, demonstrating an off-policy agent learning a stable landing policy from scratch.

→ **Benchmarked PPO vs A2C head-to-head** — confirming PPO's more stable convergence, attributable to its clipped surrogate objective that constrains the size of each policy update.

→ **100% reproducible runs** — seeded training and deterministic evaluation mean every metric in this repo can be regenerated by re-running the scripts.

→ **Full experiment observability** — 4 training runs tracked end-to-end in TensorBoard (reward, value loss, entropy), turning opaque training into an auditable process.

### Benchmark Summary

| Experiment | Algorithm | Environment | Result | Threshold |
|---|---|---|---|---|
| 1 | **PPO** | CartPole-v1 | **500 ± 0** (perfect) | 500 (max) |
| 3 | **DQN** | LunarLander-v3 | **> 200** (solved) | 200 |
| 4 | **PPO vs A2C** | CartPole-v1 | PPO converges faster & more stably | — |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **RL Library** | Stable-Baselines3 2.9.0 (PyTorch) |
| **Deep Learning** | PyTorch 2.8+ |
| **Environments** | Gymnasium (CartPole-v1, LunarLander-v3) + Box2D physics |
| **Algorithms** | PPO, A2C (on-policy / actor-critic), DQN (off-policy) |
| **Experiment Tracking** | TensorBoard |
| **Evaluation** | SB3 `evaluate_policy`, `EvalCallback` |
| **Visualization** | Matplotlib (reward curves), imageio (agent GIFs) |
| **Language** | Python 3.11 |

---

## Implemented Algorithms

| **Name**         | **Recurrent**      | `Box`          | `Discrete`     | `MultiDiscrete` | `MultiBinary`  | **Multi Processing**              |
| ------------------- | ------------------ | ------------------ | ------------------ | ------------------- | ------------------ | --------------------------------- |
| ARS<sup>[1](#f1)</sup>   | :x: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | :heavy_check_mark: |
| A2C   | :x: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| CrossQ<sup>[1](#f1)</sup>   | :x: | :heavy_check_mark: | :x:                | :x:                 | :x:                | :heavy_check_mark: |
| DDPG  | :x: | :heavy_check_mark: | :x:                | :x:                 | :x:                | :heavy_check_mark: |
| DQN   | :x: | :x: | :heavy_check_mark: | :x:                 | :x:                | :heavy_check_mark: |
| HER   | :x: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | :heavy_check_mark: |
| PPO   | :x: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
| QR-DQN<sup>[1](#f1)</sup>  | :x: | :x: | :heavy_check_mark: | :x:                 | :x:                | :heavy_check_mark: |
| RecurrentPPO<sup>[1](#f1)</sup>   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
| SAC   | :x: | :heavy_check_mark: | :x:                | :x:                 | :x:                | :heavy_check_mark: |
| TD3   | :x: | :heavy_check_mark: | :x:                | :x:                 | :x:                | :heavy_check_mark: |
| TQC<sup>[1](#f1)</sup>   | :x: | :heavy_check_mark: | :x:                | :x:                 | :x: | :heavy_check_mark: |
| TRPO<sup>[1](#f1)</sup>  | :x: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
| Maskable PPO<sup>[1](#f1)</sup>   | :x: | :x: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:  |

---

## Setup

**Note:** Stable-Baselines3 supports PyTorch >= 2.8 and requires Python 3.10+.

<img width="1071" height="218" alt="Install dependencies" src="https://github.com/user-attachments/assets/9d58cc33-0cb0-4b00-b83d-9357676a9e2c" />

### Install dependencies

```sh
# Core library with extras (Tensorboard, OpenCV, plotting)
pip install -e '.[docs,tests,extra]'

# Box2D physics engine for LunarLander (Steps 3 & 5)
pip install 'gymnasium[box2d]'

# GIF recording (Step 5)
pip install imageio imageio-ffmpeg
```

<img width="1058" height="371" alt="Screenshot 2026-06-22 at 12 23 15 PM" src="https://github.com/user-attachments/assets/c9abe416-fd09-4872-8cc5-5912f08ba3fe" />


### Validate the installation

All 873 unit tests in Stable-Baselines3 can be run using the `pytest` runner:

```sh
make pytest
```

<img width="1076" height="709" alt="Pytest run starting" src="https://github.com/user-attachments/assets/27cef1ad-274d-42e5-b5ba-ab730b62eac7" />

<img width="1073" height="709" alt="Pytest results 873 passed 95% coverage" src="https://github.com/user-attachments/assets/dc3bb94e-1f12-489c-b649-9ff3903f6328" />

Optional static type check with `mypy`:

```sh
make type
```

<img width="1025" height="159" alt="mypy type check" src="https://github.com/user-attachments/assets/e05ce041-81f7-444e-86f7-0f94fec7a117" />

Codestyle check with `ruff`:

```sh
make lint
```

<img width="1023" height="156" alt="ruff lint check" src="https://github.com/user-attachments/assets/937222d1-3654-44cb-a1e9-4c501db54a09" />

---

## Experiment Workflow

The project is organized as five sequential experiments, each a standalone script that trains, evaluates, and saves its outputs to `results/`.

| Step | Experiment | Algorithm | Output |
|---|---|---|---|
| 1 | CartPole | PPO | Training logs, reward curve |
| 2 | TensorBoard | — | Live training dashboard |
| 3 | LunarLander | DQN | Train → Save → Reload → Evaluate |
| 4 | Algorithm Comparison | PPO vs A2C | Side-by-side benchmark chart |
| 5 | GIF Recording | PPO + DQN | Animated agent rollouts |

<img width="1058" height="371" alt="Project structure" src="https://github.com/user-attachments/assets/c45582bd-827a-4665-b823-2f99ca13c4ad" />

---

### Step 1 — CartPole with PPO (~2 min)

Trains a PPO agent for 100k timesteps, auto-checkpoints the best model, evaluates over 20 episodes, and saves a reward curve.

```bash
python step1_cartpole_ppo.py
```

<img width="831" height="557" alt="PPO training output" src="https://github.com/user-attachments/assets/fc4047b2-6869-4dff-a986-cba1fda7b361" />

<img width="712" height="544" alt="PPO evaluation result" src="https://github.com/user-attachments/assets/50f5a017-98ea-4108-9f93-69add1d761ca" />

<img width="821" height="545" alt="PPO model saved" src="https://github.com/user-attachments/assets/431a0714-0763-4719-ba4f-660bf833b2e6" />

<img width="996" height="496" alt="CartPole PPO reward curve" src="https://github.com/user-attachments/assets/8a6d1610-df13-4bdf-bacb-9d81a98f2ef8" />

**Result:** Mean reward **500 ± 0** (perfect) — the agent reaches and holds the maximum achievable score.

---

### Step 2 — TensorBoard Experiment Tracking

Launches TensorBoard to inspect live training curves — reward, value loss, and entropy across all runs.

```bash
tensorboard --logdir logs/
```

<img width="1025" height="401" alt="TensorBoard launch" src="https://github.com/user-attachments/assets/8a91d774-ee87-4827-b126-73ad4fee9db5" />

Open `http://localhost:6006`:

<img width="1534" height="966" alt="TensorBoard reward curve" src="https://github.com/user-attachments/assets/b36ff33f-90c5-42d3-88c5-600d45fea4c2" />
<img width="1539" height="967" alt="TensorBoard entropy loss" src="https://github.com/user-attachments/assets/f53a1d15-3bb8-4132-879f-35ff505a3a95" />
<img width="1538" height="927" alt="TensorBoard value loss" src="https://github.com/user-attachments/assets/79631e16-4484-46a0-ae69-fc264e2e317a" />

---

### Step 3 — LunarLander with DQN (~8 min)

Demonstrates the full ML lifecycle: train an off-policy DQN agent, save it, delete it from memory, reload it from disk, then evaluate — proving model persistence works end to end.

```bash
python step3_lunarlander_dqn.py
```

<img width="1042" height="714" alt="DQN training progress" src="https://github.com/user-attachments/assets/80ef5d8f-2ebd-4b56-911b-97db657e685a" />

<img width="791" height="553" alt="DQN reload from disk" src="https://github.com/user-attachments/assets/fddd13d3-f143-4238-ade9-6a4520c0c713" />

<img width="1023" height="710" alt="DQN evaluation result" src="https://github.com/user-attachments/assets/413ee9c2-789f-4b1b-94cb-2994e3101e36" />

<img width="1318" height="652" alt="LunarLander DQN reward curve" src="https://github.com/user-attachments/assets/f0b94122-1b8d-4b40-9067-4262227c6570" />

**Result:** Mean reward **> 200** — the documented "solved" threshold for LunarLander-v3.

---

### Step 4 — Algorithm Comparison: PPO vs A2C (~4 min)

The core benchmark. Trains both algorithms on the identical environment with the same seed, then plots their learning curves side by side.

```bash
python step4_algorithm_comparison.py
```

<img width="1034" height="383" alt="PPO vs A2C summary table" src="https://github.com/user-attachments/assets/796187e8-27b2-446c-9b21-2c445f04f293" />

<img width="1317" height="467" alt="PPO vs A2C comparison chart" src="https://github.com/user-attachments/assets/7ba09330-50c3-4684-b929-cde03dec651f" />

**Insight:** PPO converges more stably than A2C, due to its clipped surrogate objective limiting the magnitude of each policy update — a textbook illustration of why PPO became the default choice for many production RL systems.

---

### Step 5 — Record Agent Rollouts (~1 min)

Records the trained agents as GIFs for visual verification and sharing.

```bash
python step5_record_gif.py
```

<img width="1031" height="643" alt="GIF recording output" src="https://github.com/user-attachments/assets/b9777618-fc18-4045-aae3-72c6c7b9d256" />

| CartPole (PPO) | LunarLander (DQN) |
|---|---|
| <img width="400" alt="CartPole agent" src="https://github.com/user-attachments/assets/c86da681-16b7-4b27-ad3e-8e73555fbc64" /> | <img width="400" alt="LunarLander agent" src="https://github.com/user-attachments/assets/d24014ff-3336-4434-b467-1e28be4df74e" /> |

---

## Project Structure

```
stable-baselines3/
├── README.md                       # This file
├── step1_cartpole_ppo.py           # PPO on CartPole + reward curve
├── step2_tensorboard.py            # TensorBoard launcher
├── step3_lunarlander_dqn.py        # DQN train/save/reload/evaluate
├── step4_algorithm_comparison.py   # PPO vs A2C benchmark
├── step5_record_gif.py             # Agent rollout GIFs
│
├── results/                        # Generated: reward curves, GIFs (gitignored)
├── logs/                           # Generated: TensorBoard logs (gitignored)
├── models/                         # Generated: saved agents (gitignored)
│
├── stable_baselines3/              # SB3 library source (PPO, A2C, DQN, SAC, TD3, DDPG, HER)
├── tests/                          # 873 unit tests, 95% coverage
└── setup.py / pyproject.toml       # Packaging
```

---

## What This Project Demonstrates

| Skill | Evidence |
|---|---|
| **RL algorithm fluency** | Trained on-policy (PPO/A2C) and off-policy (DQN) agents and explained their behavioural differences |
| **Experiment discipline** | Seeded, reproducible runs with TensorBoard tracking |
| **ML lifecycle** | Train → checkpoint → save → reload → evaluate, fully automated |
| **Quantitative rigor** | Mean ± std evaluation against documented solved-thresholds |
| **Benchmarking** | Controlled head-to-head algorithm comparison |
| **Communication** | Reward curves, telemetry dashboards, and recorded rollouts |

---

## References & Citing

This project builds on the Stable-Baselines3 library. To cite SB3 in publications:

```bibtex
@article{stable-baselines3,
  author  = {Antonin Raffin and Ashley Hill and Adam Gleave and Anssi Kanervisto and Maximilian Ernestus and Noah Dormann},
  title   = {Stable-Baselines3: Reliable Reinforcement Learning Implementations},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {268},
  pages   = {1-8},
  url     = {https://jmlr.org/papers/v22/20-1364.html}
}
```

- **Stable-Baselines3 Documentation:** https://stable-baselines3.readthedocs.io/
- **Original Repository:** https://github.com/DLR-RM/stable-baselines3
- **JMLR Paper:** https://jmlr.org/papers/volume22/20-1364/20-1364.pdf
- **RL Baselines3 Zoo (training framework):** https://github.com/DLR-RM/rl-baselines3-zoo

## Maintainers

Stable-Baselines3 is currently maintained by [Ashley Hill](https://github.com/hill-a) (aka @hill-a), [Antonin Raffin](https://araffin.github.io/) (aka [@araffin](https://github.com/araffin)), [Maximilian Ernestus](https://github.com/ernestum) (aka @ernestum), [Adam Gleave](https://github.com/adamgleave) (@AdamGleave), [Anssi Kanervisto](https://github.com/Miffyli) (@Miffyli) and [Quentin Gallouédec](https://github.com/qgallouedec) (@qgallouedec).

**Important Note: We do not provide technical support or consulting** and do not answer personal questions via email.
Please post your question on the [RL Discord](https://discord.com/invite/xhfNqQv), [Reddit](https://www.reddit.com/r/reinforcementlearning/), or [Stack Overflow](https://stackoverflow.com/) in that case.


## How To Contribute

For anyone interested in making the baselines better, there is still some documentation that needs to be done.
If you want to contribute, please read [**CONTRIBUTING.md**](./CONTRIBUTING.md) guide first.

## Acknowledgments

The initial work to develop Stable Baselines3 was partially funded by the project *Reduced Complexity Models* from the *Helmholtz-Gemeinschaft Deutscher Forschungszentren*, and by the EU Horizon 2020 Research and Innovation Programme under grant number 951992 ([VeriDream](https://www.veridream.eu/)).


---

## Author

**Marcos Oliveira** — [LinkedIn](https://www.linkedin.com/in/mfilho1/) | [GitHub](https://github.com/MAOFILHO) | [Website](https://maofilho.github.io/) 

Built on Stable-Baselines3 and PyTorch. SB3 is maintained by the DLR-RM team and released under the MIT License.
