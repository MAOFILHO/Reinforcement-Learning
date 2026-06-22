<!-- [![pipeline status](https://gitlab.com/araffin/stable-baselines3/badges/master/pipeline.svg)](https://gitlab.com/araffin/stable-baselines3/-/commits/master) -->
[![CI](https://github.com/DLR-RM/stable-baselines3/workflows/CI/badge.svg)](https://github.com/DLR-RM/stable-baselines3/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/stable-baselines/badge/?version=master)](https://stable-baselines3.readthedocs.io/en/master/?badge=master) [![coverage report](https://gitlab.com/araffin/stable-baselines3/badges/master/coverage.svg)](https://github.com/DLR-RM/stable-baselines3/actions/workflows/ci.yml)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


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


---


# Stable Baselines3

<img src="docs/\_static/img/logo.png" align="right" width="40%"/>

Stable Baselines3 (SB3) is a set of reliable implementations of reinforcement learning algorithms in PyTorch. It is the next major version of [Stable Baselines](https://github.com/hill-a/stable-baselines).


You can read a detailed presentation of Stable Baselines3 in the [v1.0 blog post](https://araffin.github.io/post/sb3/) or our [JMLR paper](https://jmlr.org/papers/volume22/20-1364/20-1364.pdf).


These algorithms will make it easier for the research community and industry to replicate, refine, and identify new ideas, and will create good baselines to build projects on top of. We expect these tools will be used as a base around which new ideas can be added, and as a tool for comparing a new approach against existing ones. We also hope that the simplicity of these tools will allow beginners to experiment with a more advanced toolset, without being buried in implementation details.

**Note: Despite its simplicity of use, Stable Baselines3 (SB3) assumes you have some knowledge about Reinforcement Learning (RL).** You should not utilize this library without some practice. To that extent, we provide good resources in the [documentation](https://stable-baselines3.readthedocs.io/en/master/guide/rl.html) to get started with RL.

## Main Features

**The performance of each algorithm was tested** (see *Results* section in their respective page).


| **Features**                | **Stable-Baselines3** |
| --------------------------- | ----------------------|
| State of the art RL methods | :heavy_check_mark: |
| Documentation               | :heavy_check_mark: |
| Custom environments         | :heavy_check_mark: |
| Custom policies             | :heavy_check_mark: |
| Common interface            | :heavy_check_mark: |
| `Dict` observation space support  | :heavy_check_mark: |
| Ipython / Notebook friendly | :heavy_check_mark: |
| Tensorboard support         | :heavy_check_mark: |
| PEP8 code style             | :heavy_check_mark: |
| Custom callback             | :heavy_check_mark: |
| High code coverage          | :heavy_check_mark: |
| Type hints                  | :heavy_check_mark: |


## Installation

**Note:** Stable-Baselines3 supports PyTorch >= 2.8

### Prerequisites
Stable Baselines3 requires Python 3.10+.

<img width="1071" height="218" alt="Screenshot 2026-06-22 at 12 00 24 PM" src="https://github.com/user-attachments/assets/9ec72376-f98e-4dfc-99c5-aae94c50f3bd" />


### Install using pip
Install the Stable Baselines3 package:
```sh
pip install 'stable-baselines3[extra]'
```

This includes optional dependencies like Tensorboard, OpenCV, `ale-py` to train on atari games, as well as `pandas` and `matplotlib` for plotting and analyzing results. If you do not need those, you can use:
```sh
pip install stable-baselines3
```

Please read the [documentation](https://stable-baselines3.readthedocs.io/) for more details and alternatives (from source, using docker).

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

<b id="f1">1</b>: Implemented in [SB3 Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) GitHub repository.

Actions `gymnasium.spaces`:
 * `Box`: A N-dimensional box that contains every point in the action space.
 * `Discrete`: A list of possible actions, where only one action can be used per timestep.
 * `MultiDiscrete`: A list of possible actions, where each timestep only one action of each discrete set can be used.
 * `MultiBinary`: A list of possible actions, where each timestep any of the actions can be used in any combination.



## Testing the installation

### Install dependencies
```sh
pip install -e '.[docs,tests,extra]'
```

<img width="1071" height="218" alt="Screenshot 2026-06-22 at 12 00 24 PM" src="https://github.com/user-attachments/assets/9d58cc33-0cb0-4b00-b83d-9357676a9e2c" />


### Run tests
All unit tests in stable baselines3 can be run using `pytest` runner:
```sh
make pytest
```

<img width="1076" height="709" alt="Screenshot 2026-06-22 at 12 01 00 PM" src="https://github.com/user-attachments/assets/27cef1ad-274d-42e5-b5ba-ab730b62eac7" />


<img width="1073" height="709" alt="Screenshot 2026-06-22 at 12 04 13 PM" src="https://github.com/user-attachments/assets/dc3bb94e-1f12-489c-b649-9ff3903f6328" />


To run a single test file:
```sh
python3 -m pytest -v tests/test_env_checker.py
```
To run a single test:
```sh
python3 -m pytest -v -k 'test_check_env_dict_action'
```

You can also do a static type check using `mypy`:
```sh
pip install mypy
make type
```

<img width="1025" height="159" alt="Screenshot 2026-06-22 at 1 54 41 PM" src="https://github.com/user-attachments/assets/e05ce041-81f7-444e-86f7-0f94fec7a117" />


Codestyle check with `ruff`:
```sh
pip install ruff
make lint
```

<img width="1023" height="156" alt="Screenshot 2026-06-22 at 1 55 29 PM" src="https://github.com/user-attachments/assets/937222d1-3654-44cb-a1e9-4c501db54a09" />


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

<img width="1058" height="371" alt="Screenshot 2026-06-22 at 12 23 15 PM" src="https://github.com/user-attachments/assets/c45582bd-827a-4665-b823-2f99ca13c4ad" />


---

## Step 1 — CartPole PPO (~2 min)

```bash
python step1_cartpole_ppo.py
```

<img width="831" height="557" alt="Screenshot 2026-06-22 at 2 05 51 PM" src="https://github.com/user-attachments/assets/fc4047b2-6869-4dff-a986-cba1fda7b361" />

<img width="712" height="544" alt="Screenshot 2026-06-22 at 2 07 25 PM" src="https://github.com/user-attachments/assets/50f5a017-98ea-4108-9f93-69add1d761ca" />

<img width="821" height="545" alt="Screenshot 2026-06-22 at 2 07 00 PM" src="https://github.com/user-attachments/assets/431a0714-0763-4719-ba4f-660bf833b2e6" />

<img width="996" height="496" alt="Screenshot 2026-06-22 at 12 26 01 PM" src="https://github.com/user-attachments/assets/8a6d1610-df13-4bdf-bacb-9d81a98f2ef8" />


**Screenshot:**
- Terminal output showing rollout/ep_rew_mean climbing toward 500
- `results/cartpole_reward.png` (reward curve)

---

## Step 2 — TensorBoard (~30 sec setup)

In a **new terminal tab** (with .venv active):
```bash
tensorboard --logdir logs/
```

<img width="1025" height="401" alt="Screenshot 2026-06-22 at 12 36 19 PM" src="https://github.com/user-attachments/assets/8a91d774-ee87-4827-b126-73ad4fee9db5" />


Open http://localhost:6006

<img width="1534" height="966" alt="Screenshot 2026-06-22 at 12 39 02 PM" src="https://github.com/user-attachments/assets/b36ff33f-90c5-42d3-88c5-600d45fea4c2" />
<img width="1539" height="967" alt="Screenshot 2026-06-22 at 12 38 40 PM" src="https://github.com/user-attachments/assets/f53a1d15-3bb8-4132-879f-35ff505a3a95" />
<img width="1538" height="927" alt="Screenshot 2026-06-22 at 12 37 46 PM" src="https://github.com/user-attachments/assets/79631e16-4484-46a0-ae69-fc264e2e317a" />


**Screenshot:**
- `ep_rew_mean` chart showing upward curve
- `train/entropy_loss` chart
- Overview showing both runs (after Step 4)

---

## Step 3 — LunarLander DQN (~8 min)

```bash
python step3_lunarlander_dqn.py
```

<img width="1042" height="714" alt="Screenshot 2026-06-22 at 12 59 28 PM" src="https://github.com/user-attachments/assets/80ef5d8f-2ebd-4b56-911b-97db657e685a" />

<img width="791" height="553" alt="Screenshot 2026-06-22 at 2 13 20 PM" src="https://github.com/user-attachments/assets/fddd13d3-f143-4238-ade9-6a4520c0c713" />

<img width="1023" height="710" alt="Screenshot 2026-06-22 at 1 00 54 PM" src="https://github.com/user-attachments/assets/413ee9c2-789f-4b1b-94cb-2994e3101e36" />

<img width="1318" height="652" alt="Screenshot 2026-06-22 at 1 02 21 PM" src="https://github.com/user-attachments/assets/f0b94122-1b8d-4b40-9067-4262227c6570" />



**Screenshot:**
- Progress bar in terminal
- Final evaluation output: `Mean reward: XXX ± YYY`
- `results/lunarlander_reward.png`

---

## Step 4 — Algorithm Comparison (~4 min)

```bash
python step4_algorithm_comparison.py
```

<img width="1034" height="383" alt="Screenshot 2026-06-22 at 1 05 12 PM" src="https://github.com/user-attachments/assets/796187e8-27b2-446c-9b21-2c445f04f293" />

<img width="1317" height="467" alt="Screenshot 2026-06-22 at 1 04 43 PM" src="https://github.com/user-attachments/assets/7ba09330-50c3-4684-b929-cde03dec651f" />


**Screenshot:**
- `results/algorithm_comparison.png` ← **Best portfolio piece**
- Summary table in terminal

---

## Step 5 — Record Agent GIFs (~1 min)

```bash
python step5_record_gif.py
```

<img width="1031" height="643" alt="Screenshot 2026-06-22 at 2 18 13 PM" src="https://github.com/user-attachments/assets/b9777618-fc18-4045-aae3-72c6c7b9d256" />

<img width="600" height="400" alt="cartpole_agent" src="https://github.com/user-attachments/assets/c86da681-16b7-4b27-ad3e-8e73555fbc64" />

<img width="600" height="400" alt="lunarlander_agent" src="https://github.com/user-attachments/assets/d24014ff-3336-4434-b467-1e28be4df74e" />


**Output:**
- `results/cartpole_agent.gif`
- `results/lunarlander_agent.gif`

Use these in your GitHub README and LinkedIn posts.

---


## Citing the Project

To cite this repository in publications:

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

Note: If you need to refer to a specific version of SB3, you can also use the [Zenodo DOI](https://doi.org/10.5281/zenodo.8123988).

## Maintainers

Stable-Baselines3 is currently maintained by [Ashley Hill](https://github.com/hill-a) (aka @hill-a), [Antonin Raffin](https://araffin.github.io/) (aka [@araffin](https://github.com/araffin)), [Maximilian Ernestus](https://github.com/ernestum) (aka @ernestum), [Adam Gleave](https://github.com/adamgleave) (@AdamGleave), [Anssi Kanervisto](https://github.com/Miffyli) (@Miffyli) and [Quentin Gallouédec](https://github.com/qgallouedec) (@qgallouedec).

**Important Note: We do not provide technical support, or consulting** and do not answer personal questions via email.
Please post your question on the [RL Discord](https://discord.com/invite/xhfNqQv), [Reddit](https://www.reddit.com/r/reinforcementlearning/), or [Stack Overflow](https://stackoverflow.com/) in that case.


## How To Contribute

For anyone interested in making the baselines better, there is still some documentation that needs to be done.
If you want to contribute, please read [**CONTRIBUTING.md**](./CONTRIBUTING.md) guide first.

## Acknowledgments

The initial work to develop Stable Baselines3 was partially funded by the project *Reduced Complexity Models* from the *Helmholtz-Gemeinschaft Deutscher Forschungszentren*, and by the EU Horizon 2020 Research and Innovation Programme under grant number 951992 ([VeriDream](https://www.veridream.eu/)).
