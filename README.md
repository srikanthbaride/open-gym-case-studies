# 📦 Open Gym Case Studies — Logistics & Robotics  

> _Textbook-aligned case studies for teaching Q-Learning / SARSA and extending to DQN / Actor–Critic._

[![CI](https://github.com/srikanthbaride/open-gym-case-studies/actions/workflows/ci.yml/badge.svg)](https://github.com/srikanthbaride/open-gym-case-studies/actions/workflows/ci.yml)
[![Last Commit](https://img.shields.io/github/last-commit/srikanthbaride/open-gym-case-studies)](https://github.com/srikanthbaride/open-gym-case-studies/commits/main)
![Textbook Alignment](https://img.shields.io/badge/Aligned_with-Reinforcement_Learning_Explained-blue)
[![CRC Press 2025](https://img.shields.io/badge/CRC%20Press-2025-blue)](https://www.routledge.com/)
![Part III – Case Studies](https://img.shields.io/badge/Part%20III-Case%20Studies-informational)
[![License](https://img.shields.io/badge/License-Educational-lightgrey.svg)](#license)

This repository contains **two real-world-motivated case studies** implemented with **Gymnasium (OpenAI Gym)**, aligned with the reinforcement-learning chapters (Bandits → MC → TD → Q-Learning/SARSA) in the textbook  
📘 **[_Reinforcement Learning Explained_ (CRC Press | Taylor & Francis, 2025)](https://github.com/srikanthbaride/rl-explained-preprint)**.

Each case study includes **reproducible training scripts**, **CSV logs**, **plots**, and **pytest smoke tests**.  
Extensions are scaffolded for **Deep RL (DQN / Actor–Critic)**.

---

## 📘 Relation to the Textbook

| Chapter | Environment | Description | Folder |
|:--|:--|:--|:--|
| **Ch. 12 – Interacting with Environments using Gymnasium** | `gymnasium` API | Unified interface for observation/action spaces and episode control | `case_studies/00_frozenlake_navigation/` |
| **Ch. 13 – Taxi-v3 : Temporal-Difference Control in a Discrete Grid World** | `Taxi-v3` | Q-Learning vs SARSA with ε-decay schedules and Bellman-optimal policy convergence | `case_studies/01_taxi_last_mile/` |
| **Ch. 14 – LunarLander-v3 : Continuous-State Control and the Curse of Dimensionality** | `LunarLander-v3` | Discretization & state aggregation bridging toward Deep RL | `case_studies/02_lunar_lander_drone/` |

---

## Case Studies

1. **Last-Mile Dispatch (Taxi-v3)**  
   _Real-world lens_: courier / ride-hailing pickup–drop logistics in a small grid.  
   _Env_: `gymnasium.envs.toy_text.taxi.TaxiEnv` (discrete).  
   _Algos_: Q-Learning, SARSA (ε-greedy, decaying ε).  
   _Metrics_: average return vs episodes, success rate, steps per episode.

2. **Drone Landing Guidance (LunarLander-v3)**  
   _Real-world lens_: autonomous drone landing under stochastic dynamics.  
   _Env_: `LunarLander-v3` (Box2D, discrete).  
   _Algos_: Q-Learning, SARSA (tabular baseline) + DQN scaffold.  
   _Metrics_: average return vs episodes, solved rate (≥ 200), crash rate.

> **Why these?** They map cleanly to real operations (dispatch & landing control) and trace a pedagogical line from **MDPs & Bellman** → **TD Control** → **Function Approximation (Deep RL)**.

---

## ⚙️ Quickstart

```bash
# Python 3.10 + recommended
pip install -r requirements.txt

# Case Study 1: Taxi (Last-Mile Dispatch)
python case_studies/01_taxi_last_mile/train_q_learning.py --episodes 4000
python case_studies/01_taxi_last_mile/plot_returns.py

# Case Study 2: Lunar Lander (Drone Landing)
python case_studies/02_lunar_lander_drone/train_q_learning.py --episodes 5000
python case_studies/02_lunar_lander_drone/plot_returns.py
```

Artifacts appear under each study’s `runs/` folder (`.csv`, `.npy`, plots).  
⏱ Typical runtime: ~5 min @ CPU for Taxi-v3, ~10 min for LunarLander-v3.

---

## 📂 Repo Layout

```
open-gym-case-studies/
├─ README.md
├─ requirements.txt
├─ Makefile
├─ common/
│  ├─ utils.py
│  └─ plotting.py
├─ case_studies/
│  ├─ 00_frozenlake_navigation/  # Tabular SARSA & Q-Learning
│  ├─ 01_taxi_last_mile/         # Q-Learning, SARSA, evaluation + plots
│  └─ 02_lunar_lander_drone/     # Tabular baselines + DQN scaffold
├─ tests/                         # pytest smoke tests
└─ .github/workflows/ci.yml       # CI with caching + artifacts
```

---

## 🧩 Deep RL — Expansion Path

- **DQN (discrete control)** → replace Q-table with neural Q-network, add replay buffer & target network.  
- **Stability extensions:** Double DQN, prioritized replay.  
- **Policy Gradients:** Actor–Critic / A2C / PPO.  
- **Continuous control:** `LunarLanderContinuous-v2` with DDPG / TD3 / SAC.

> Optional: `pip install torch` to activate DQN scaffold in `02_lunar_lander_drone/dqn_scaffold.py`.

---

## 🧪 Reproducibility & Evaluation

- Deterministic seeding (`--seed`, NumPy + Gymnasium).  
- CSV logs → `episode, return, epsilon, steps, success`.  
- Plots → moving average return with 95 % CI (standard error).  
- Pytest smoke tests ensure training-loop integrity.

---

## 👩‍💻 Contributing (for RAs / students)

See [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).  
Open PRs with small, reviewable chunks — CI runs on every PR.

---

## 📚 Citation

If you use this code or the accompanying book in research or teaching, please cite:

**Book (forthcoming):**
```bibtex
@book{baride2025rlexplained,
  author    = {Srikanth Baride and Rodrigue Rizk and KC Santosh},
  title     = {Reinforcement Learning Explained},
  publisher = {CRC Press | Taylor \& Francis Group},
  year      = {2025},
  isbn      = {9781041252993},
  note      = {Accepted for publication; preprint available at \url{https://github.com/srikanthbaride/rl-explained-preprint}}
}
```

---

## 🪪 License

Educational use; adapt as needed for your textbook’s distribution terms.

---

## © Copyright & Attribution

© 2025 Dr. Srikanth Baride, Dr. Rodrigue Rizk, and Prof. KC Santosh.  
All rights reserved. This repository accompanies the textbook:

> **_Reinforcement Learning Explained_**  
> CRC Press / Taylor & Francis Group, 2025.

The source code and instructional content were developed by the authors for educational and research purposes.  
Algorithms (Q-Learning, SARSA, DQN scaffolds) are implemented from scratch following established formulations.

This project uses **Gymnasium (OpenAI Gym)** under the MIT License; no third-party proprietary code is included.

**Permitted use:** Educators and students may reproduce, modify, and distribute this material for non-commercial educational purposes, with proper credit and citation.

For permissions beyond classroom use, contact the authors via the University of South Dakota AI Research Lab.
