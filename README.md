# 📦 Open Gym Case Studies — Logistics & Robotics (classical RL + Deep RL hooks)

> _Textbook-aligned case studies for teaching Q-learning/SARSA and extending to DQN/Actor–Critic._

<!-- Badges (replace srikanthbaride/open-gym-case-studies after you push) -->
[![CI](https://github.com/srikanthbaride/open-gym-case-studies/actions/workflows/ci.yml/badge.svg)](https://github.com/srikanthbaride/open-gym-case-studies/actions/workflows/ci.yml)
[![Last Commit](https://img.shields.io/github/last-commit/srikanthbaride/open-gym-case-studies)](https://github.com/srikanthbaride/open-gym-case-studies/commits/main)
[![License](https://img.shields.io/badge/License-Educational-lightgrey.svg)](#license)

This repository contains **two real-world-motivated case studies** implemented with **Gymnasium (OpenAI Gym)**, aligned with the reinforcement learning chapters (Bandits → MC → TD → Q-learning/SARSA) in my textbook. Each case study includes **reproducible training scripts**, **CSV logs**, **plots**, and **pytest smoke tests**. Extensions are scaffolded for **Deep RL (DQN / Actor–Critic)**.

---

## Case Studies

1. **Last-Mile Dispatch (Taxi-v3)**  
   _Real-world lens_: courier/ride-hailing pickup–drop logistics in a small city grid.  
   _Env_: `gymnasium.envs.toy_text.taxi.TaxiEnv` (discrete).  
   _Algos_: Q-learning, SARSA (ε-greedy, decaying ε).  
   _Metrics_: average return vs. episodes, success rate, steps/episode.

2. **Drone Landing Guidance (LunarLander-v3)**  
   _Real-world lens_: autonomous drone landing on a pad under stochastic dynamics.  
   _Env_: `LunarLander-v3` (discrete; Box2D).  
   _Algos_: Q-learning, SARSA (tabular baseline), with **DQN** scaffold to expand.  
   _Metrics_: average return vs. episodes, solved-rate (≥ 200), crash rate.

> **Why these?** They map cleanly to real operations (dispatch & landing control) and trace a pedagogical line from **MDPs & Bellman** → **TD control** → **function approximation (Deep RL)**.

---

## Quickstart

```bash
# Python 3.10+ recommended
pip install -r requirements.txt

# Case Study 1: Taxi (Last-Mile Dispatch)
python case_studies/01_taxi_last_mile/train_q_learning.py --episodes 4000
python case_studies/01_taxi_last_mile/plot_returns.py

# Case Study 2: Lunar Lander (Drone Landing)
python case_studies/02_lunar_lander_drone/train_q_learning.py --episodes 5000
python case_studies/02_lunar_lander_drone/plot_returns.py
```

Artifacts live under each case study’s `runs/` folder (CSV logs, `.npy` Q-tables, plots).

---

## Repo Layout

```
open-gym-case-studies/
├─ README.md
├─ requirements.txt
├─ Makefile
├─ common/
│  ├─ utils.py
│  └─ plotting.py
├─ case_studies/
│  ├─ 01_taxi_last_mile/        # Q-learning, SARSA, evaluate, plots
│  └─ 02_lunar_lander_drone/    # tabular baselines + DQN scaffold
├─ tests/                        # pytest smoke tests
└─ .github/workflows/ci.yml      # CI with caching + artifacts
```

---

## Makefile shortcuts

```bash
# Train short smoke runs for CI
make taxi-smoke
make lander-smoke

# Run tests locally
make test
```

---

## Deep RL — Expansion Path

- **DQN (discrete control)**: replace Q-table with a neural Q-network; add replay buffer, target network; ε-greedy.
- **Stability**: Double DQN, prioritized replay.
- **Policy Gradients**: Actor–Critic / A2C / PPO.
- **Continuous control**: `LunarLanderContinuous-v2` with **DDPG / TD3 / SAC**.

> Optional: `pip install torch` to implement DQN in `02_lunar_lander_drone/dqn_scaffold.py`.

---

## Reproducibility & Evaluation

- Deterministic seeding (`--seed`, NumPy + Gymnasium).
- CSV logs: `episode,return,epsilon,steps,success`.
- Plots: moving average return with 95% CI (via standard error).
- Pytest smoke tests ensure training loop and interface invariants.

---

## Contributing (for RAs / students)

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Open PRs with small, reviewable chunks. CI runs on every PR.

---

## 📚 How to Cite

If you use this code or the accompanying book in your research or teaching, please cite:

**Book (forthcoming):**
```bibtex
@book{baride2025rlexplained,
  author    = {Srikanth Baride and Rodrigue Rizk and K. C. Santosh},
  title     = {Reinforcement Learning Explained},
  publisher = {CRC Press | Taylor \& Francis Group},
  year      = {2025},
  isbn      = {9781041252993},
  note      = {Accepted for publication; preprint available at \url{https://github.com/srikanthbaride/rl-explained-preprint}}
}

```


---

## License

Educational use; adapt as needed for your textbook’s distribution terms.


---

## Copyright & Attribution

© 2025 Dr. Srikanth Baride, Dr. Rodrigue Rizk, and Prof. K.C. Santosh.  
All rights reserved. This repository accompanies the textbook:

> **_Reinforcement Learning Explained_**  
> CRC Press / Taylor & Francis Group, 2025.

The source code and instructional content in this repository were developed
exclusively by the authors for educational and research purposes.
Algorithms (e.g., Q-Learning, SARSA, DQN scaffolds) follow
well-known formulations from the scientific literature and are implemented
from scratch.  

This project uses **Gymnasium (OpenAI Gym)** environments under the
MIT License.  No proprietary or third-party copyrighted code is included.

**Permitted use:**  
Educators and students may reproduce, modify, and distribute this material
for non-commercial educational purposes, provided that proper credit is given
to the above authors and the textbook citation is included.

For questions or permissions beyond classroom use, please contact  
the authors through the University of South Dakota AI Research Lab.

