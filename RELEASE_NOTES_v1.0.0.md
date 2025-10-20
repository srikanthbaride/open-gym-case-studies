# CRC Press Companion Release — v1.0.0
**Date:** 2025-10-20

This release publishes the teaching companion for the textbook:

> **_Reinforcement Learning Explained_**  
> Dr. Srikanth Baride · Dr. Rodrigue Rizk · Prof. K.C. Santosh  
> CRC Press / Taylor & Francis Group, 2025

## Highlights
- **Case Study A — Taxi-v3 (Last-Mile Dispatch):** Q-learning & SARSA (ε-greedy, GLIE), CSV logs, moving-average plots.
- **Case Study B — LunarLander-v3 (Drone Landing):** discretized tabular baselines + **DQN scaffold** to extend with PyTorch.
- **Reproducibility:** deterministic seeds, pytest smoke tests, Makefile tasks.
- **CI:** GitHub Actions with pip cache, artifact upload, and `swig` install for Box2D builds.
- **Docs:** README with badges, CRC-compatible Copyright & Attribution.

## Installation
```bash
pip install -r requirements.txt
```

## Quickstart
```bash
# Taxi (Last-Mile Dispatch)
python case_studies/01_taxi_last_mile/train_q_learning.py --episodes 4000
python case_studies/01_taxi_last_mile/plot_returns.py

# Lunar Lander (Drone Landing)
python case_studies/02_lunar_lander_drone/train_q_learning.py --episodes 5000
python case_studies/02_lunar_lander_drone/plot_returns.py
```

## Notes
- The Lander example uses `LunarLander-v3`. If training on CI, ensure `swig` is available (already in workflow).
- For Deep RL, implement DQN in `case_studies/02_lunar_lander_drone/dqn_scaffold.py` (PyTorch suggested).

## Citation
Please cite the textbook if you use this repository for teaching or research.
