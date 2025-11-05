# Case Study 0 — FrozenLake-v1 (Tabular SARSA & Q-Learning)

A minimal grid-world with stochastic slippage (`is_slippery=True`) and a deterministic variant for controlled experiments.
Designed to match the style of your Part III case studies.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install gymnasium numpy
# optional: pip install matplotlib
```

### Train (Q-Learning)
```bash
python train_frozenlake.py --algo qlearning --episodes 5000 --slippery 1 --seed 123
```

### Train (SARSA)
```bash
python train_frozenlake.py --algo sarsa --episodes 5000 --slippery 0 --seed 123
```

### Evaluate a saved policy
```bash
python evaluate_policy.py --policy_path runs/qlearning_slip1_seed123/policy.npy --episodes 50
```

**Logged fields**: `episode, return, steps, epsilon, success, seed` (printed summaries every 500 episodes).

**Notes**
- Deterministic experiments: `--slippery 0` (good for exact optimal paths).
- Stochastic experiments: `--slippery 1` (good for SARSA vs Q-learning under uncertainty).
- We save both `Q.npy` and the greedy `policy.npy` for reproducible evaluation.

**Repo placement**: put this folder under `case_studies/00_frozenlake_navigation/`.
