# 🚕 Case Study 1 — Last-Mile Dispatch (Taxi-v3)

**Goal:** Compare **Q-Learning** and **SARSA** in a small discrete MDP with reward shaping.

## Environment
- Gymnasium: `Taxi-v3`
- Observation: `Discrete(500)` (row, col, passenger, destination)
- Actions: `{N, S, E, W, Pickup, Dropoff}`
- Rewards: `+20` (successful drop-off), `-10` (illegal), `-1` per step
- Episode cap: typically 200 steps (`env.spec.max_episode_steps`)

## Quick start
Train with a linear ε-decay schedule (GLIE-like):

```bash
# Examples (replace with your actual training script names/options)
python train_taxi.py --algo qlearning --episodes 4000 --seed 123 --log logs/taxi_qlearning.csv
python train_taxi.py --algo sarsa     --episodes 4000 --seed 123 --log logs/taxi_sarsa.csv
```

**Seeding pattern** (use in your training scripts):
```python
import gymnasium as gym, numpy as np, random
SEED = 123
env = gym.make("Taxi-v3")
obs, info = env.reset(seed=SEED)
env.action_space.seed(SEED)
random.seed(SEED); np.random.seed(SEED)
```

## Plot learning curves
```bash
python ../../scripts/plot_curves.py   --logs logs/taxi_qlearning.csv logs/taxi_sarsa.csv   --window 50   --out figures
```

Outputs:
- `figures/taxi_learning_curves.pdf` — moving-average episode returns
- `figures/taxi_eval_summary.csv` — evaluation mean ± 95% CI, success rate

## Evaluation protocol
- Evaluate greedy policy `ε=0` for `N=50` episodes
- Report mean return ± 95% CI; track success and timeouts separately

## Notes
- Use `terminated or truncated` for `done`
- Consider a stronger penalty for illegal actions in an ablation (`-20` vs `-10`)
