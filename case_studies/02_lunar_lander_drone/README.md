# 🛰️ Case Study 2 — Drone Landing (LunarLander-v3)

**Goal:** Apply tabular TD control to a **continuous** state space via discretization; examine bin-size effects.

## Environment
- Gymnasium: `LunarLander-v3` (Box2D)
- Observation: `(x, y, v_x, v_y, θ, ω, c_1, c_2)` (continuous + contacts)
- Actions: `{No thrust, Left, Main, Right}`
- Episode cap: typically 1000 steps (`env.spec.max_episode_steps`)
- Install extras: `pip install "gymnasium[box2d]"` (may need `swig` installed)

## Quick start
```bash
python train_lander.py --algo qlearning --bins 10 --episodes 5000 --seed 123 --log logs/lander_qlearning_b10.csv
python train_lander.py --algo sarsa     --bins 10 --episodes 5000 --seed 123 --log logs/lander_sarsa_b10.csv
```

### Discretization hint
Clip and digitize each continuous variable:
```python
# Example: vx in [-2, 2] into 10 bins → 0..9
bin_vx = int(np.clip((vx + 2.0) / 4.0 * 10, 0, 9))
```
Total discrete states: product of bins over variables (curse of dimensionality).

## Plot learning curves
```bash
python ../../scripts/plot_curves.py   --logs logs/lander_qlearning_b10.csv logs/lander_sarsa_b10.csv   --window 50   --out figures
```

Outputs:
- `figures/lander_learning_curves.pdf` — moving-average episode returns
- `figures/lander_eval_summary.csv` — evaluation mean ± 95% CI, success rate

## Notes
- Lower learning rate or finer bins can reduce oscillations near touchdown
- Report truncated episodes separately (TimeLimit wrapper)
