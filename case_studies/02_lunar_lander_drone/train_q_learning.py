import argparse, os, csv
import sys
import numpy as np
import gymnasium as gym
import os, sys
sys.path.append(os.path.abspath("."))
from common.utils import set_global_seeds, epsilon_greedy_action, linear_decay

def discretize(obs, bins):
    d = []
    for i, b in enumerate(bins):
        d.append(int(np.digitize(obs[i], b)))
    return tuple(d)

def build_bins():
    return [
        np.linspace(-1.5, 1.5, 8),
        np.linspace(-.5, 1.5, 8),
        np.linspace(-2.0, 2.0, 8),
        np.linspace(-2.0, 2.0, 8),
        np.linspace(-1.5, 1.5, 8),
        np.linspace(-2.0, 2.0, 8),
        np.array([0.5]),
        np.array([0.5])
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--epsilon_decay_episodes", type=int, default=4500)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--outdir", type=str, default="runs/q_learning")
    args = parser.parse_args()

    set_global_seeds(args.seed)
    env = gym.make("LunarLander-v3")
    obs, info = env.reset(seed=args.seed)

    n_actions = env.action_space.n
    bins = build_bins()
    q_shape = tuple(len(b)+1 for b in bins) + (n_actions,)
    Q = np.zeros(q_shape, dtype=np.float32)

    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "train_log.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode","return","epsilon","steps","success"])

        for ep in range(1, args.episodes + 1):
            obs, info = env.reset(seed=args.seed + ep)
            done = False
            truncated = False
            total_r = 0.0
            steps = 0
            epsilon = linear_decay(args.epsilon_start, args.epsilon_end, ep, args.epsilon_decay_episodes)

            state = discretize(obs, bins)
            while not (done or truncated):
                a = epsilon_greedy_action(Q[state], epsilon)
                next_obs, r, done, truncated, info = env.step(a)
                next_state = discretize(next_obs, bins)

                best_next = np.max(Q[next_state])
                td_target = r + args.gamma * best_next * (not (done or truncated))
                td_error = td_target - Q[state + (a,)]
                Q[state + (a,)] += args.alpha * td_error

                state = next_state
                total_r += r
                steps += 1

            success = int((info or {}).get("lander_contact", 0))
            writer.writerow([ep, total_r, epsilon, steps, success])

    np.save(os.path.join(args.outdir, "q_table.npy"), Q)
    print(f"Saved: {csv_path}")

if __name__ == "__main__":
    main()
