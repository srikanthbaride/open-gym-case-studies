import argparse, os, csv
import sys
import numpy as np
import gymnasium as gym
import os, sys
sys.path.append(os.path.abspath("."))
from common.utils import set_global_seeds, epsilon_greedy_action, linear_decay

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--epsilon_decay_episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--outdir", type=str, default="runs/sarsa")
    args = parser.parse_args()

    set_global_seeds(args.seed)
    env = gym.make("Taxi-v3")
    obs, info = env.reset(seed=args.seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions), dtype=np.float32)

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
            a = epsilon_greedy_action(Q[obs], epsilon)

            while not (done or truncated):
                next_obs, r, done, truncated, info = env.step(a)
                total_r += r
                steps += 1

                next_a = epsilon_greedy_action(Q[next_obs], epsilon)
                td_target = r + args.gamma * Q[next_obs, next_a] * (not (done or truncated))
                td_error = td_target - Q[obs, a]
                Q[obs, a] += args.alpha * td_error

                obs, a = next_obs, next_a

            success = int(info.get("success", 0)) if isinstance(info, dict) else 0
            writer.writerow([ep, total_r, epsilon, steps, success])

    np.save(os.path.join(args.outdir, "q_table.npy"), Q)
    print(f"Saved: {csv_path}")

if __name__ == "__main__":
    main()
