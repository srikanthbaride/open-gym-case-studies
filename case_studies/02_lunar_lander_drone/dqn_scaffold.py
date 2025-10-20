# Minimal DQN scaffold to extend: replace Q-table with NN approximator.
# Recommended: PyTorch (torch), ReplayBuffer, Target network, Double DQN.
import argparse, os, csv
import numpy as np
import gymnasium as gym

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--outdir", type=str, default="runs/dqn")
    # Add hyperparameters: lr, gamma, eps, target_update, buffer_size, batch_size, etc.
    args = parser.parse_args()

    env = gym.make("LunarLander-v3")
    obs, info = env.reset(seed=args.seed)

    # TODO: import torch, define QNetwork, ReplayBuffer, epsilon schedule, optimizer, etc.
    # This file shows the interface; classical baselines already run end-to-end.

    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "train_log.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode","return","epsilon","steps"])
        # Implement DQN training loop here.

    print("DQN scaffold written. Extend with torch to train.")

if __name__ == "__main__":
    main()
