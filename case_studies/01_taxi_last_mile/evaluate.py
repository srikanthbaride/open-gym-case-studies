import argparse, os, json
import numpy as np
import gymnasium as gym

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q_path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=999)
    args = parser.parse_args()

    Q = np.load(args.q_path)
    env = gym.make("Taxi-v3")
    total = 0.0
    success = 0
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        truncated = False
        ret = 0.0
        while not (done or truncated):
            a = int(np.argmax(Q[obs]))
            obs, r, done, truncated, info = env.step(a)
            ret += r
        total += ret
        success += int(info.get("success", 0)) if isinstance(info, dict) else 0
    print(json.dumps({"avg_return": total/args.episodes, "success_rate": success/args.episodes}))

if __name__ == "__main__":
    main()
