import argparse, os, json
import numpy as np
import gymnasium as gym

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q_path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--discrete_bins", action="store_true", help="Required because Q-table uses discretized states")
    args = parser.parse_args()

    if not args.discrete_bins:
        raise SystemExit("Use --discrete_bins for tabular Lander evaluation.")

    Q = np.load(args.q_path)

    def build_bins():
        import numpy as np
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
    def discretize(obs, bins):
        import numpy as np
        d = []
        for i, b in enumerate(bins):
            d.append(int(np.digitize(obs[i], b)))
        return tuple(d)

    bins = build_bins()
    env = gym.make("LunarLander-v3")
    total = 0.0
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        truncated = False
        ret = 0.0
        s = discretize(obs, bins)
        while not (done or truncated):
            a = int(np.argmax(Q[s]))
            obs, r, done, truncated, info = env.step(a)
            s = discretize(obs, bins)
            ret += r
        total += ret
    print(json.dumps({"avg_return": total/args.episodes}))

if __name__ == "__main__":
    main()
