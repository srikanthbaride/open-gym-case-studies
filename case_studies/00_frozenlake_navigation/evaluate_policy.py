import argparse, numpy as np
from make_env import make_env

def run_episode(env, policy, render=False):
    s, info = env.reset()
    done, G = False, 0.0
    while not done:
        a = int(policy[s])
        s, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        G += r
    return G

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_path", required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--slippery", type=int, default=1)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    policy = np.load(args.policy_path)
    env = make_env(bool(args.slippery), render_mode="human" if args.render else None, seed=args.seed)
    rets = [run_episode(env, policy, render=args.render) for _ in range(args.episodes)]
    mean_ret = float(np.mean(rets))
    succ = float(np.mean((np.array(rets) > 0).astype(float)))
    print(f"mean_return={mean_ret:.3f}  success_rate={succ:.3f}")
if __name__ == "__main__":
    main()
