import argparse, os, numpy as np
from make_env import make_env
from utils import Sched

def choose_action(Q, s, eps, nA, rng):
    return rng.integers(nA) if rng.random() < eps else int(np.argmax(Q[s]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["qlearning","sarsa"], required=True)
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--slippery", type=int, default=1)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--render_eval", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    env = make_env(bool(args.slippery), render_mode=None, seed=args.seed)

    nS = env.observation_space.n
    nA = env.action_space.n
    Q = np.zeros((nS, nA), dtype=np.float32)

    sched = Sched()
    logdir = f"runs/{args.algo}_slip{args.slippery}_seed{args.seed}"
    os.makedirs(logdir, exist_ok=True)

    returns = []
    for ep in range(args.episodes):
        s, info = env.reset()
        done = False
        G = 0.0
        eps = sched.epsilon(ep)
        a = choose_action(Q, s, eps, nA, rng) if args.algo == "sarsa" else None

        while not done:
            if args.algo == "qlearning":
                a = choose_action(Q, s, eps, nA, rng)
            s2, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            G += r

            if args.algo == "qlearning":
                td_target = r + args.gamma * (0.0 if done else np.max(Q[s2]))
            else:  # SARSA
                a2 = choose_action(Q, s2, eps, nA, rng)
                td_target = r + args.gamma * (0.0 if done else Q[s2, a2])

            Q[s, a] += args.alpha * (td_target - Q[s, a])
            if args.algo == "sarsa" and not done:
                s, a = s2, a2
            else:
                s = s2

        returns.append(G)
        if (ep+1) % 500 == 0:
            avg = np.mean(returns[-100:]) if len(returns) >= 100 else np.mean(returns)
            print(f"ep={ep+1} avg_return(last100)={avg:.3f} eps={eps:.3f}")

    np.save(os.path.join(logdir, "Q.npy"), Q)
    # Greedy policy for evaluation
    policy = np.argmax(Q, axis=1).astype(np.int64)
    np.save(os.path.join(logdir, "policy.npy"), policy)

if __name__ == "__main__":
    main()
