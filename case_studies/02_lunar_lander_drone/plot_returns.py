import os
from common.plotting import plot_returns

if __name__ == "__main__":
    for sub in ["runs/q_learning", "runs/sarsa", "runs/dqn"]:
        csv_path = os.path.join(sub, "train_log.csv")
        if os.path.exists(csv_path):
            out_png = os.path.join(sub, "returns_ma.png")
            plot_returns(csv_path, out_png, window=100)
            print(f"Saved plot to {out_png}")
