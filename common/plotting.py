import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def plot_returns(csv_path: str, out_path: str, window: int = 50):
    df = pd.read_csv(csv_path)
    r = df["return"].to_numpy()
    if len(r) >= window:
        ma = moving_average(r, window)
        x = np.arange(len(ma)) + window
        plt.figure()
        plt.plot(x, ma, label=f"Moving Avg Return (w={window})")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
