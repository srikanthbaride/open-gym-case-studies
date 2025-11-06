#!/usr/bin/env python3
"""
plot_curves.py — Convert training CSV logs into publication-ready PDFs and summaries.

Usage:
  python scripts/plot_curves.py \
    --logs path/to/run1.csv path/to/run2.csv \
    --window 50 \
    --out outdir

Assumptions:
- CSV has at least: episode, return
- Optional columns: algo, env, steps, epsilon, success, seed, ...
- Will group multiple logs on a single figure.
"""

import argparse
import os
import math
import pandas as pd
import matplotlib.pyplot as plt

def moving_average(x, w):
    if w <= 1:
        return x
    return x.rolling(window=w, min_periods=1, center=False).mean()

def ci95(series):
    # mean ± 1.96 * (std / sqrt(n))
    s = series.dropna()
    n = len(s)
    if n == 0:
        return (float('nan'), float('nan'))
    m = s.mean()
    se = s.std(ddof=1) / math.sqrt(n) if n > 1 else 0.0
    return (m, 1.96 * se)

def infer_label(df, path):
    algo = None
    env = None
    for cand in ['algo', 'algorithm']:
        if cand in df.columns:
            algo = str(df[cand].iloc[0])
            break
    for cand in ['env', 'environment']:
        if cand in df.columns:
            env = str(df[cand].iloc[0])
            break
    base = os.path.splitext(os.path.basename(path))[0]
    if algo and env:
        return f"{algo} ({env})"
    if algo:
        return f"{algo}"
    return base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True, help="CSV files")
    ap.add_argument("--window", type=int, default=50, help="moving average window")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    curves = []
    summaries = []

    for p in args.logs:
        df = pd.read_csv(p)
        if 'episode' not in df.columns or 'return' not in df.columns:
            raise ValueError(f"{p} must contain columns: episode, return")
        lab = infer_label(df, p)
        df = df.sort_values('episode')
        df['ma_return'] = moving_average(df['return'], args.window)
        curves.append((lab, df))

        # Evaluation summary if available (e.g., last N episodes flagged), else simple overall stats
        # We'll compute mean ± 95% CI over the last 50 episodes as a proxy.
        tail = df.tail(50)
        m, ci = ci95(tail['return'])
        succ = None
        if 'success' in tail.columns:
            try:
                succ = 100.0 * (tail['success'].astype(float).mean())
            except Exception:
                succ = None
        summaries.append({
            "label": lab,
            "file": os.path.basename(p),
            "episodes": int(df['episode'].max()) if len(df)>0 else 0,
            "mean_return_last50": m,
            "ci95_last50": ci,
            "success_rate_last50_pct": succ
        })

    # Plot — one figure per set (single axes, default colors)
    plt.figure(figsize=(8, 4.5))
    for lab, df in curves:
        plt.plot(df['episode'], df['ma_return'], label=lab)
    plt.xlabel("Episode")
    plt.ylabel(f"Average return (MA window={args.window})")
    plt.title("Learning Curves")
    plt.grid(True, which="both", linestyle="-", linewidth=0.5, alpha=0.6)
    plt.legend(loc="lower right")
    # Output file name selection by env if possible
    # Try to guess from first label
    out_pdf = os.path.join(args.out, "learning_curves.pdf")
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()

    # Write summary CSV
    out_csv = os.path.join(args.out, "eval_summary.csv")
    pd.DataFrame(summaries).to_csv(out_csv, index=False)

    print(f"Wrote: {out_pdf}")
    print(f"Wrote: {out_csv}")

if __name__ == "__main__":
    main()
