import os, numpy as np, subprocess, sys

def test_train_and_artifacts():
    cmd = [sys.executable, "case_studies/01_taxi_last_mile/train_q_learning.py", "--episodes", "10"]
    subprocess.check_call(cmd)
    assert os.path.exists("runs/q_learning/train_log.csv")
    assert os.path.exists("runs/q_learning/q_table.npy")
    Q = np.load("runs/q_learning/q_table.npy")
    assert Q.ndim == 2  # states x actions
