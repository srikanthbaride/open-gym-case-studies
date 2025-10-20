.PHONY: taxi-smoke lander-smoke test

taxi-smoke:
\tpython case_studies/01_taxi_last_mile/train_q_learning.py --episodes 50 || true

lander-smoke:
\tpython case_studies/02_lunar_lander_drone/train_q_learning.py --episodes 50 || true

test:
\tpytest -q
