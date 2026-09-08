# 5.9.1. MARIN — read these learning results

Comparator: **constant_3**.
Positive interval-reward and coverage differences favor the learned policy. Negative energy differences save energy.
Intervals resample whole world-seed blocks and condition on these specific trained policies. They do not establish a population-level PPO advantage.

- ppo_seed0: reward difference -0.00814, 95% interval [-0.01253, -0.00391]: negative on these missions.
- ppo_seed1: reward difference +0.00000, 95% interval [+0.00000, +0.00000]: inconclusive.
- ppo_seed2: reward difference -0.00814, 95% interval [-0.01253, -0.00391]: negative on these missions.

Check `scenario_means.csv` and the coverage/energy panels before deciding that a reward gain is useful.
Development results may guide tuning. After inspecting a final test set, reserve new world seeds for the next revision.
