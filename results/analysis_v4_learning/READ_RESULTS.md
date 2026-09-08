# 5.9.1. MARIN — read these learning results

Comparator: **constant_3**.
Positive interval-reward and coverage differences favor the learned policy. Negative energy differences save energy.
Intervals resample whole world-seed blocks and condition on these specific trained policies. They do not establish a population-level PPO advantage.

- ppo_seed0: reward difference -0.00911, 95% interval [-0.01114, -0.00673]: negative on these missions.
  It selected action 1 at every evaluated decision. This is constant behavior on this test set, not evidence of state-dependent adaptation.
- ppo_seed1: reward difference +0.00000, 95% interval [+0.00000, +0.00000]: identical on these missions.
  It selected action 3 at every evaluated decision. This is constant behavior on this test set, not evidence of state-dependent adaptation.
- ppo_seed2: reward difference -0.00911, 95% interval [-0.01114, -0.00673]: negative on these missions.
  It selected action 1 at every evaluated decision. This is constant behavior on this test set, not evidence of state-dependent adaptation.

Check `scenario_means.csv` and the coverage/energy panels before deciding that a reward gain is useful.
Development results may guide tuning. After inspecting a final test set, reserve new world seeds for the next revision.
