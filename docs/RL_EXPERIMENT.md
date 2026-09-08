> **Historical v3 workflow and results.** For the current Windows scripts, three PPO pilots and all-nine-settings comparison, start with [WINDOWS_RL_START.md](WINDOWS_RL_START.md).

# Residual learning: working setup and honest evaluation

The v3 learner selects **how far to propose moving** and **how much to penalize
movement energy**. The economic planner and simulated vessel controller still
execute the motion. It does not learn raw thrust commands.

## What actually ran

| Experiment | Completed | Interpretation |
|---|---|---|
| Native linear-softmax REINFORCE | 50 episodes, seed 0 | An initial learner, not an optimized algorithm |
| Native held-out comparison | 18 missions per policy; six paired seeds across three scenarios | Coverage −0.805 points and fleet energy −54.0 Wh vs neutral economic; evaluation utility also decreased |
| Native continuation | Five additional episodes, ending at 55 | Checkpoint continuation works; exact episode-boundary continuation also has a regression test |
| PPO environment checker | Passed; complete 40-step mission | Observation/action API works |
| PPO training + continuation | 512 + 128 = 640 steps, seed 0 | Working optimization, save and load path |
| PPO saved-policy evaluation | Two one-hour storm missions | Execution check, too small to claim learning improvement |

The native evaluation used `results/native_residual_seed0/checkpoint_50.npz`.
The evaluation manifest records the original filename `checkpoint.npz` at the
time of that evaluation; its content hash matches the preserved `_50` file.
The current `checkpoint.npz` has 55 episodes and is deliberately a different file.
Do not accidentally reproduce the reported evaluation using the later checkpoint.

## Option A: no extra neural-network dependencies

Use the base Python environment from the main guide. Train a fresh policy:

```bat
python -m usv_research.native_pg train --config configs/train.json --out results/my_native_seed0 --episodes 100 --seed 0
```

Continue for **100 additional episodes**, keeping the same seed and configuration:

```bat
python -m usv_research.native_pg train --config configs/train.json --out results/my_native_seed0 --episodes 100 --seed 0 --resume
```

Evaluate the policy and its neutral action on the same fresh missions:

```bat
python -m usv_research.native_pg evaluate --config configs/short_radio.json --checkpoint results/my_native_seed0/checkpoint.npz --out results/my_native_test --seeds 20 --seed-start 12000
python -m usv_research.native_pg evaluate --config configs/short_radio.json --checkpoint results/my_native_seed0/checkpoint.npz --out results/my_neutral_test --seeds 20 --seed-start 12000 --constant-action 4
```

Use a new evaluation output folder for every different checkpoint. Native
evaluation writes scenario/seed filenames; reusing the folder overwrites them.
Copy a checkpoint before continuing training if you want to preserve its identity.

The saved state includes policy weights, Adam moments, return baseline, completed
episode count and random-generator state. An interrupted unfinished episode is
recomputed after the last committed episode. The regression test checks exact
continuation at that boundary, not bitwise identity across all operating systems.

## Option B: optional PPO

Create a separate environment so the core study stays easy to reproduce:

```bat
conda create -n usv-ppo python=3.12 -y
conda activate usv-ppo
python -m pip install -r requirements-tested.txt
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-rl-tested.txt
python -m pip install -e . --no-deps
python -m usv_research.ppo check --config configs/train.json --out results/my_ppo_check
```

The exact exercised stack was Python 3.12.13, CPU PyTorch 2.8.0, Gymnasium 1.3.0
and Stable-Baselines3 2.9.0. The CPU wheel command targets Windows/Linux. On macOS,
install `torch==2.8.0` from the regular package index instead of that CPU index.
No GPU is required for the small policy network; simulation is the main cost.

Start with a short timing/training run:

```bat
python -m usv_research.ppo train --config configs/train.json --out results/my_ppo_seed0 --steps 2048 --seed 0 --workers 2
```

Then continue for approximately 20,000 **additional** environment steps:

```bat
python -m usv_research.ppo train --config configs/train.json --out results/my_ppo_seed0 --steps 20000 --seed 0 --workers 2 --resume results/my_ppo_seed0/final.zip
```

PPO rounds requested steps up to full rollout batches of `128 × workers`.
`training_timing.jsonl` records actual added steps, total steps and wall time.
Periodic checkpoints are written about every 10,000 total environment steps;
`final.zip` is written on successful completion. Resume restores the model and
optimizer, but resets environment trajectories and their random streams. It is
not an exact bitwise continuation. Keep the associated `manifest.json` beside
the checkpoint. Changing the causal source/configuration/seed/stack is rejected
for resuming an old experiment.

Evaluate on fresh missions:

```bat
python -m usv_research.ppo evaluate --config configs/short_radio.json --checkpoint results/my_ppo_seed0/final.zip --out results/my_ppo_test --seeds 20 --seed-start 12000
```

Use the native constant-action evaluation above with the supplied native
checkpoint to obtain a matching neutral economic baseline on seeds 12000–12019.
The checkpoint weights are unused for a constant action; they supply the
observation-history metadata. Both learners use the same physical mission adapter.

## Observation, actions and reward

Each observation has `13 × N + 5` entries: received position, sensor-belief
summaries, cache age, energy, previous waypoint error, noisy current weather,
wind direction, mission progress and previous residual action. Four frames
provide 332 inputs when N = 6. No future realized channel, hidden sensor health,
or free true coverage field is exposed. The native policy is linear with a
softmax; PPO uses a two-layer 128-unit MLP. Neither is an implemented recurrent
neural network.

Action `a = 3i + j` chooses step `[60, 120, 180][i]` metres and energy price
`[0.0125, 0.025, 0.05][j]`. Thus action 4 reproduces the ordinary economic
controller. A paired trajectory test verifies that equivalence.

Each interval reward is actual delivered coverage minus
`0.025 × incremental fleet energy / (N × 100 W × interval duration / 3600)`.
Reward is returned by the simulated environment, as in ordinary RL. The learner
cannot read unobserved state merely because reward depends on mission outcomes.
Training randomizes nominal radio and radar parameters and scenario choice.
The planner still receives those configured nominal values: this is randomized
conditions, not proof of robustness to completely unknown model parameters.

## How to test whether learning adds anything

1. Develop on seeds 200–299 and evaluate all **nine constant residual actions**
   there using `--constant-action 0` through `8`. Select the best constant using
   a declared coverage–energy criterion. Comparing only with action 4 is a pilot.
2. Train at least three independent policies (`--seed 0`, `1`, `2`) in separate
   output folders. Training randomness and test-mission randomness are different
   sources of variation; do not pool them as though all observations were independent.
3. Freeze the checkpoint selection rule before opening final test seeds.
   Compare every policy with neutral economic and the selected constant on the
   same 20–30 fresh seeds, across hardware profiles and scenarios.
4. Report delivered coverage, energy, outage/tail behavior, command age and
   uncertainty. Compare four frames vs one and domain randomization on/off only
   as declared follow-up experiments. Do not claim superiority from training return.

The general episode analysis CLI summarizes arbitrary policy labels but its
predeclared paired contrasts target the main eight-policy study. For your own
learning contrast, use `scripts/compare_learning.py`:

```bat
python scripts/compare_learning.py results/my_native_test results/my_neutral_test --out results/my_learning_analysis
```

The same command accepts a PPO evaluation folder as the first argument. This
comparison covers one learned checkpoint; it does not replace an analysis of
variation across multiple training seeds.

## Time budget

Measured here under concurrent CPU workloads: 50 native episodes took **8.42
minutes**. PPO's first 512 steps took **129 seconds**, or **3.97 steps/s** with
one worker. At that measured rate, 20,000 steps would take about **84 minutes**,
100,000 about **7 hours**, and 300,000 about **21 hours**. Three independent
training seeds approximately triple total compute work. These are extrapolations,
not measurements of a long run or your computer. Time a 2,048-step pilot using
your chosen worker count and replace the rate with its measured value.

More time buys a better test of learning; it does not guarantee a better policy.
If a well-tuned constant already performs similarly, retain the simpler method
and investigate physical model calibration or computational efficiency next.
