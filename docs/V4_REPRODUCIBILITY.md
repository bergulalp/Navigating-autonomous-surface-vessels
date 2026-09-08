# 5.9.1. MARIN — version 4 reproducibility

Version 4 adds new experiments while preserving the physical simulator, v3 control
module and residual-learning source. The earlier 1,897 model-based missions and 38
learning evaluations remain included. Their hashes are verified by
`scripts/verify_release.py`; the current complete release is checked with
`scripts/verify_v4.py`.

## Added evidence

| Directory under `results/` | Records | Purpose |
|---|---:|---|
| `bio_pilot` | 12 | Integration/timing, no parameter selection |
| `bio_nominal` | 288 | Six policies, four scenarios, twelve held-out seeds |
| `bio_short` | 288 | Same design, shorter radio range |
| `bio_shifted` | 96 | Probit physical sensor, longer disturbance correlations |
| `bio_integration128` | 48 | Increased planner integration, two scenarios, four seeds |
| `rl_constant_development` | 108 | Nine constants × four scenarios × three seeds |
| `rl_ppo_development` | 36 | Three trained policies on the same development missions |
| `rl_final_test` | 160 | Three PPO policies, selected constant and neutral constant × 32 missions |

There are **732 additional model-based missions** and **304 additional learning
evaluations**, or **1,036 additional evaluation missions** in this version. Across
both included studies there are 2,629 model-based missions and 342 learning evaluations.
Automated test fixtures and training interactions are not added to these counts.

Three new PPO policies each received 2,048 environment steps (6,144 total). A separate
seed-99 driver check used two 128-step chunks (256 steps). Those are software/learning
interactions, not another 6,400 evaluation missions. The prior v3 640-step PPO software
check and 55 NumPy training episodes remain historical evidence.

## Independent selections and paired uncertainty

`V4_FROZEN_PROTOCOL.json` records the biological code hash, parameters, world-seed sets,
comparisons and learning selection before main testing. New main world seeds are
17000–17011. Model-shift seeds are 17500–17503 and integration seeds 17700–17703.
The 16000 integration pilot did not select parameters.

RL development uses world seeds 15000–15002; final evaluation uses 18000–18007.
The constant selection is saved in `results/rl_selection.json` and copied into the final
evaluation manifest. Training worlds are generated from the randomized training stream;
the listed evaluation world seeds are not training worlds.

The coverage intervals resample full world-seed blocks after averaging the scenario/profile
cells. The three new biological coverage tests receive Holm correction. Energy and
diagnostic outcomes are additional descriptive comparisons. RL intervals condition on the
three specific trained policies. They are not 96 independent training runs and do not
estimate all sources of deep-RL variability.

## Identity, restart and compatibility

Every new biological record identifies its full configuration, biological settings,
causal source hash and scenario/seed/policy identity. Every learner evaluation identifies
the evaluator source and checkpoint SHA-256. Exogenous tape hashes verify that paired
controllers encountered the same random environment.

The code refuses to resume into a directory with different causal source or settings.
Evaluation also refuses a directory associated with a different checkpoint or seed set.
Completed mission records are committed through a temporary file and atomic replacement.
The PPO driver saves complete rollout chunks; a process killed inside a chunk can lose
that unfinished chunk. PPO resume restores policy, optimizer and counters but restarts
environment trajectories. The current `usv_bio/ppo_train.py` runner records a distinct stream seed from training seed and completed steps, avoiding the repeated initial sequence of the legacy runner when it is resumed in frequent chunks. The three research pilots used the frozen legacy runner in one uninterrupted chunk each; their results are unchanged. An earlier 256-step driver debugging run used the old repeated-stream behavior and is not part of the released research results; the replacement driver check is the released two-chunk run. The NumPy baseline has a separately verified complete-episode
RNG restoration mechanism; these are different guarantees.

New biological code lives in `usv_bio/control.py` and `usv_bio/experiment.py`; its hash
includes the frozen physical and v3 controller hashes. Evaluation code has a separate
hash. Analysis and manuscript edits are not represented as new causal simulation results.

## Verification and build

```sh
python -m unittest discover -s tests -v
python scripts/verify_v4.py
python scripts/verify_v4.py --checksums
```

All **22 tests passed** in the authoring environment. The five new checks cover population
simplex preservation and numerical resolution, reciprocal timing, settling restrictions,
stale-observation behavior and exact reduction to economic control. They supplement the
existing forwarding, hidden-state separation, joint reliability and energy tests.

To regenerate the new analysis and manuscript inputs from the included raw data:

```sh
python scripts/analyze_v4_bio.py
python scripts/bio_trace_diagnostics.py
python scripts/analyze_v4_learning.py --data results/rl_final_test --selection results/rl_selection.json --out results/analysis_v4_learning
python scripts/make_v4_report.py
```

Then run `latexmk -pdf main.tex` inside `paper/`. The old figures and supplementary
tables are already included; their generators remain in `scripts/make_paper.py` and
`scripts/make_supplement.py`. Rebuilding figures on another platform may change PDF
metadata or rendering bytes without changing the underlying numbers. File checksums
apply to the shipped bytes, so regenerate the release manifest when intentionally
rebuilding a modified release.

## Execution limitations

The Linux runtime used Python 3.12.13, NumPy 2.3.5, SciPy 1.17.0, CPU PyTorch 2.8.0,
Gymnasium 1.3.0 and Stable-Baselines3 2.9.0. Pinned requirement files are included.
The first attempt to launch multi-environment PPO failed at the runtime's Unix socket
restriction before training. It generated no trained policy. Independent single-environment
runs completed and supply the reported data; no access-control change was attempted.

Windows batch files were inspected, and their underlying Python training, resume and
evaluation commands were executed on Linux. The Windows host and GitHub CI execution
are not claimed tested. Timing records came from a shared nine-logical-CPU environment;
they are measurements of those runs, not hardware-independent speed guarantees.

The current manuscript author and PDF metadata read **5.9.1. MARIN**. Biological papers
retain their own real author names. Historical source hashes and simulation outcomes
have not been changed to create apparent new experimental evidence.
