# 5.9.1. MARIN — start here on Windows

These commands require the full **MARIN_Swarm_v4.zip** archive. The code is ready for local simulation and PPO training. More training may improve the
learned decisions, but it can also plateau or perform worse. The included comparison
workflow is how you find out. This is a synthetic environmental-monitoring experiment;
the learner is not a navigation or submarine-localization system.

## 1. Extract and open the project

Install **64-bit Python 3.12** using the official [Python Install Manager](https://www.python.org/downloads/windows/). With the manager installed, run `py install 3.12` in Command Prompt. If Python 3.12 is already installed, `py -3.12 --version` should print its version and you can continue. Extract the full release ZIP into a simple local folder,
for example `C:\MARIN\marin_swarm_v4`. Do not run inside the ZIP viewer.

Open **Command Prompt** in that folder. Type:

```bat
cd /d C:\MARIN\marin_swarm_v4
windows\01_setup.bat
windows\02_check.bat
```

Setup downloads NumPy, SciPy, plotting tools, CPU PyTorch, Gymnasium and
Stable-Baselines3 into a project-local `.venv`. It does not require a GPU, Docker,
Linux, WSL or a LaTeX installation. Allow several GB of free disk space. Use an
internet connection for this installation; simulation and training run locally afterward.
The setup script stops if any installation step fails.

The Python commands and training/evaluation paths were exercised on Linux. These
Windows batch files were inspected but were **not executed on a Windows machine**.
The repository includes Windows CI configuration; a live GitHub CI run is still needed.

## 2. Measure your own training speed

```bat
windows\03_train.bat 2048 1 0
```

Arguments mean **additional training steps**, **parallel environments**, **training seed**.
One step represents one minute of the simulated mission; it is not one second of wall
time. This command saves `runs\windows_seed0\final.zip` after 2,048 steps. Each episode
contains 40 steps. Progress appears after the saved chunk, so the first chunk can be
quiet for several minutes.

Do this pilot before choosing an overnight run. The script prints measured steps/second
and the remaining-time estimate. For a requested budget `B`, approximate time in hours
is `B / (3600 * steps_per_second)`. Hardware load, worker count and scenario mix change it.

| Budget for one training seed | At 2 steps/s | At 4 steps/s | At 8 steps/s |
|---|---:|---:|---:|
| 2,048 steps | 17 min | 9 min | 4 min |
| 20,000 steps | 2.8 h | 1.4 h | 42 min |
| 100,000 steps | 13.9 h | 6.9 h | 3.5 h |
| 300,000 steps | 41.7 h | 20.8 h | 10.4 h |

Our three 2,048-step pilots measured 2.74–3.70 steps/s, taking 9.2–12.5 minutes each while other jobs shared the CPU. At those rates, 20k additional steps is about 1.5–2.0 hours and 100k about 7.5–10.1 hours per training seed.

These are arithmetic estimates, not promised training times or convergence times.
Three seeds require roughly three times the CPU work. Evaluations add time.

## 3. Continue a run without losing the checkpoint

```bat
windows\03_train.bat 20000 1 0
```

This **adds** about 20,000 steps to the existing seed-0 policy; it does not reset it or
set a total target of 20,000. PPO rounds a chunk up to complete rollout blocks of
`128 * workers`. The driver saves every 2,048 requested steps by default. Resume loads
the policy, optimizer and timestep count, but starts fresh environment trajectories.
The current driver derives a distinct, recorded environment-stream seed from the training seed and completed step count at every chunk. This avoids deliberately replaying the same sequence on each resume. Chunking is still a training choice, not a bit-for-bit continuation.

To preserve a milestone before continuing:

```bat
xcopy runs\windows_seed0 runs\milestone_seed0_early\ /E /I
```

Keep `final.zip` **and** `manifest.json` together. Evaluation checks their source and
observation dimensions. Use these supplied checkpoints only in the matching simulator.
Changing a simulator/controller source file requires new training output; the code
deliberately rejects incompatible resume attempts.

Run seeds 1 and 2 too once seed 0 works:

```bat
windows\03_train.bat 20000 1 1
windows\03_train.bat 20000 1 2
```

Start with one environment. On a multi-core Windows PC you can try two, using a new
training seed/output for a fair speed comparison. More workers are not guaranteed
faster. Do not compare a larger rollout batch to a smaller one as if only runtime changed.
Use Command Prompt, not an interactive Python notebook, for multiprocessing.

## 4. Development: decide whether there is a useful signal

```bat
windows\04_development.bat after20k
```

This evaluates seed 0 and all **nine constant parameter combinations** on common
development missions. The best conventional setting is saved in
`runs\selection_after20k.json`. Graphs and a readable interpretation go to
`runs\dev_after20k\analysis`.

Use a fresh tag (`after100k`, for example) whenever the checkpoint changes. Evaluation
refuses to mix different checkpoint contents in an existing output folder.
You can repeat development while tuning. Do not call these results a final held-out test.

To include all three training seeds, activate the environment and use:

```bat
call .venv\Scripts\activate.bat
python -m usv_bio.learning_eval --config configs/bio_short.json --out runs/dev_three_seeds --split development --constants 0 1 2 3 4 5 6 7 8 --checkpoints runs/windows_seed0/final.zip runs/windows_seed1/final.zip runs/windows_seed2/final.zip --seeds 6 --seed-start 15000 --workers 1
python scripts/windows_rl.py select --data runs/dev_three_seeds --out runs/selection_three_seeds.json
```

## 5. Final comparison: use unseen missions once the choices are fixed

For the simple seed-0 path:

```bat
windows\05_final_test.bat after20k
```

This compares the policy with the development-selected conventional setting and neutral
economic action 4 on 80 new missions per policy. It can take longer than the training
pilot. For the stronger three-training-seed comparison:

```bat
python -m usv_bio.learning_eval --config configs/bio_short.json --out runs/test_three_seeds --split test --selection runs/selection_three_seeds.json --checkpoints runs/windows_seed0/final.zip runs/windows_seed1/final.zip runs/windows_seed2/final.zip --seeds 20 --seed-start 19000 --workers 1
python scripts/analyze_v4_learning.py --data runs/test_three_seeds --selection runs/selection_three_seeds.json --out runs/test_three_seeds/analysis
```

The analysis produces `comparison.png`, `comparison.pdf`, `means.csv`, `paired.json`
and `READ_RESULTS.md`. Positive reward difference is favorable; positive coverage
difference is favorable; negative energy difference saves energy. An interval crossing
zero is inconclusive. Check each scenario and each training seed, not only a pooled mean.
Bootstrap intervals condition on the trained policies; three training seeds are too few
for a strong population-level algorithm claim.

Once you have looked at this test set, it becomes part of your development knowledge.
For the next substantive revision, reserve new test world seeds (e.g. 20000–20019).
Do not keep choosing the checkpoint that happens to score best on the same final test.

## What RL is actually doing

Every 60 simulated seconds, PPO sees four recent coordinator observations: received
positions, estimated sensing health and uncertainty, message ages, reported energy,
previous goals, noisy weather and time. It chooses one of nine `(movement step, price)`
pairs. An ordinary model-based controller generates and screens the proposed formation
change. Onboard guidance and avoidance remain ordinary code.

This is **high-level parameter adaptation**. It is not learned trajectory planning,
multi-agent RL, a full task allocator or recurrent PPO. The existing four-frame stack
provides finite history; it does not make the hidden-state process fully observable.

The reward is delivered-report coverage during that interval minus normalized actual
energy cost. Training uses discount 0.99. Development selects settings using mean
undiscounted interval reward. Post-warmup coverage and total energy are also reported;
they should not be confused with the discounted training objective.

## Continuing the supplied short PPO pilots

To resume one of our checkpoints in a new folder, after activating `.venv`:

```bat
python -m usv_bio.ppo_train --config configs/train.json --out runs/from_supplied_seed0 --steps 20000 --workers 1 --seed 0 --resume results/ppo_v4_single_seed0/final.zip
```

This direct command saves the final policy and periodic checkpoints but does not use
the chunked progress wrapper. After it finishes, subsequent wrapper calls can use
`--out runs/from_supplied_seed0`. Keep supplied evidence folders unchanged.

## Troubleshooting and a useful research sequence

- **`py` not found:** install Python 3.12 with its launcher, then reopen Command Prompt.
- **No module named `torch` / `stable_baselines3`:** rerun setup and use the project's `.venv`.
- **Permission or worker-launch error:** use `workers=1`; do not disable security software.
- **A launch wrote a manifest but no checkpoint:** use a new output directory. The failed
  manifest records an incomplete launch, not completed training.
- **Resume rejected:** match seed, simulator source, training config and four-frame stack.
- **Laptop stops overnight:** connect power and choose a suitable Windows sleep setting.
  A checkpoint survives; in-progress work since the last checkpoint may not.

First establish reproducibility, then run 20k steps with three seeds, inspect development,
and budget 100k only if learning shows a useful signal. If all seeds collapse to a constant
action, compare that constant directly: it may be parameter tuning rather than adaptive
intelligence. Better sensing/radio calibration and a clearly agreed mission metric may
be more valuable than another million uncalibrated training steps.

References: [SB3 evaluation guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html),
[SB3 vector environments](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html).

## A specific next biological diagnostic (supplied, not executed)

The ten virtual proposal populations rarely crossed the original 0.3 inhibition midpoint.
A supplied exploratory configuration lowers it to 0.1. This is a hypothesis prompted
by the observed traces, not an improved controller already validated by this report:

```bat
python -m usv_bio.experiment --config configs/bio_short.json --bio configs/bio_threshold_01.json --out runs/threshold_development --policies economic linear_inhibition nonlinear_inhibition --scenarios storm moving_front --seeds 10 --seed-start 23000 --workers 1
```

Treat those missions as development. If there is a useful signal, freeze the choice and
use new world seeds, additional scenarios and the changed detector model for a final
comparison. Both linear and nonlinear controls remain in the test.
