# 5.9.1. MARIN — USV formation research, version 4

A reproducible **synthetic maritime monitoring** study: six surface vehicles maintain
useful sensing reports while weather, sensor health and communication change.
The main quantity is information received before a deadline, together with actual
movement energy. Instantaneous graph connectivity is reported, not guaranteed.

**Windows users: start with [WINDOWS_RL_START.md](docs/WINDOWS_RL_START.md).**

```bat
windows\01_setup.bat
windows\02_check.bat
windows\03_train.bat 2048 1 0
```

Extract the complete project first, install 64-bit Python 3.12, and run those commands
from Command Prompt in the project root. Full installation, continuation, timing,
development and final evaluation instructions are in the guide.

## What is new

- Two additional biological mechanisms: reciprocal goal-change timing inspired by
  zebrafish, and thresholded proposal inhibition inspired by honeybee models.
- Periodic/greedy handover and linear-inhibition controls, with matched restrictions.
- 732 additional model-based missions: 576 main, 144 robustness diagnostics, 12 pilot.
- Three independent 2,048-step PPO runs and saved checkpoints; all nine constant
  parameter settings tested on development missions, then a separate final comparison.
- A tested Python training-progress/resume driver, Windows batch entry points,
  scientific plots, raw records, updated LaTeX and a separate Overleaf ZIP.
- Project identity changed throughout the current manuscript and metadata to
  **5.9.1. MARIN**. Earlier simulator and learning source remain reproducible.

The biological ideas are not automatically improvements. Reciprocal handovers saved
energy but lost coverage; nonlinear inhibition did not convincingly outperform its
linear comparison in the main study. The short PPO policies behaved as fixed settings
on both development and held-out missions. See the final results and action logs before investing in
longer training. No field performance or first-ever biological novelty is claimed.

## Read the work

| You want to... | Open this |
|---|---|
| Understand the project without coding | [Group brief](docs/GROUP_BRIEF.md) |
| Send a short update to teammates | [Ready message](docs/TEAM_MESSAGE.md) |
| Run or continue PPO on Windows | [Windows guide](docs/WINDOWS_RL_START.md) |
| Read the scientific manuscript | [Paper PDF](paper/main.pdf) |
| Recompile or use Overleaf | [Overleaf instructions](paper/OVERLEAF_README.md) |
| Inspect new biological mechanisms and evidence | [Methods source](usv_bio/control.py), [new study protocol](docs/V4_FROZEN_PROTOCOL.json), `results/analysis_v4_bio/` |
| Inspect final RL comparisons | `results/analysis_v4_learning/`, `results/rl_final_test/` |
| Understand what is frozen / how to reproduce | [Reproducibility](docs/V4_REPRODUCIBILITY.md) |
| See primary research and novelty limitations | [Source ledger](docs/SOURCE_LEDGER.md) |

## Python setup and a small additional simulation

For Linux/macOS, or an already activated Windows Python 3.12 environment:

```sh
python -m pip install -r requirements-tested.txt
python -m pip install -e . --no-deps
python -m unittest discover -s tests -v
python scripts/verify_v4.py
python -m usv_bio.experiment --config configs/bio_short.json --out runs/my_bio_pilot --seeds 2 --seed-start 22000 --scenarios storm moving_front --workers 1 --trace
```

PPO additionally needs the optional pinned dependencies; see the Windows guide for
CPU PyTorch and the `requirements-rl-tested.txt` installation order. Ordinary simulations,
analysis and scientific checks do not require PyTorch.

To repeat the full new biological study in a separate output tree:

```sh
python scripts/reproduce_v4.py --stage bio --workers 2 --out runs/reproduce_v4
```

To retrain the three short PPO pilots and repeat their evaluation:

```sh
python scripts/reproduce_v4.py --stage learning --workers 1 --out runs/reproduce_v4
```

These are substantial experiments, not installation checks. The raw released evidence
is already included; you do not need to rerun it to read the report.

## Architecture and scope

`usv_sim/` is the frozen physical, sensing and network simulator. `usv_research/` is the
frozen v3 economic/fungal/ant and residual-learning layer. `usv_bio/` adds the new
coordinator-side mechanisms and stricter learning comparisons. A central, stationary
coordinator issues goals; vessels execute finite-response motion and local avoidance.
This is not decentralized learned navigation, a full task-allocation system, a fitted
marine digital twin or a calibrated sonar/ASW model.

The broad research direction is established. The possible contribution is the narrow
combination of timely multihop information, movement cost, matched mechanism tests and
transparent negative evidence. A journal submission still needs closer optimization
baselines, stronger uncertainty calibration and a clearly agreed monitoring requirement.

## GitHub handoff

The release includes source, raw evidence, tests, figures, citation metadata, a proposed
MIT license and a Linux/Windows CI workflow. Review contributor credit and licensing
with the group before public release. No GitHub repository has been created or published.
After creating your own repository, from the extracted project:

```sh
git init
git add .
git commit -m "Add reproducible MARIN swarm study"
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

The run-output and environment exclusions are included. GitHub CI and the Windows batch
files have not been executed on a Windows host here; the Python workflow has been tested
on Linux. The manuscript acknowledges AI assistance and keeps synthetic results distinct
from validated vessel performance.
