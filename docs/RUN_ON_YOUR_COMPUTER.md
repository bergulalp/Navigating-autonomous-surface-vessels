> **For the current Windows RL setup, use [WINDOWS_RL_START.md](WINDOWS_RL_START.md).** This guide preserves the earlier v3 reproduction commands.

# Run, extend and evaluate the project

## 1. Install once (Windows / Anaconda Prompt)

Extract the ZIP. Open **Anaconda Prompt**, enter the folder containing this README,
and run the following commands. Use a new environment, not an existing course or
OpenSim environment. Commands that launch multiple workers belong in the terminal,
not inside Spyder's interactive console.

```bat
conda create -n usv-research python=3.12 -y
conda activate usv-research
python -m pip install -r requirements-tested.txt
python -m pip install -e . --no-deps
python -m unittest discover -s tests -v
```

On macOS/Linux, the same commands work if Conda is installed. Alternatively use
`python3 -m venv .venv`, activate it, and run the pip/test commands. Python 3.12 is
the tested version. The main simulation needs no GPU, ROS, MATLAB or Gazebo.

Allow approximately 15–30 minutes for first installation, depending on your
connection. These are planning estimates, not measured installation times on
your computer. The numerical dependency versions are pinned in
`requirements-tested.txt`; optional RL has separate dependencies.

## 2. Run a small new experiment and measure your computer

From the project root, as single-line commands:

```bat
python -m usv_research.experiment --config configs/short_radio.json --control configs/control_short.json --out results/my_pilot --policies static_tuned economic fungal deadline_fusion --scenarios storm moving_front --seeds 2 --seed-start 9000 --workers 2 --trace
python -m usv_research.analyze results/my_pilot --out results/my_pilot_analysis
python scripts/plot_experiment.py results/my_pilot --out results/my_pilot_analysis
python scripts/estimate_runtime.py results/my_pilot --episodes 480
```

This runs **16 one-hour simulated missions**. The hour is model time, not one
hour spent waiting. The timing file measures actual wall time and includes
parallel processing. On the development runtime, 45 mixed-policy missions took
180 seconds with four workers; the deadline-fusion controller took more compute
than the other new controllers. Your pilot is the estimate to trust. CPU speed,
power-saving mode, temperature, other applications and worker count matter.

Start with two workers. Try four using a new output folder and the same episode
set if your machine stays responsive. Increasing workers beyond physical CPU
capacity often gives no benefit. Keep the laptop plugged in for extended jobs.
There is no reason to buy a GPU for the current model-based experiments.

The result folder contains one JSON per episode, a CSV, configuration/identity
metadata and a timing log. `--trace` stores detailed trajectories for the first
seed of each scenario/policy. Repeating the identical command reuses completed
episodes. An interrupted, uncompleted episode is recomputed. Do not edit causal
code and reuse an old output directory: the source check will reject it.
Open `results/my_pilot_analysis/my_pilot_overview.png` to see the new comparison.
The plot script accepts your own output directories; the paper scripts rebuild
the specifically frozen published study. A two-seed pilot checks execution,
not whether a method is reliably better.

## 3. Obtain stronger evidence

```bat
python -m usv_research.experiment --config configs/short_radio.json --control configs/control_short.json --out results/my_validation --policies static_tuned economic fungal deadline_fusion greedy_sparse recruitment --scenarios storm moving_front sensor_fault --seeds 30 --seed-start 9100 --workers 4 --trace
python -m usv_research.analyze results/my_validation --out results/my_validation_analysis
```

This is 540 missions. Estimate time from a pilot with a similar mix of policies.
At the illustrative development throughput of 4 seconds per episode in a
four-worker batch, 540 missions take about 36 minutes; that is not a promise for
your hardware. Deadline fusion is more expensive, and finer grids or longer
missions change the rate. Use the runtime-estimator script after your own pilot.

Do not tune parameters on these validation seeds. Start exploratory work with
seeds 200–299, compare candidate settings there, freeze one setting, then evaluate
on a fresh set such as 10000–10029. More seeds reduce statistical uncertainty;
they do **not** improve a fixed controller. Approximately four times as many
independent seeds are needed to halve a standard-error scale.

## 4. What to change, and what each change can accomplish

* `--energy-price 0.01` or `0.05`: change the willingness to spend propulsion
  energy for coverage. Always compare both outcomes; neither value is universally
  better. Write to a new `--out` folder.
* `--step 60` or `180`: change the maximum local proposal displacement in metres.
  Larger steps can respond faster but may cost more and invalidate local forecasts.
* `configs/slow_short.json`: a slower moving priority; useful for testing whether
  failure is caused by an impossible mission speed rather than intelligence.
* `configs/shifted_short.json`: sensing-curve mismatch and longer wave correlation.
  This probes one model shift, not all real maritime uncertainty.
* `configs/integration_128.json`: four times the planner samples. This improves
  numerical integration fidelity; it is not training and need not improve outcomes.
* In a copied config, change `duration_s` and keep a valid `warmup_s`. A longer
  duration also slows a priority scheduled as turns per mission. To keep the same
  front speed, scale `priority_turns` with duration and document that change.

The main bottleneck for credibility is now physical calibration: measure packet
delivery vs distance/weather and detection probability for known reference
objects, then fit and independently validate those curves. Longer runs of
unfitted curves cannot substitute for these measurements.

## 5. Learning options

The inherited, dependency-light Double-Q implementation remains runnable:

```bat
python -m usv_sim train --config configs/train.json --out results/legacy_q_seed42 --episodes 200 --training-seed 42
python -m usv_sim train --config configs/train.json --out results/legacy_q_seed42 --episodes 200 --training-seed 42 --resume
```

On resume, `--episodes 200` means **200 additional episodes**. Keep the same
training seed, configuration and source. This legacy learner selects the old
eight formation options; it does not train the new v3 local controllers. Earlier
training did not establish superiority, so it is a reproducible learning baseline,
not the recommended way to obtain an immediate performance improvement.

For v3, a better learning experiment is a small residual policy choosing the step
size and energy price around the economic controller, with recent observation
history. The optional residual workflow is documented separately in
`RL_EXPERIMENT.md`. Its NumPy learner uses the base environment; PPO requires
additional dependencies and a smoke test before long training. A trained policy must be compared with the
unchanging economic controller and the best constant residual action on held-out
seeds, with at least three independent training seeds.

## 6. Time to improvement: realistic milestones

| Work | Planning allowance | What it buys |
|---|---:|---|
| Installation and first pilot | 30–60 min | A working, measured local setup |
| Inspect figures and repeat paired tests | Half a day | Understanding and reproducibility |
| Small parameter sweep + fresh evaluation | 1–2 days | Evidence for a useful setting, if one exists |
| Residual-RL pilot and three training seeds | Several compute hours to days, measured from the pilot | A test of learning value, no guarantee of improvement |
| Fit real sensor/radio curves | Roughly 1–3 weeks if equipment/data access exists | The largest improvement in practical credibility |
| Hardware tests and journal-strength study | Often several further weeks or months | Evidence needed beyond this simulation manuscript |

These are project estimates; equipment availability can dominate the schedule.
A useful internship outcome is a validated decision rule plus a reproducible
testbed. A publication requires a defensible novelty claim and validation, not
only more training or a journal-style PDF.

## 7. Make the GitHub repository

The ZIP is arranged as a repository root, with code, tests, figures, paper sources,
documentation and GitHub Actions configuration. After reviewing the files:

```bat
git init
git add .
git commit -m "Add reproducible USV formation study"
git branch -M main
```

Create an empty repository in your own GitHub account, then follow GitHub's
displayed commands to add its remote and push. No external repository is created
or published automatically. The generated release manifest records file hashes;
regenerate it after intentional changes. Large future datasets belong in a
versioned data release rather than repeated commits. Do not add private client
information or third-party course PDFs to a public repository.
