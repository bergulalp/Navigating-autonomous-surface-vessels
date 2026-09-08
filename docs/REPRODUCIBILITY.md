> **This inventory describes the preserved v3 study.** See V4_REPRODUCIBILITY.md for the added experiments and current release verification.

# Reproducing the v3 release

## Source and data identity

The physical `usv_sim/*.py` files are unchanged from v2. This allows the new
high-level control methods to be compared inside the same physical model.

| Identity | SHA-256 |
|---|---|
| Physical code | `24bfb8b9aac5b87f9f268595cf4462964bd3a01da9d8089f51ed740d780fee86` |
| Causal v3 extension | `f000d324411431a807bac72657424c44472cc457a2b2f856442e941850c00046` |
| Residual-learning source | `5113616c5fa00215ac2ca5cfa63d5545c3ead1d5a52d320aad84b253498818e5` |
| Native checkpoint evaluated at 50 episodes | `d303d8e5aa1a38d6afabfcc45dbd84e9aac82f219d411126965ab4bf2cb6e27e` |
| PPO checkpoint after 640 smoke-test steps | `963da9d8a27020b8b69a9f56258f8774db3efb6eb9cb15649d704380eae25763` |

`docs/FROZEN_CODE.json` and each raw episode retain provenance. A code change
requires a new output directory. The release verification script checks expected
counts, run identities, source hashes, paired random-tape hashes, finite metrics,
and learning-checkpoint identities. File checksums are in `RELEASE_MANIFEST.json`.
Checksums identify files; they do not independently prove the scientific model.

## Data inventory

| Directory under `results/` | Episodes | Purpose |
|---|---:|---|
| `development_v3` | 45 | Final-code development check, excluded from held-out estimates |
| `static_selection_nominal` | 192 | Select a nominal-radio static ring |
| `static_selection_short` | 192 | Select a shorter-radio static ring |
| `main_nominal` | 480 | Main held-out nominal-radio test |
| `main_short` | 480 | Main held-out shorter-radio test |
| `ablation_short` | 140 | Remove energy, fusion and social mechanisms |
| `price_low`, `price_high` | 60 each | Sensitivity to movement price |
| `shifted_short` | 96 | Changed actual detector and wave correlation |
| `slow_short` | 32 | Slower scheduled observation priority |
| `integration_128` | 24 | Within-setting policy comparison with 128 planning paths |
| `negative_nominal`, `negative_short` | 48 each | Calm and complete temporary-blackout controls |
| `native_eval`, `native_neutral` | 18 each | Evaluated 50-episode learner vs constant action 4 |
| `ppo_eval_smoke` | 2 | Load/evaluate PPO software check |

Training episodes and PPO steps are recorded separately. An older 28-episode v3
prototype used a different causal hash; it is excluded from the final release
and all final estimates. No unfavorable final-code episode was removed.

## Recreate static selection

These scripts rerun all selection episodes and overwrite matching names. Use a
separate extracted copy or new output folders to preserve the released evidence.

```bat
python scripts/select_static.py --config configs/benchmark.json --out results/my_static_nominal --seeds 4 --seed-start 110 --workers 4
python scripts/select_static.py --config configs/short_radio.json --out results/my_static_short --seeds 4 --seed-start 110 --workers 4
```

The selection reports should choose 0.90R and 0.85R respectively under the
released environment. Floating-point/SciPy changes can produce small numerical
differences and, near action ties, different trajectories. The selected control
files used in the main suite are already in `configs/control_nominal.json` and
`configs/control_short.json`.

The generic analysis loader rejects mixed settings for identical policy/seed
keys. Static-selection directories intentionally contain multiple radii and
should be inspected through `selection.json` or the selection script, not passed
to the main episode analysis CLI.

## Recreate the main suite and diagnostics

```bat
python scripts/reproduce_study.py --suite all --workers 4
python -m usv_research.analyze results/main_nominal results/main_short --out results/analysis_main
python scripts/make_paper.py
python scripts/make_supplement.py
python scripts/verify_release.py
```

The runner reuses completed episodes. To inspect every command before a fresh
run, add `--show`. Set aside existing result folders in a separate working copy
if you want to execute every episode again. The frozen data live in the release
and should not be modified to make a new method appear more successful.

Bootstrap intervals resample whole seed blocks. Main biological coverage tests
receive Holm correction; diagnostic intervals are exploratory and pointwise.
Counts of time steps are not used as independent sample sizes. The source/tape
checks ensure paired experiments actually share the stipulated conditions.

## Software checks

```bat
python -m unittest discover -s tests -v
python scripts/verify_release.py
```

Tests target scientific failure modes: causal forwarding, common-route
dependence, health-estimation observation rules, report deadlines, native
checkpoint continuation, physical-energy integration, hydraulic derivatives,
conditional link influence and neutral residual equivalence. The optional PPO
checker and trained checkpoint were separately exercised; see `RL_EXPERIMENT.md`.

The GitHub Actions workflow runs core tests on Linux and Windows with Python
3.12. Its cloud runs will occur after the group creates a repository; a supplied
workflow is not a claim that GitHub CI has already passed.

## Hardware and wall time

The runtime exposed nine logical CPUs. Four-worker main batches took **35.30
minutes for nominal radio** and **32.26 minutes for shorter radio**, each for
480 missions, while other jobs shared the runtime. These are observed batch
durations, not isolated hardware benchmarks or additive estimates of total
elapsed project time. Episode `runtime_s` includes worker contention.

Use your own representative pilot and `scripts/estimate_runtime.py` to budget a
larger study. Finer planning integration and deadline interventions cost more
than static control. Doubling workers need not double throughput.

## Build the manuscript

After regenerating the numerical inputs:

```bat
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The article uses standard packages in a full TeX Live/MiKTeX installation. It is
a two-column scientific manuscript, not a specific journal's official template.
Before an actual journal submission, agree on authorship, choose the target
journal, adapt its required template, and substantiate the novelty/validation
claims. The PDF and generated figures were visually inspected for this release.

## Provenance and reuse

The uploaded project and course slides were inputs to the work. Third-party
course PDFs and research-article copies are not included. Source attribution
appears in `paper/references.bib` and `docs/SOURCE_LEDGER.md`. Bibliographic metadata
and equations should still receive the group's normal scientific review.
Code and manuscript preparation were AI-assisted; the authors remain responsible
for their research claims and any institutional requirements before publication.

The supplied MIT license is a proposed code-sharing license for group review.
It does not grant rights to third-party course materials or papers. The release
contains no remote GitHub URL, invented DOI or claimed journal acceptance.
