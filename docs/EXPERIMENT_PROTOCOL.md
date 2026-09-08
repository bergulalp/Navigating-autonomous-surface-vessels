# Frozen v3 evaluation protocol

Recorded 7 September 2026, before opening the v3 held-out runs. This is an internal
protocol record, not an externally registered study. Earlier v2 results and v3
development seeds 100–102 informed the design. The controller parameters remain
at their engineering defaults; there was no exhaustive controller tuning.

## Study question and scope

For six simulated monitoring vessels, do small, costed formation changes improve
the fraction of priority-weighted sensing reports arriving within 30 seconds?
Can biological proposal mechanisms add value beyond an economic local controller?
This is a synthetic planar simulation study, with no field-validation claim.

## Selection and evaluation separation

* Static rings: radii 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10 times 2.5 km.
* Static selection: seeds 110–113, equal mixture of all six scenarios, separately
  for nominal (4.6 km) and short (3.2 km) calm radio half-success distances.
* Selection score: C - 0.025 E / (N * 100 W * T / 3600), with E in Wh.
  This selects one ring for a hardware profile; it is not the best static
  nonuniform geometry and not a static oracle selected per held-out scenario.
* Main held-out tests: seeds 7000–7019, storm, moving_front, sensor_fault,
  both radio profiles, all eight main policies. Exactly 960 episodes.
* Main policies: fixed_ring, static_tuned, adaptive (v2), economic,
  fungal, deadline_fusion, greedy_sparse, recruitment.
* All begin on the same outer ring; deployment and subsequent movement energy
  are charged. All use the same vessel dynamics and delayed command channel.
* Each controller sees cached coordinator poses/health estimates, their ages,
  current noisy weather and the declared priority schedule. There is no free
  true health, future weather, realized future links or ideal coverage field.

## Outcomes and interpretation

Primary engineering comparison: economic vs tuned static, delivered coverage C,
reported together with energy E. Also compare economic vs v2 adaptive for both
C and E. The equal-weight average of the six hardware/scenario cells is the
pooled estimand. Resample whole seed blocks across those cells; do not treat
the six cells of one seed, or individual time steps, as independent replicates.

Biological contrasts are exploratory: fungal vs economic, deadline_fusion vs
fungal, recruitment vs greedy_sparse. Report paired differences with 95%
percentile bootstrap intervals and sign-randomization p-values. Adjust those
three pooled p-values with Holm's method; do not use a favorable single scenario
as evidence of universal superiority. Exact equality is a valid negative result.

Secondary outcomes: clipped receiver AoI, its saturation fraction, 10th percentile
coverage, fraction of time below 50% coverage, command/telemetry staleness,
reference-traffic early reports, traveled distance, minimum separation and runtime.
AoI is clipped at 300 s, including never-received cells. Connectivity probability
is measured, not imposed as a guaranteed chance constraint. A blackout cannot be
fixed by formation change. The priority-front speed is 4.36 m/s in the one-hour
case and exceeds the 3 m/s through-water speed limit; it is deliberately demanding.

## Mechanism and robustness experiments

Predeclared exploratory checks, with fresh seeds separate from main tests:
cost removal, hydraulic fusion removal, and social recruitment removal; low
and high energy prices; probit sensing with longer wave correlation; slower
priority motion; finer planner integration. Negative controls include calm
water and complete temporary blackout. These checks diagnose mechanisms and
model sensitivity, not extra opportunities to tune on the main test outcomes.

Local controllers share 4N finite-difference field calls, ten goal candidates,
three trajectory quadrature times and the same energy criterion. Deadline fusion
additionally evaluates edge interventions over the time-expanded network; its
computation is explicitly greater. Biology does not receive free compute or
movement in the accounting. No continuous/global optimum or full MPC solution
is claimed for the conventional baseline.

## Reproduction and exclusions

Each episode records physical and extension code hashes, parameters, seed and
exogenous-tape hash. Existing outputs with different causal code or settings are
rejected. Completed files are reused on restart. Development files are excluded
from main figures. No episode is dropped because its result is unfavorable.
Training or further tuning must use new directories and development seeds;
reserve a fresh hold-out set for the next algorithm version.
