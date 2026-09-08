# 5.9.1. MARIN — group briefing

## What problem are we studying?

We have six unmanned surface vessels and a coordinator. The boats sense a specified
monitoring region and pass reports through a changing radio network. A report only
helps if it reaches the coordinator soon enough. Spreading out can improve sensing,
but it can make communication harder; moving also takes time and battery energy.

The current coordinator is stationary. This version does not implement a moving
frigate, the seven proposed missions or submarine localization. Its demonstrated
scope is generic maritime monitoring with synthetic surface-sensing and radio models.

## What is already working?

The simulator can change weather conditions, move the region of greatest monitoring
interest, degrade a sensor and interrupt radio communication. Boats take time to turn
and move. Reports, sensor-health estimates and commands can arrive late. Different
controllers receive the same underlying random environment in each paired comparison,
which makes their differences easier to interpret.

The strongest result from the earlier main study is an ordinary **economic controller**:
it asks whether a small formation change is likely to deliver enough extra useful
information to justify the movement. Against a fixed formation selected on separate
examples, delivered coverage increased by about **0.95 percentage points**. Against the
earlier large-formation-switching controller, it used about **62.5% less modeled energy**
while improving coverage by about **1.69 percentage points**. Those are simulation results
for the declared benchmark, not measured gains from real boats.

## What new biological ideas did we actually try?

| Mechanism | Plain explanation | What the new main tests found |
|---|---|---|
| Reciprocal timing, inspired by zebrafish | Partners adjust when they are allowed to receive a new movement goal, responding to one another’s timing | Against periodic handovers: coverage fell by 0.21 percentage points, while energy decreased. Against unrestricted economic control: coverage fell by 1.35 points. |
| Nonlinear inhibition, inspired by honeybee decision models | Support accumulates for movement options; a thresholded stop signal suppresses competing options | Coverage differed from linear inhibition by only +0.005 percentage points, with an interval including zero. No convincing advantage. |
| Earlier fungal branching/fusion | Separate sensing expansion from communication repair | Small gains in some earlier settings, but they did not hold consistently in robustness checks. |
| Earlier ant recruitment | Let some boats move while others maintain their stations | It did not reliably outperform simpler participation rules. |

These findings do not mean that biological research is useless. They mean these
particular transfers, parameters and mission models did not justify replacing the
strong ordinary controller. We did not add extra sensing range or free energy savings
merely because an animal can exploit a different physical mechanism.

One useful diagnosis is that this model allows sensing while moving, so alternating movements do not automatically create a sensing benefit. Also, ten similar proposal choices split the virtual bee support: in the eight saved nonlinear-inhibition traces, support exceeded the chosen threshold at only 1.29% of nonstale decisions. This suggests a mismatch in the transfer, not a universal failure of the animal mechanism.

The research also considered recent work on electric-fish sensing/communication and
biased collective decisions in robot swarms. Electric-fish signal borrowing needs a
sensor model with the corresponding physical interactions, so it was not claimed as
an implemented capability. Related robotics work already exists; a broad “nobody has
tried biological swarms” novelty claim would be incorrect.

## What happened with reinforcement learning?

PPO was trained three times from different random initializations, with **2,048 steps
per run**. Each step is a high-level decision after one simulated minute. It selects
one of nine combinations of movement size and willingness to spend energy. It does
not learn steering, collision avoidance or task allocation.

We first evaluated all nine fixed combinations on development examples. The selected
combination was action 3: a 120 m movement-proposal size and price 0.0125. We then
compared all three PPO policies, that selected setting and the original economic setting
on 32 new missions per policy.

One trained policy exactly matched the selected fixed setting because it always chose
that same action. The other two always chose another fixed setting, saving energy but
losing coverage and reward. **None demonstrated state-dependent adaptation in these
short evaluations.** This is a pilot result, not proof that a longer or better-designed
learning experiment can never work.

The new Windows driver saves progress in chunks and records a different environment
seed stream for each chunk, so repeated continuation does not intentionally recycle
the same starting sequence. Its training, saving, loading and continuation commands
were exercised on Linux. Actual Windows execution and GitHub CI still need to be run.

## What has been delivered, and how can we use it?

- **Full repository ZIP:** source, configurations, raw results, analyses, tests and saved
  policies. A coder can reproduce results or add a controller without guessing the model.
- **Scientific PDF and Overleaf ZIP:** read the research, inspect the graphs, revise the
  manuscript collaboratively and compile it without installing Python.
- **Windows guide and batch files:** install once, perform a short speed test, train more,
  then compare against fixed settings on separate development and test missions.
- **Reproducibility checks:** 22 scientific tests passed. The additional release contains
  732 model-based missions and 304 learning evaluations, separately counted from training.

The report’s first part explains the sensing/communication model and earlier economic,
fungal and ant comparisons. “Further biological mechanisms: a separate experiment”
contains the zebrafish/honeybee equations, new graphs, robustness tests and short-PPO
results. “Interpretation and practical research direction” explains limitations and
possible contributions. The appendix documents reproducibility and earlier diagnostics.

## What should the group decide next?

1. **Define success.** Must every boat be connected continuously, or is receiving its
   information within a deadline sufficient? Our current score addresses timely delivery;
   it does not enforce an always-connected network.
2. **Agree the autonomy boundary.** A manageable research design has a coordinator setting
   monitoring priorities, a high-level allocation/formation layer, and ordinary onboard
   navigation and avoidance. The current implementation changes formation parameters;
   task allocation is an additional research question.
3. **Specify the uncertain quantities.** Sensor health, visibility, packet delivery,
   current disturbances and message age are already uncertain here. Calibrating their
   distributions matters more than making the simulator arbitrarily noisy.
4. **Choose one narrow contribution.** Strong candidates are calibrated acceptance of a
   manoeuvre under model uncertainty, explicit deadline-connectivity constraints with
   infeasibility handling, or a well-controlled test of a useful high-level learning task.
   These are future directions, not capabilities already proven by the code.
5. **Spend compute only against a clear comparator.** Run 20k PPO steps with several seeds
   after the short setup test. Use development results to decide whether 100k is worth
   trying. Keep final test missions unseen until choices are fixed.

Non-coders can contribute by defining the mission score, reviewing model assumptions,
collecting sensor/radio evidence, checking the biological-to-engineering mapping and
interpreting plots. Coding is only one part of turning this into a useful study.

## Two points about optimisation and publication

A globally exact optimiser with a correct model cannot be beaten on the same modeled
objective. But determinism does not guarantee that such an optimiser is available or
fast enough. RL can sometimes approximate repeated decisions cheaply or use observation
history; it is not automatically preferable under uncertainty. Robust or stochastic
model-based optimisation belongs in the comparison too.

The current manuscript is ready to inspect and place in a research repository. A Q1
journal contribution is **not established yet**. The next step is a narrow, defensible
claim supported by closer prior-art baselines, calibrated uncertainties and stronger
validation. Honest negative results prevent us from spending the internship on a
biological label that adds no measurable value.
