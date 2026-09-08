# Primary-source ledger and novelty boundaries

Search/review dates: 6–7 September 2026. This is a focused research search, not a
systematic review with database-complete screening or proof of priority.

| Primary source | What it supports | Consequence for this project |
|---|---|---|
| [Cortés et al., 2004](https://doi.org/10.1109/TRA.2004.824698) | Coverage control for mobile sensing networks | Coverage-gradient optimization is established prior art. |
| [Santos, Diaz-Mercado & Egerstedt, 2018](https://mariasantos.me/files/Santos_RAL2018_HeterogenousCoverage.pdf) | Coverage with heterogeneous sensing capabilities | Health-dependent sensing responsibilities alone are not a new field. |
| [Wang & Fu, 2019](https://doi.org/10.1109/ACCESS.2019.2953563) | Bioinspired neurodynamics in constrained waterjet-USV formation control | Biological neural dynamics have already been used on USVs; metadata/abstract checked, not reproduced here. |
| [Yates & Kaul, arXiv:1608.08622](https://arxiv.org/abs/1608.08622) | Receiver timestamp age for update systems | The age/deadline metric builds on established AoI concepts. |
| [Bafarassat & Coleri, 2026 preprint](https://arxiv.org/html/2608.00061v1) | Joint UAV freshness, reliability, connectivity and optimization; direct payload forwarding to fusion center in the stated model | Combining AoI and connectivity is already attempted; our multihop temporal delivery model is a narrower focus, not demonstrated superiority over their optimizer. |
| [Oyarte Galvez et al., Nature 2025](https://www.nature.com/articles/s41586-025-08614-x) | Measured fungal tip/filament travelling waves and the BARE equations | Motivates distinct expansion and density/transport roles. Full primary HTML and equations were read. A fixed-size vessel team does not reproduce fungal growth. |
| [Lucas et al., arXiv:2601.03877, 2026](https://arxiv.org/html/2601.03877v1) | Local branching, fusion and stopping with a network-material budget | Inspires a proposal family. This is a preprint; its biological trade-off findings do not prove vessel-controller optimality. |
| [Żukowski et al., PNAS 2024](https://arxiv.org/html/2212.08878v4) | Physical mechanism for breakthrough-induced loop formation | Loop formation is a transport phenomenon with specific physics; radio reinforcement needs a separate justification. |
| [Fernández-López et al., PNAS 2025](https://www.pnas.org/doi/10.1073/pnas.2506930122) | Movement heterogeneity and information transfer in foraging ants | Motivates recruitment. Primary abstract/metadata and [the authors' institute explanation](https://www.ceab.csic.es/en/les-formigues-un-cervell-liquid-on-la-diversitat-de-moviments-i-la-complexitat-dinteraccions-son-claus-per-a-leficiencia-collectiva/) were accessible. Full supplementary mathematics was blocked; no claim to reproduce it is made. |
| [Martinelli et al., Scientific Reports 16, 3457 (2026)](https://www.nature.com/articles/s41598-025-33456-y) | Physarum-inspired mesh formation in a simulated port with three robots | Adjacent prior art rules out a broad first-use claim for biological transport-inspired robot networking. Physarum is a slime mould; this is not evidence that an identical fungal-growth controller was used. Online publication was 26 December 2025. |
| [Fossen's author-maintained marine-craft model](https://fossen.biz/html/marineCraftModel.html) | Inertia/Coriolis/damping marine dynamics and current-relative formulation | Appropriate next physical model; current simulator is not a fitted Fossen-model implementation. |

The specifically implemented engineering extension is conditional edge-round
pivotality for report delivery before a deadline, mapped through radio probability
gradients into costed formation proposals. The Bernoulli differentiation identity
and grounded-Laplacian derivative are established mathematics. We derived and
tested their application here, rather than asserting a new general theorem.
No exact equivalent was identified in the focused search, but that is not proof
that none exists. A publication claim needs a closer reliability-importance,
temporal-network optimization and robot-MPC prior-art search and comparison.

Rejected shortcuts: ordinary flocking, PSO/ACO under a new name, unvalidated mast
optima, assuming an expected connected graph guarantees every realized graph,
treating absent messages as radar misses, or declaring a neural policy better
from one training curve. Course slides are background only, not the scientific
basis for a novelty claim.

Software references: [Gymnasium environment API](https://gymnasium.farama.org/api/env/),
[SB3 custom environments](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html),
[official PyTorch version installation instructions](https://pytorch.org/get-started/previous-versions/).
The NumPy residual learner does not depend on these optional packages.
The supplied CI syntax follows the official [checkout](https://github.com/actions/checkout)
and [setup-python](https://github.com/actions/setup-python) documentation checked
on 7 September 2026; the current examples use v7. Cloud CI has not been run here.


## Version 4 research update — 7 September 2026

| Source inspected | What transfers | What we actually implemented / claim boundary |
|---|---|---|
| [Amichay et al., Nature Communications 2024](https://doi.org/10.1038/s41467-024-48458-z), full primary text, phase-response and VR sections | Reciprocal lag-dependent timing, approximate T=2 tau in a response window | Coordinator goal-change eligibility with settling gate, fixed pairs and engineering time units. Matched periodic and greedy gates. No tailbeat, drag-saving or distributed-communication reproduction. |
| [March-Pons et al., Communications Physics 2025](https://doi.org/10.1038/s42005-025-02046-9), full primary text, Eqs. 1–2 | Discovery/recruitment/abandonment dynamics and sigmoid-bounded cross-inhibition | Ten virtual proposal populations, linear/nonlinear paired comparison. Quality normalization and memory over dynamic proposal families are our choices. No six-boat mean-field theorem. |
| [Zakir et al., arXiv:2509.07561v2](https://arxiv.org/abs/2509.07561v2), revised June 2026, primary full text | Bias-aware collective decisions; direct switch versus cross-inhibition | Adjacent robot-swarm precedent. Not reproduced, and it defeats a broad first-use-of-inhibition novelty claim. |
| [Singh et al., arXiv:2511.08436v2](https://arxiv.org/abs/2511.08436v2), revised July 2026, primary full text | Joint active sensing and communication studied using MARL in electric-fish-like collectives | Literature lead only: the current sensor model cannot represent electrical signal borrowing. No implementation or sensor-range gain claimed. |
| [SB3 official evaluation guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html) | Separate evaluation, multiple runs and reward/observation checks | Three short PPO runs, all nine constant controls, separate development selection and held-out comparison. |
| [SB3 vector environments](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html) | Multiprocessing constraints and vector rollout semantics | Windows instructions start with one environment. Linux multi-environment attempt failed due to AF_UNIX restrictions; independent single-environment runs used instead. |

Search scope included temporal coupling / robot fish, nonlinear honeybee inhibition, biased robot consensus, electric-fish collective sensing, army-ant bridges and recent self-organizing robotic aggregates. Only the first two led to implemented new controllers. Bridges and structural aggregates require different embodied mechanics and were not forced into this fixed-fleet radio model. Primary texts and explicit adjacent prior art were favored over novelty guesses. This is a bounded search, not a systematic proof that no one has tried an exact combination.
