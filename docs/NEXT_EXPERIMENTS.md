# A focused next study

## Recommended starting point

Keep `economic` as the main practical baseline. The tested biological additions
are too small and sensitive to model settings to justify selecting them solely
from their favorable pooled default result. A useful next claim would connect
deadline-dependent radio reinforcement to **measured**, condition-dependent
delivery failures and an explicit mission constraint.

## Agree on the requirement before tuning

Choose a report deadline and an acceptable outage level with the internship
supervisor. Decide whether connectivity means an instantaneous connected graph,
a timely path from every required boat, or timely mission-level information.
These are different requirements. Set a movement-energy budget and a failure
response for conditions where the requirement is physically infeasible.

The current score is mission-level freshness. Merely increasing its weight does
not prove an all-node connectivity requirement is met. Similarly, a fully
disconnected channel needs a declared fallback, not a larger formation gain.

## Collect three small, useful datasets

1. **Radio:** timestamp, sender, receiver, distance, antenna height/orientation,
   transmitted/received packet counts, retries, payload rate and environmental
   observations. Simultaneous links matter: common outages cannot be inferred
   from one isolated link's average success rate. Retain missing packets.
2. **Sensing:** known reference object identity and location, range/bearing,
   sensor setup, detection decision, visibility/weather and timestamp. Define
   a detection and a false alarm before recording. Ground-truth opportunities
   must include missed observations, not just successful detections.
3. **Movement:** position, heading, through-water or estimated speed, current,
   thrust/control commands, voltage/current and elapsed time for station
   keeping, a short relocation, a turn and a larger relocation.

Separate fitting and validation records by complete session/day or environmental
condition. Randomly splitting adjacent samples from the same outage or wave
sequence gives an overly easy validation problem.

## One mathematical extension worth implementing next

For the same ten candidate moves, evaluate improvement on an **independent
validation bank** after proposing moves. This can reduce selection optimism
from choosing and scoring on the same 32 planning paths. It still validates
only the assumed model. Use paired comparisons within each validation path to
reduce variance, and account for the number of actions compared.

If a valid bound `|estimated J(g) − true J(g)| ≤ δ` holds for every compared
action, an estimated advantage greater than `2δ` certifies a positive true
advantage. The triangle-inequality proof is straightforward; obtaining a useful
bound from calibrated model error, delayed poses and sensor-health uncertainty
is the substantive research task. More Monte Carlo samples alone cannot bound
unmodeled sea clutter or interference.

An explicit deadline-reliability constraint is another candidate extension.
Compare it with a conventional robust local optimizer using the same calibrated
uncertainty and information. Keep the biological proposal as one optional search
direction so its incremental value can be measured.

## Freeze and test

Use development seeds 200–299 to debug and select parameters. Freeze code and
parameters, then use a fresh test bank such as 10000–10029. Include tuned static,
economic, the new robust/constrained economic method, and the same method with
deadline proposals. This isolates the added proposal mechanism from improvements
caused by the common uncertainty treatment.

Evaluate both radio profiles and retain storm, moving-priority, fault and
blackout conditions. If you select a specialized storm controller, state that
scope before its new test. Record delivered freshness, energy, the chosen
connectivity requirement, tail failures and computation. Add 3/9-vessel cases
only after you budget their runtime; no fleet-size scalability claim has yet
been tested in the current release.

In parallel, a practical software contribution is faster conditional link
influence through cached reachability and pruning of irrelevant edge-round
interventions. Validate it against the exact enumeration test before a speed
comparison. A reproducible computational improvement can be useful even if the
biological interpretation adds no fleet-performance gain.
