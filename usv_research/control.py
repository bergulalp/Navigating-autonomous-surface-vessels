"""Economic local planning and explicit biological proposal ablations.

These are coordinator-assisted engineering adaptations, not reproductions of
fungal morphogenesis or the ant experiment, and carry no stability guarantee.
The only online inputs are the v2 coordinator's cached telemetry and weather.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.special import ndtr, ndtri
from usv_sim.planning import Planner, freshness_probability, mission_weights, ring
from usv_sim.physics import footprints, radio_prob


@dataclass(frozen=True)
class ControlConfig:
    horizon_s: float = 300.0
    step_m: float = 120.0
    gradient_delta_m: float = 80.0
    energy_price: float = 0.025
    improvement_floor: float = 0.0005
    active_vessels: int = 2
    activity_memory_s: float = 120.0
    refractory_memory_s: float = 480.0
    social_gain: float = 0.5
    static_radius: float = 1.0
    def dict(self):
        return asdict(self)


def guidance_prediction(positions, goals, times, vmax=3.0, gain=0.025):
    """Closed-form radial guidance surrogate; ignores yaw/current/wave error.

    d'=-min(gain*d,vmax). Propulsion increment is integral 18*v^3 dt/3600.
    Station-keeping hotel power is common and excluded. This is a cost estimate,
    not the actual energy or a certified lower/upper bound in moving water.
    """
    delta = goals - positions
    distance = np.linalg.norm(delta, axis=1)
    unit = delta / np.maximum(distance[:, None], 1e-12)
    cruise = np.maximum(distance - vmax / gain, 0) / vmax
    initial_tail = np.minimum(distance, vmax / gain)
    predictions, energy = [], []
    for t in np.asarray(times):
        at_cruise = np.minimum(t, cruise)
        tail_time = np.maximum(t - cruise, 0)
        residual = np.maximum(distance - vmax * at_cruise - initial_tail *
                              (1 - np.exp(-gain * tail_time)), 0)
        predictions.append(positions + unit * (distance - residual)[:, None])
        joules = 18 * (vmax ** 3 * at_cruise + (gain * initial_tail) ** 3 *
                       (1 - np.exp(-3 * gain * tail_time)) / (3 * gain))
        energy.append(float(joules.sum() / 3600))
    return np.array(predictions), np.array(energy)


def hydraulic_potential(positions, severity, cfg, demand):
    """Grounded resistor analogy U=b^T L_g^-1 b and analytic spatial gradient.

    Conductance equals nominal per-round packet probability. The potential is
    NOT a deadline-delivery probability. Demand is held fixed for differentiation.
    """
    q = radio_prob(positions, severity, cfg)
    np.fill_diagonal(q, 0)
    lap = np.diag(q.sum(axis=1)) - q
    v = np.r_[np.linalg.solve(lap[:-1, :-1], demand), 0.0]
    nodes = np.vstack((positions, np.zeros((1, 2))))
    delta = nodes[:, None] - nodes[None, :]
    d = np.linalg.norm(delta, axis=-1)
    r = cfg.radio_r50_calm_m * np.exp(-cfg.radio_weather_slope * severity)
    # dq/dp_i = -0.995*6*d^4/(r^6*(1+(d/r)^6)^2) * (p_i-p_j)
    factor = -0.995 * 6 * d ** 4 / (r ** 6 * (1 + (d / r) ** 6) ** 2)
    grad = np.sum(-(v[:, None] - v[None, :])[:, :, None] ** 2 *
               factor[:, :, None] * delta, axis=1)[:-1]
    return float(demand @ v[:-1]), grad


class Evaluator(Planner):
    def paths(self, positions, moments, severity, wind, sensor_only=False):
        """Same joint visibility/network paths and deadline as v2's planner."""
        cfg, n, samples = self.cfg, self.cfg.n, len(self.draws)
        pd, _, a = footprints(positions, self.points, severity, moments[:, 0], moments[:, 2],
                              cfg, wind=wind, model="logistic")
        d2 = np.sum((positions[:, None] - positions[None, :]) ** 2, axis=-1)
        chol = np.linalg.cholesky(.35 * np.exp(-d2 / (2 * cfg.wave_correlation_m ** 2)) + .65 * np.eye(n))
        z = ndtri(self.draws[:, :n]) @ chol.T
        decay = np.exp(-cfg.dt_s / cfg.wave_memory_s)
        stay, enter = np.exp(-cfg.dt_s / 25), 1 - np.exp(-cfg.dt_s / 400)
        bad = self.draws[:, -1] < enter / (1 - stay + enter)
        q = radio_prob(positions, severity, cfg)
        visible, graphs, fades = [], [], []
        for k in range(self.window):
            u = self.path_draws[:, k + 1]
            z = decay * z + np.sqrt(1 - decay ** 2) * (ndtri(u[:, :n]) @ chol.T)
            visible.append(ndtr(z) < a)
            bad = u[:, -1] < np.where(bad, stay, enter)
            fades.append(np.where(bad, .25, 1.0))
            ok = u[:, n:-1] < q[self.edge_i, self.edge_j][None] * fades[-1][:, None]
            adj = np.broadcast_to(np.eye(n + 1, dtype=bool), (samples, n + 1, n + 1)).copy()
            adj[:, self.edge_i, self.edge_j] = ok
            adj[:, self.edge_j, self.edge_i] = ok
            if sensor_only:
                adj[:] = True
            graphs.append(adj)
        return pd, np.array(visible), np.array(graphs), np.array(fades)

    def field(self, positions, moments, severity, wind, sensor_only=False):
        pd, visible, graphs, _ = self.paths(positions, moments, severity, wind, sensor_only)
        return freshness_probability(pd, visible, graphs).mean(axis=0)

    def deadline_gradient(self, positions, moments, severity, wind, weights):
        pd, visible, graphs, fade = self.paths(positions, moments, severity, wind)
        importance = deadline_pivotality(pd, visible, graphs, fade, weights)
        nodes = np.vstack((positions, np.zeros((1, 2))))
        delta = nodes[:, None] - nodes[None, :]
        d = np.linalg.norm(delta, axis=-1)
        r = self.cfg.radio_r50_calm_m * np.exp(-self.cfg.radio_weather_slope * severity)
        dq = -.995 * 6 * d ** 4 / (r ** 6 * (1 + (d / r) ** 6) ** 2)
        return np.sum(importance[:, :, None] * dq[:, :, None] * delta, axis=1)[:-1]


def deadline_pivotality(pd, visible, graphs, fade, weights):
    """Conditional Bernoulli influence of a nominal link probability.

    Each q_ij is shared over the deadline window; realized success probability
    is fade[k,s]*q_ij. Other edges remain as sampled, including shared fades.
    Sum_k E[fade_k * (F(edge_k=1)-F(edge_k=0))]. Sensor footprints and visibility
    law are held fixed. This is a radio-only partial derivative, not total dF/dp.
    Scrambled Sobol paths approximate the expectation; no MC error guarantee.
    """
    nodes = graphs.shape[-1]
    result = np.zeros((nodes, nodes))
    for i in range(nodes):
        for j in range(i + 1, nodes):
            value = 0.0
            for k in range(len(graphs)):
                modified = graphs.copy()
                modified[k, :, i, j] = modified[k, :, j, i] = True
                up = freshness_probability(pd, visible, modified) @ weights
                modified[k, :, i, j] = modified[k, :, j, i] = False
                down = freshness_probability(pd, visible, modified) @ weights
                value += float(np.mean(fade[k] * (up - down)))
            result[i, j] = result[j, i] = value
    return result


def normalized(direction):
    scale = np.max(np.linalg.norm(direction, axis=1))
    return direction / max(scale, 1e-12)


class LocalPolicy:
    def __init__(self, name, cfg, settings=None):
        self.name, self.cfg = name, cfg
        self.settings = settings or ControlConfig()
        self.planner = Evaluator(cfg)
        self.current = 0  # v2 metadata compatibility; real decisions in diagnostics.
        self.action_counts = np.zeros(8, dtype=int)
        self.decisions = self.switches = 0
        self.goals = ring(cfg.n, cfg.mission_radius_m)
        self.activity = np.zeros(cfg.n)
        self.refractory = np.zeros(cfg.n)
        self.diagnostics = []

    def decide(self, positions, moments, ages, severity, wind, t, scenario, forced_action=None):
        cfg, c, name = self.cfg, self.settings, self.name
        self.decisions += 1
        self.action_counts[0] += 1
        if name == "static_tuned":
            self.goals = ring(cfg.n, cfg.mission_radius_m * c.static_radius)
            return self.goals.copy()
        if np.max(ages) > cfg.command_timeout_s:
            # Preserve previously issued goals; vessels independently hold on timeout.
            self.diagnostics.append(dict(t=t, accepted=False, stale_hold=True, proposed=0))
            return self.goals.copy()
        horizon = min(c.horizon_s, max(cfg.decision_s, cfg.duration_s - t))
        future_w = mission_weights(self.planner.angles, t + horizon / 2, cfg.duration_s,
                                   scenario, cfg.priority_turns)
        grad = np.zeros_like(positions)
        fungal = name.startswith("fungal") or name == "deadline_fusion"
        # Same 4N finite-difference evaluations in all proposal families.
        for i in range(cfg.n):
            for k in range(2):
                plus, minus = positions.copy(), positions.copy()
                plus[i, k] += c.gradient_delta_m
                minus[i, k] -= c.gradient_delta_m
                grad[i, k] = (self.planner.field(plus, moments, severity, wind, fungal) @ future_w -
                              self.planner.field(minus, moments, severity, wind, fungal) @ future_w) / (2 * c.gradient_delta_m)
        priority = np.linalg.norm(grad, axis=1)
        sparse = name in ("recruitment", "greedy_sparse", "recruitment_no_social")
        mask = np.ones(cfg.n)
        if sparse:
            drive = priority / max(priority.max(), 1e-12)
            if name.startswith("recruitment"):
                q = radio_prob(positions, severity, cfg)[:-1, :-1].copy()
                np.fill_diagonal(q, 0)
                q *= np.exp(-ages[None, :] / cfg.telemetry_ttl_s)
                q /= np.maximum(q.sum(axis=1, keepdims=True), 1e-12)
                social = 0 if name == "recruitment_no_social" else c.social_gain
                decay = np.exp(-cfg.decision_s / c.activity_memory_s)
                target = np.tanh(2 * drive + social * q @ self.activity - .6 * self.refractory)
                self.activity = decay * self.activity + (1 - decay) * np.maximum(target, 0)
                priority = self.activity
            mask[:] = 0
            mask[np.argsort(-priority, kind="stable")[:c.active_vessels]] = 1
        radial = positions / np.maximum(np.linalg.norm(positions, axis=1)[:, None], 1)
        directions = []
        if fungal:
            pd, _, av = footprints(positions, self.planner.points, severity, moments[:, 0], moments[:, 2], cfg, wind)
            # Frozen demand weights unique instantaneous sensing contribution.
            p = av[:, None] * pd
            miss = np.prod(1 - p, axis=0)
            demand = (p * miss[None] / np.maximum(1 - p, 1e-8)) @ future_w + 1e-5
            demand /= demand.sum()
            _, cost_gradient = hydraulic_potential(positions, severity, cfg, demand)
            branch, fuse = normalized(grad), normalized(-cost_gradient)
            if name == "deadline_fusion":
                fuse = normalized(self.planner.deadline_gradient(positions, moments, severity, wind, future_w))
            if name == "fungal_no_fusion":
                # Same proposal count, varying displacement instead of fusion mixture.
                directions = [branch * s for s in (.25, .4, .55, .7, .85, 1.0)]
            else:
                directions = [normalized((1 - f) * branch + f * fuse) for f in (0, .2, .4, .6, .8, 1)]
        else:
            direction = normalized(grad * mask[:, None])
            top = np.zeros(cfg.n)
            top[np.argsort(-priority, kind="stable")[:max(1, min(2, cfg.n))]] = 1
            rotation = np.stack((-direction[:, 1], direction[:, 0]), axis=1)
            directions = [direction * .5, direction, normalized(grad * top[:, None] * mask[:, None]),
                          normalized(direction + .35 * rotation), normalized(direction - .35 * rotation),
                          direction * .25]
        directions.extend([radial * mask[:, None], -radial * mask[:, None]])
        proposals = [self.goals.copy(), positions.copy()]
        for d in directions:
            goal = positions + c.step_m * d
            if sparse:
                goal[mask == 0] = self.goals[mask == 0]
            proposals.append(goal)
        # Same ten prospective trajectories / three quadrature samples per policy.
        times = horizon * np.array([1 / 6, 1 / 2, 5 / 6])
        weights = [mission_weights(self.planner.angles, t + x, cfg.duration_s, scenario, cfg.priority_turns) for x in times]
        scores, covers, energies = [], [], []
        for goal in proposals:
            path, energy = guidance_prediction(positions, goal, np.r_[times, horizon], cfg.max_speed_mps)
            # Pair separation along the predicted path; local avoidance remains authoritative.
            distance = np.linalg.norm(path[:, :, None] - path[:, None, :], axis=-1)
            for d in distance:
                np.fill_diagonal(d, np.inf)
            if np.min(distance) < 220 or np.max(np.linalg.norm(goal, axis=1)) > 1.4 * cfg.mission_radius_m:
                scores.append(-1e6); covers.append(0.0); energies.append(float(energy[-1])); continue
            cover = np.mean([self.planner.field(p, moments, severity, wind) @ w for p, w in zip(path[:3], weights)])
            price = 0 if name in ("economic_no_cost", "fungal_no_stop") else c.energy_price
            scores.append(float(cover - price * energy[-1] / (cfg.n * 100 * horizon / 3600)))
            covers.append(float(cover)); energies.append(float(energy[-1]))
        chosen = int(np.argmax(scores))
        floor = 0 if name == "fungal_no_stop" else c.improvement_floor
        if scores[chosen] <= scores[0] + floor:
            chosen = 0
        changed = np.linalg.norm(proposals[chosen] - self.goals, axis=1) > 1
        if chosen:
            self.switches += 1
            self.goals = proposals[chosen].copy()
        dh = np.exp(-cfg.decision_s / c.refractory_memory_s)
        self.refractory = dh * self.refractory + (1 - dh) * changed
        self.diagnostics.append(dict(t=t, accepted=bool(chosen), stale_hold=False, proposed=len(proposals),
                                     proposal=chosen, predicted_gain=scores[chosen] - scores[0],
                                     predicted_coverage=covers[chosen], predicted_energy_wh=energies[chosen],
                                     moving_vessels=int(changed.sum()), activity=self.activity.tolist()))
        return self.goals.copy()
