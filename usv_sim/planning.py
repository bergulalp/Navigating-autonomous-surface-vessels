"""Finite formation library, joint Monte Carlo score, information and memory policies."""
import numpy as np
from scipy.special import expit, logit, ndtr, ndtri
from scipy.stats import qmc
from scipy.optimize import linear_sum_assignment
from .physics import footprints, radio_prob, nominal_sensor, reachable

ACTION_NAMES = ("ring", "contract", "health_balance", "ellipse", "one_relay", "two_relays",
                "priority", "calibrate")


def freshness_probability(pd, visible, graphs):
    """Exact detector-noise integral conditional on wave/network sample paths.

    pd: [observer, point]; visible: [time, sample, observer];
    graphs: [time, sample, node, node]. Backward reachability answers whether a
    report generated at each time can reach the base by the common deadline.
    Reports can wait in a cache; every forwarding hop still costs a time step.
    """
    samples, n = visible.shape[1:]
    future = np.zeros((samples, n + 1), dtype=bool)
    future[:, -1] = True
    miss = np.ones((samples, pd.shape[1]))
    for k in range(len(graphs) - 1, -1, -1):
        future = np.any(graphs[k] & future[:, None, :], axis=-1) | future
        eligible = visible[k] & future[:, :n]
        miss *= np.prod(1 - pd[None] * eligible[:, :, None], axis=1)
    return 1 - miss


def ring(n, radius, phase=0.0):
    a = phase + np.arange(n) * 2 * np.pi / n
    return radius * np.stack((np.cos(a), np.sin(a)), axis=1)


def mission_grid(cfg):
    angles = np.arange(cfg.grid_angles) * 2 * np.pi / cfg.grid_angles
    # Quadrature of a specified annular monitoring distribution, not a Cartesian area grid.
    radii = cfg.mission_radius_m * np.array([0.90, 1.0, 1.10])
    points = np.concatenate([r * np.stack((np.cos(angles), np.sin(angles)), axis=1) for r in radii])
    return points, np.tile(angles, len(radii))


def mission_weights(angles, t, duration, scenario, turns=1.0):
    if scenario == "moving_front":
        focus = 2 * np.pi * turns * t / duration
        w = 0.35 + 2.5 * np.exp(3 * (np.cos(angles - focus) - 1))
    else:
        w = np.ones_like(angles)
    return w / w.sum()


def match_slots(slots, positions):
    cost = np.linalg.norm(positions[:, None, :] - slots[None, :, :], axis=-1)
    i, j = linear_sum_assignment(cost)
    ordered = np.empty_like(slots)
    ordered[i] = slots[j]
    return ordered


def candidates(positions, moments, severity, wind, t, scenario, cfg):
    n, r = cfg.n, cfg.mission_radius_m
    slots = [ring(n, r), ring(n, 0.78 * r), ring(n, 1.12 * r)]
    ell = ring(n, 1.0)
    ell *= np.array([1.12 * r, 0.76 * r])
    rot = np.array([[np.cos(wind), -np.sin(wind)], [np.sin(wind), np.cos(wind)]])
    slots.append(ell @ rot.T)
    slots.append(np.vstack((ring(n - 1, r, np.pi / (n - 1)), [[0.45 * r, 0]])))
    slots.append(np.vstack((ring(n - 2, r, np.pi / (n - 2)), [[0.42 * r, 0], [-0.42 * r, 0]])))
    aa = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    ww = mission_weights(aa, t, cfg.duration_s, scenario, cfg.priority_turns)
    quant = np.interp((np.arange(n) + 0.5) / n, np.cumsum(ww), aa)
    slots.append(r * np.stack((np.cos(quant), np.sin(quant)), axis=1))
    goals = [match_slots(s, positions) for s in slots]
    # Assign less angular territory to a less capable observer, retaining order.
    # A weighted partition is an engineering baseline, not claimed biological novelty.
    radar, camera, a0 = nominal_sensor(severity, cfg)
    health = expit(logit(a0) + moments[:, 2]) * (radar * moments[:, 0] + 0.25 * camera)
    order = np.argsort(np.arctan2(positions[:, 1], positions[:, 0]))
    widths = np.maximum(health[order], 0.2 * np.mean(health))
    gaps = (widths + np.roll(widths, -1)) * np.pi / widths.sum()
    proposed = np.r_[0., np.cumsum(gaps[:-1])]
    old = np.arctan2(positions[order, 1], positions[order, 0])
    phase = np.angle(np.mean(np.exp(1j * (old - proposed))))
    repair = np.empty_like(positions)
    repair[order] = r * np.stack((np.cos(proposed + phase), np.sin(proposed + phase)), axis=1)
    goals[2] = repair
    # Costed rendezvous: preserve other stations, select uncertain observer and
    # its nearest reference; both move symmetrically towards an informative range.
    uncertain = moments[:, 1] / 0.25 + moments[:, 3] / 0.7
    i = int(np.argmax(uncertain))
    dist = np.linalg.norm(positions - positions[i], axis=1)
    dist[i] = np.inf
    j = int(np.argmin(dist))
    radar, _, _ = nominal_sensor(severity, cfg)
    separation = np.clip(0.85 * radar * moments[i, 0], 250, 1400)
    center = 0.5 * (positions[i] + positions[j])
    direction = (positions[j] - positions[i]) / max(dist[j], 1)
    cal = positions.copy()
    cal[i], cal[j] = center - direction * separation / 2, center + direction * separation / 2
    goals.append(cal)
    return np.array(goals)


def information_score(goals, moments, severity, cfg):
    """Local Gaussian Fisher information proxy, retaining parameter covariance.

    Log-determinant couples availability/range identifiability. This is a planning
    heuristic, not the expected mission value of calibration or a dual-control proof.
    """
    radar, _, a0 = nominal_sensor(severity, cfg)
    rs, sr, off, so, cov = moments[:, :5].T
    d = np.linalg.norm(goals[:, None, :] - goals[None, :, :], axis=-1)
    r = radar * rs[:, None]
    z = (r - d) / (0.2 * r)
    f, a = expit(z), expit(logit(a0) + off)[:, None]
    p = a * 0.94 * f
    gr = a * 0.94 * f * (1 - f) * d / (0.2 * radar * rs[:, None] ** 2)
    ga = a * (1 - a) * 0.94 * f
    grad = np.stack((gr, ga), axis=-1)
    grad[np.arange(cfg.n), np.arange(cfg.n)] = 0
    fisher = np.einsum("ija,ijb,ij->iab", grad, grad, 1 / (p * (1 - p) + 1e-4))
    sigma = np.zeros((cfg.n, 2, 2))
    sigma[:, 0, 0], sigma[:, 1, 1] = sr ** 2, so ** 2
    sigma[:, 0, 1] = sigma[:, 1, 0] = np.clip(cov, -0.98 * sr * so, 0.98 * sr * so)
    # Three effective independent reference scans in a prospective short dwell.
    det = np.linalg.det(np.eye(2)[None] + 3 * np.matmul(sigma, fisher))
    return float(np.mean(0.5 * np.log(np.maximum(det, 1))))


class Planner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.points, self.angles = mission_grid(cfg)
        m = int(np.ceil(np.log2(cfg.planner_samples)))
        size = cfg.n + (cfg.n + 1) * cfg.n // 2 + 1
        # Independent planning integration points, never simulation's random tape.
        self.window = int(cfg.report_ttl_s / cfg.dt_s) + 1
        draws = qmc.Sobol(size * (self.window + 1), scramble=True, seed=1701).random_base2(m)[:cfg.planner_samples]
        self.path_draws = np.clip(draws.reshape(-1, self.window + 1, size), 1e-6, 1 - 1e-6)
        self.draws = self.path_draws[:, 0]
        self.edge_i, self.edge_j = np.triu_indices(cfg.n + 1, 1)

    def coverage_score(self, positions, moments, severity, wind, weights):
        if self.cfg.temporal_planner:
            return self.temporal_score(positions, moments, severity, wind, weights)
        return self.snapshot_score(positions, moments, severity, wind, weights)

    def temporal_score(self, positions, moments, severity, wind, weights):
        cfg, n, samples = self.cfg, self.cfg.n, len(self.draws)
        pd, _, a = footprints(positions, self.points, severity, moments[:, 0], moments[:, 2], cfg,
                              wind=wind, model="logistic")
        d2 = np.sum((positions[:, None] - positions[None, :]) ** 2, axis=-1)
        covariance = 0.35 * np.exp(-d2 / (2 * cfg.wave_correlation_m ** 2)) + 0.65 * np.eye(n)
        chol = np.linalg.cholesky(covariance)
        z = ndtri(self.draws[:, :n]) @ chol.T
        wave_decay = np.exp(-cfg.dt_s / cfg.wave_memory_s)
        bad_stay = np.exp(-cfg.dt_s / 25)
        good_to_bad = 1 - np.exp(-cfg.dt_s / 400)
        stationary_bad = good_to_bad / (1 - bad_stay + good_to_bad)
        bad = self.draws[:, -1] < stationary_bad
        q = radio_prob(positions, severity, cfg)
        visible, graphs = [], []
        for k in range(self.window):
            u = self.path_draws[:, k + 1]
            z = wave_decay * z + np.sqrt(1 - wave_decay ** 2) * (ndtri(u[:, :n]) @ chol.T)
            visible.append(ndtr(z) < a)
            bad = u[:, -1] < np.where(bad, bad_stay, good_to_bad)
            fade = np.where(bad, .25, 1.0)
            ok = u[:, n:-1] < q[self.edge_i, self.edge_j][None] * fade[:, None]
            adj = np.broadcast_to(np.eye(n + 1, dtype=bool), (samples, n + 1, n + 1)).copy()
            adj[:, self.edge_i, self.edge_j] = ok
            adj[:, self.edge_j, self.edge_i] = ok
            graphs.append(adj)
        return float(np.mean(freshness_probability(pd, np.array(visible), np.array(graphs)) @ weights))

    def snapshot_score(self, positions, moments, severity, wind, weights):
        cfg, n, samples = self.cfg, self.cfg.n, len(self.draws)
        pd, _, a = footprints(positions, self.points, severity, moments[:, 0], moments[:, 2], cfg,
                              wind=wind, model="logistic")
        d2 = np.sum((positions[:, None] - positions[None, :]) ** 2, axis=-1)
        covariance = 0.35 * np.exp(-d2 / (2 * cfg.wave_correlation_m ** 2)) + 0.65 * np.eye(n)
        wave = ndtri(self.draws[:, :n]) @ np.linalg.cholesky(covariance).T
        visible = ndtr(wave) < a
        q = radio_prob(positions, severity, cfg)
        # Integrate shared channel fade; within a sample, shared edges/routes are
        # represented explicitly, avoiding multiplication of marginal path success.
        q = np.broadcast_to(q, (samples, n + 1, n + 1)).copy()
        fade = np.where(self.draws[:, -1] < 0.06, 0.25, 1.0)
        q *= fade[:, None, None]
        adj = np.broadcast_to(np.eye(n + 1, dtype=bool), q.shape).copy()
        ok = self.draws[:, n:-1] < q[:, self.edge_i, self.edge_j]
        adj[:, self.edge_i, self.edge_j] = ok
        adj[:, self.edge_j, self.edge_i] = ok
        connected = reachable(adj, min(n, int(cfg.report_ttl_s / cfg.dt_s)))[:, :n]
        joint = visible & connected
        union = 1 - np.prod(1 - pd[None] * joint[:, :, None], axis=1)
        # Snapshot graph/one-scan surrogate; actual delivered freshness is evaluated
        # with time-varying links, retries and one-hop-per-step forwarding in World.
        return float(np.mean(union @ weights))

    def scores(self, positions, moments, ages, severity, wind, t, scenario, information=False):
        cfg = self.cfg
        goals = candidates(positions, moments, severity, wind, t, scenario, cfg)
        weights = mission_weights(self.angles, t, cfg.duration_s, scenario, cfg.priority_turns)
        cover = np.array([self.coverage_score(g, moments, severity, wind, weights) for g in goals])
        if cfg.transit_lookahead_s > 0:
            delta = goals - positions
            distance = np.linalg.norm(delta, axis=-1)
            fraction = np.minimum(1, cfg.max_speed_mps * cfg.transit_lookahead_s / np.maximum(distance, 1))
            attainable = positions + delta * fraction[..., None]
            # Forecast only the explicitly scheduled monitoring priority. Weather
            # is persisted from its current observation, not read from future truth.
            future_w = mission_weights(self.angles, t + cfg.transit_lookahead_s, cfg.duration_s,
                                       scenario, cfg.priority_turns)
            transit = np.array([self.coverage_score(g, moments, severity, wind, future_w) for g in attainable])
            cover = 0.5 * cover + 0.5 * transit
        movement = np.mean(np.linalg.norm(goals - positions, axis=-1), axis=1) / cfg.mission_radius_m
        info = np.array([information_score(g, moments, severity, cfg) for g in goals]) if information else np.zeros(len(goals))
        # Information obtained after travel is worth less within a finite horizon.
        info *= np.exp(-movement * cfg.mission_radius_m / (cfg.max_speed_mps * 600))
        scores = cover - cfg.movement_weight * movement + cfg.information_weight * info
        # Stale-state uncertainty is not magically removed by extrapolating poses.
        # Every policy shares a conservative suppression of ambitious motion when stale.
        scores -= 0.015 * np.mean(np.clip(ages / cfg.command_timeout_s, 0, 2)) * movement
        return goals, scores, cover, info


class Policy:
    def __init__(self, name, cfg, rng=None):
        self.name, self.cfg = name, cfg
        self.planner = Planner(cfg)
        self.rng = rng or np.random.default_rng(0)
        self.current = 0
        self.last_switch = -np.inf
        self.evidence = np.zeros(len(ACTION_NAMES))
        self.habituation = np.zeros(len(ACTION_NAMES))
        self.switches = 0
        self.decisions = 0
        self.action_counts = np.zeros(len(ACTION_NAMES), dtype=int)
        self.fixed_goals = None
        self.current_goals = None
        self.last_scores = np.zeros(len(ACTION_NAMES))

    def decide(self, positions, moments, ages, severity, wind, t, scenario, forced_action=None):
        cfg, name = self.cfg, self.name
        mm = moments.copy()
        if name in ("model", "fixed_robust"):
            mm[:, 0], mm[:, 2] = 1, 0
        active = name in ("active", "periodic", "hysteresis", "bio", "bio_no_habituation") or forced_action is not None
        if forced_action is not None:
            chosen = int(forced_action)
            if not 0 <= chosen < len(ACTION_NAMES):
                raise ValueError("Action must be in 0..7")
            goals = candidates(positions, mm, severity, wind, t, scenario, cfg)[chosen]
            if chosen == self.current == 7 and self.current_goals is not None and t - self.last_switch < 240:
                goals = self.current_goals.copy()
        elif name == "fixed_ring":
            if self.fixed_goals is None:
                self.fixed_goals = ring(cfg.n, cfg.mission_radius_m)
            chosen, goals = 0, self.fixed_goals
        elif name == "fixed_robust" and self.fixed_goals is not None:
            chosen, goals = self.current, self.fixed_goals
        else:
            candidate_goals, scores, cover, info = self.planner.scores(positions, mm, ages, severity, wind, t, scenario, active)
            if name == "fixed_robust":
                # Nominal static design over three declared synthetic severities;
                # no test-seed optimization and no claim of a measured climate.
                weights = np.ones(len(self.planner.angles)) / len(self.planner.angles)
                scores = np.array([np.mean([self.planner.coverage_score(g, mm, s, 0, weights)
                                           for s in (0.15, 0.5, 0.85)]) for g in candidate_goals])
                scores[7] = -np.inf
            elif not active:
                scores[7] = -np.inf
            self.last_scores = np.nan_to_num(scores, neginf=-1)
            best = int(np.argmax(scores))
            chosen = best
            if forced_action is not None:
                chosen = int(forced_action)
            elif name == "random":
                chosen = int(self.rng.integers(0, len(ACTION_NAMES)))
            elif name == "periodic":
                in_window = 600 <= (t % 1800) < 1200
                scores[7] = -np.inf
                chosen = 7 if in_window else int(np.argmax(scores))
            elif name in ("bio", "bio_no_habituation"):
                gain = self.last_scores - self.last_scores[self.current]
                decay = np.exp(-cfg.decision_s / cfg.evidence_memory_s)
                self.evidence = decay * self.evidence + (1 - decay) * gain
                dh = np.exp(-cfg.decision_s / cfg.habituation_memory_s)
                surprise = np.clip(np.mean(mm[:, 5]) - 0.75, 0, 1.5)
                self.habituation *= dh
                self.habituation[self.current] += (1 - dh) / (1 + surprise)
                h = self.habituation if name == "bio" else np.zeros_like(self.habituation)
                threshold = cfg.switch_threshold * (1 + cfg.habituation_gain * h) / (1 + surprise)
                drive = self.evidence - threshold
                best = int(np.argmax(drive))
                chosen = best if drive[best] > 0 and t - self.last_switch >= cfg.minimum_dwell_s else self.current
                # A large immediate predicted loss overrides accumulated memory.
                immediate = int(np.argmax(scores))
                if scores[immediate] - scores[self.current] > 0.12:
                    chosen = immediate
            elif name == "hysteresis":
                if scores[best] - scores[self.current] <= cfg.switch_threshold or t - self.last_switch < cfg.minimum_dwell_s:
                    chosen = self.current
            goals = candidate_goals[chosen]
            if name == "fixed_robust":
                self.fixed_goals = goals.copy()
            # A rendezvous must persist long enough to be physically executed;
            # recomputing the pair every minute would chase a moving target.
            if chosen == self.current == 7 and self.current_goals is not None and t - self.last_switch < 240:
                goals = self.current_goals.copy()
        if chosen != self.current:
            self.switches += 1
            self.last_switch = t
            self.evidence[:] = 0
        self.current = chosen
        self.current_goals = goals.copy()
        self.decisions += 1
        self.action_counts[chosen] += 1
        return goals
