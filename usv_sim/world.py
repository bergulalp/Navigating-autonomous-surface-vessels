"""Paired stochastic episodes with causal telemetry, commands and report routing."""
import hashlib
import time
import numpy as np
from scipy.special import ndtr
from .belief import SensorBelief
from .physics import (severity_at, footprints, radio_prob, reachable, forward_latest, Fleet)
from .planning import Policy, ring, mission_grid, mission_weights
from .config import run_id, source_hash


class World:
    def __init__(self, cfg, scenario="storm", seed=0, policy="bio", trace=False):
        self.cfg, self.scenario, self.seed = cfg.validate(), scenario, int(seed)
        self.policy_name = policy
        self.keep_trace = trace
        streams = np.random.SeedSequence([self.seed, 92741]).spawn(8)
        hardware, weather, target, sensor, channel, motion, policy_rng, field = [np.random.default_rng(s) for s in streams]
        self.sensor_rng, self.channel_rng, self.motion_rng = sensor, channel, motion
        n = cfg.n
        self.true_range = hardware.uniform(0.82, 1.18, n)
        self.true_offset = hardware.normal(0, 0.25, n)
        self.true_camera = hardware.uniform(0.9, 1.1, n)
        self.true_radio = float(hardware.uniform(0.92, 1.08))
        self.initial_range = self.true_range.copy()
        self.initial_offset = self.true_offset.copy()
        self.steps = int(round(cfg.duration_s / cfg.dt_s))
        self.weather_noise = weather.normal(0, 0.06, self.steps + 1)
        self.wind = float(weather.uniform(-np.pi, np.pi))
        # Random Fourier wave field: same exogenous field, evaluated at each
        # controller's physical positions. Idiosyncratic vessel response is separate.
        self.wave_k = field.normal(0, 1 / cfg.wave_correlation_m, (12, 2))
        self.wave_coeff = field.normal(size=24)
        self.wave_id = field.normal(size=n)
        self.wave_rng = field
        self.fade_bad = False
        self.fleet = Fleet(ring(n, cfg.mission_radius_m), cfg)
        self.local_goals = self.fleet.p.copy()
        self.hold_points = self.fleet.p.copy()
        self.was_stale = np.zeros(n, dtype=bool)
        self.beliefs = [SensorBelief() for _ in range(n)]
        self.policy = Policy(policy, cfg, policy_rng)
        self.points, self.angles = mission_grid(cfg)
        self.m = len(self.points)
        k = cfg.target_count
        self.target_speed = target.uniform(4, 7, k)
        travel_to_cross = (4000 - cfg.crossing_radius_m) / self.target_speed
        last_birth = np.maximum(0, cfg.duration_s - travel_to_cross - cfg.dt_s)
        self.target_birth = target.uniform(size=k) * last_birth
        self.target_angle = target.uniform(0, 2 * np.pi, k)
        # Same monitoring reference traffic for all controllers. In moving_front,
        # some tracks follow the declared moving priority, others remain uniform.
        if scenario == "moving_front":
            select = target.uniform(size=k) < 0.65
            self.target_angle[select] = 2 * np.pi * cfg.priority_turns * self.target_birth[select] / cfg.duration_s + target.normal(0, 0.5, select.sum())
        self.target_direction = np.stack((np.cos(self.target_angle), np.sin(self.target_angle)), axis=1)
        self.target_crossing = self.target_birth + travel_to_cross
        self.target_eligible = (self.target_crossing >= cfg.warmup_s + cfg.required_lead_s) & (self.target_crossing <= cfg.duration_s)
        self.first_local = np.full(k, np.inf)
        self.first_report = np.full(k, np.inf)
        self.probe_times = np.full((n + 1, self.m), -np.inf)
        self.target_times = np.full((n + 1, k), -np.inf)
        # Telemetry fields: x,y, range mean/std, availability-offset mean/std,
        # covariance, predictive surprise, observation count, energy Wh.
        self.telemetry_times = np.full((n + 1, n), -np.inf)
        self.telemetry = np.zeros((n + 1, n, 10))
        for i, b in enumerate(self.beliefs):
            self.telemetry[:, i, :2] = self.fleet.p[i]
            self.telemetry[:, i, 2:9] = b.moments()
        # A predeployment manifest gives initial poses/prior; no later free state.
        self.telemetry_times[:] = 0
        self.command_times = np.full((n + 1, n), -np.inf)
        self.commands = np.zeros((n + 1, n, 2))
        self.commands[:] = self.fleet.p
        self.command_times[:] = 0
        self.t, self.index = 0.0, 0
        self.logs = []
        self.trace = []
        self.start_clock = time.perf_counter()
        tape = np.concatenate([self.initial_range, self.initial_offset, self.weather_noise,
                               self.target_speed, self.target_birth, self.target_angle])
        self.exogenous_sha256 = hashlib.sha256(tape.tobytes()).hexdigest()

    def observation(self):
        """Only cached state available at the coordinator, plus current weather proxy."""
        cfg, n = self.cfg, self.cfg.n
        severity = severity_at(self.t, cfg.duration_s, self.scenario)
        observed_weather = np.clip(severity + self.weather_noise[self.index], 0, 1)
        positions = self.telemetry[n, :, :2].copy()
        moments = self.telemetry[n, :, 2:9].copy()
        ages = self.t - self.telemetry_times[n]
        # Inflate cached uncertainty with age; an old confident report is not fresh evidence.
        moments[:, 1] = np.sqrt(moments[:, 1] ** 2 + (cfg.range_process_std ** 2) * np.maximum(ages, 0))
        moments[:, 3] = np.sqrt(moments[:, 3] ** 2 + (cfg.availability_process_std ** 2) * np.maximum(ages, 0))
        if self.policy_name == "truth_informed":
            # Privileged health comparator only; same delayed poses, commands,
            # action library and local planner. Not an optimality bound.
            moments[:, 0], moments[:, 2] = self.true_range, self.true_offset
            moments[:, 1], moments[:, 3], moments[:, 4] = 0, 0, 0
        return positions, moments, ages, float(observed_weather)

    def features(self):
        p, m, age, severity = self.observation()
        n = self.cfg.n
        energy = self.telemetry[n, :, 9]
        return np.array([severity, np.mean(m[:, 0]), np.mean(m[:, 1]), np.mean(m[:, 2]),
                         np.mean(m[:, 3]), np.mean(m[:, 5]), np.mean(np.clip(age / 180, 0, 2)),
                         np.mean(np.linalg.norm(p, axis=1)) / self.cfg.mission_radius_m,
                         np.mean(energy) / self.cfg.battery_wh, self.t / self.cfg.duration_s,
                         self.policy.current / 7.0, np.sin(self.wind), np.cos(self.wind)], dtype=np.float32)

    def control(self, forced_action=None):
        p, m, ages, observed = self.observation()
        goals = self.policy.decide(p, m, ages, observed, self.wind, self.t, self.scenario, forced_action)
        self.commands[-1] = goals
        self.command_times[-1] = self.t

    def step(self):
        if self.index >= self.steps:
            raise RuntimeError("Episode has ended; construct or reset a World")
        cfg, n, dt = self.cfg, self.cfg.n, self.cfg.dt_s
        self.index += 1
        self.t = t = self.index * dt
        severity = severity_at(t, cfg.duration_s, self.scenario)
        if self.scenario == "sensor_fault" and t >= 0.4 * cfg.duration_s:
            self.true_range[0] = 0.43 * self.initial_range[0]
            self.true_offset[0] = self.initial_offset[0] - 1.15
        own_command_age = t - self.command_times[np.arange(n), np.arange(n)]
        stale = own_command_age > cfg.command_timeout_s
        entering = stale & ~self.was_stale
        self.hold_points[entering] = self.fleet.p[entering]
        self.local_goals = self.commands[np.arange(n), np.arange(n)].copy()
        self.local_goals[stale] = self.hold_points[stale]
        self.was_stale = stale
        current = np.array([0.20 + 0.15 * np.sin(t / 500), 0.10 * np.cos(t / 350)])
        current += np.array([0.12, -0.08]) * severity
        wave_force = self.motion_rng.normal(size=n)
        self.fleet.step(self.local_goals, current, wave_force, severity)
        p = self.fleet.p
        decay = np.exp(-dt / cfg.wave_memory_s)
        self.wave_coeff = decay * self.wave_coeff + np.sqrt(1 - decay ** 2) * self.wave_rng.normal(size=24)
        self.wave_id = decay * self.wave_id + np.sqrt(1 - decay ** 2) * self.wave_rng.normal(size=n)
        phase = p @ self.wave_k.T
        basis = np.concatenate((np.cos(phase), np.sin(phase)), axis=1) / np.sqrt(12)
        wave_z = np.sqrt(0.35) * (basis @ self.wave_coeff) + np.sqrt(0.65) * self.wave_id
        target_radius = 4000 - self.target_speed * (t - self.target_birth)
        tp = self.target_direction * target_radius[:, None]
        target_active = (t >= self.target_birth) & (target_radius >= -4000)
        all_points = np.vstack((self.points, tp, p))
        fused, radar, a = footprints(p, all_points, severity, self.true_range, self.true_offset, cfg,
                                    self.wind, self.true_camera)
        visible = ndtr(wave_z) < a
        # Fixed dimensions and separate streams prevent controller-dependent tape shifts.
        u = self.sensor_rng.uniform(size=fused.shape)
        detected = (u < fused) & visible[:, None]
        # Reference observations use radar alone, with independent detector noise
        # but the SAME obscuration state as the monitoring sensors.
        ref_u = self.sensor_rng.uniform(size=(n, n))
        ref_y = (ref_u < radar[:, -n:]) & visible[:, None]
        detected[:, self.m:self.m + cfg.target_count] &= target_active[None]
        self.probe_times[:n] = np.where(detected[:, :self.m], t, self.probe_times[:n])
        target_detected = detected[:, self.m:self.m + cfg.target_count]
        self.target_times[:n] = np.where(target_detected, t, self.target_times[:n])
        self.first_local = np.where(np.any(target_detected, axis=0) & np.isinf(self.first_local), t, self.first_local)
        # Reference identity/position must have arrived in a recent beacon; no
        # packet is treated as a radar miss. Simulated references share target class.
        for i, belief in enumerate(self.beliefs):
            belief.predict(dt, cfg)
            fresh = (t - self.telemetry_times[i]) <= 2 * dt
            fresh[i] = False
            reference_position = self.telemetry[i, fresh, :2]
            delta = reference_position - p[i]
            d = np.linalg.norm(delta, axis=1)
            angle = np.arctan2(delta[:, 1], delta[:, 0])
            observed_weather = float(np.clip(severity + self.weather_noise[self.index], 0, 1))
            belief.update(d, angle, ref_y[i, fresh], observed_weather, self.wind, cfg)
            self.telemetry_times[i, i] = t
            self.telemetry[i, i] = np.r_[p[i], belief.moments(), self.fleet.energy[i]]
        # Two-state burst process; common fade couples links in a scan.
        fade_u = self.channel_rng.uniform()
        self.fade_bad = fade_u < (np.exp(-dt / 25) if self.fade_bad else 1 - np.exp(-dt / 400))
        q = radio_prob(p, severity, cfg, self.true_radio, 0.25 if self.fade_bad else 1)
        blackout = cfg.full_blackout or (self.scenario == "blackout" and 0.35 <= t / cfg.duration_s <= 0.55)
        if blackout:
            q[:] = 0
            np.fill_diagonal(q, 1)
        link_u = self.channel_rng.uniform(size=(n + 1, n + 1))
        adj = np.triu(link_u < q, 1)
        adj = adj | adj.T | np.eye(n + 1, dtype=bool)
        self.probe_times, _ = forward_latest(self.probe_times, None, adj)
        self.target_times, _ = forward_latest(self.target_times, None, adj)
        self.telemetry_times, self.telemetry = forward_latest(self.telemetry_times, self.telemetry, adj)
        self.command_times, self.commands = forward_latest(self.command_times, self.commands, adj)
        valid_report = (t - self.target_times[-1]) <= cfg.report_ttl_s
        self.first_report = np.where(valid_report & np.isinf(self.first_report), t, self.first_report)
        w = mission_weights(self.angles, t, cfg.duration_s, self.scenario, cfg.priority_turns)
        fresh_local = np.any((t - self.probe_times[:n]) <= cfg.report_ttl_s, axis=0)
        fresh_base = (t - self.probe_times[-1]) <= cfg.report_ttl_s
        h_cond = 1 - np.prod(1 - fused[:, :self.m] * visible[:, None], axis=0)
        moments = np.array([b.moments() for b in self.beliefs])
        cover_range = np.mean([b.interval(b.rs)[0] <= r <= b.interval(b.rs)[1]
                               for b, r in zip(self.beliefs, self.true_range)])
        cover_offset = np.mean([b.interval(b.off)[0] <= aoff <= b.interval(b.off)[1]
                                for b, aoff in zip(self.beliefs, self.true_offset)])
        row = dict(t=t, delivered=float(w @ fresh_base), local_fresh=float(w @ fresh_local),
                   raw_scan=float(w @ np.any(detected[:, :self.m], axis=0)),
                   conditional_expected=float(w @ h_cond),
                   graph_connected=float(np.all(reachable(adj, n))),
                   telemetry_stale=float(np.mean(t - self.telemetry_times[-1] > cfg.telemetry_ttl_s)),
                   command_stale=float(np.mean(stale)),
                   range_rmse=float(np.sqrt(np.mean((moments[:, 0] - self.true_range) ** 2))),
                   offset_rmse=float(np.sqrt(np.mean((moments[:, 2] - self.true_offset) ** 2))),
                   range_interval90=float(cover_range), offset_interval90=float(cover_offset),
                   energy_wh=float(self.fleet.energy.sum()), severity=float(severity), action=self.policy.current)
        self.logs.append(row)
        if self.keep_trace and self.index % max(1, int(cfg.decision_s / dt)) == 0:
            self.trace.append(dict(**row, positions=p.copy(), range_mean=moments[:, 0].copy(),
                                   range_truth=self.true_range.copy(), goals=self.local_goals.copy()))
        return row

    def step_decision(self, action=None):
        """One high-level action for RL; physical substeps and network are unchanged."""
        before_energy = self.fleet.energy.sum()
        self.control(action)
        rows = []
        count = min(int(round(self.cfg.decision_s / self.cfg.dt_s)), self.steps - self.index)
        for _ in range(count):
            rows.append(self.step())
        reward = float(np.mean([r["delivered"] for r in rows])) if rows else 0.0
        # Energy per boat normalized by a declared 100-W reference for this interval.
        reference_wh = self.cfg.n * 100 * self.cfg.decision_s / 3600
        reward -= 0.08 * (self.fleet.energy.sum() - before_energy) / reference_wh
        if rows:
            reward -= 0.10 * np.mean([r["command_stale"] for r in rows])
        return self.features(), reward, self.index >= self.steps

    def run(self, learned_policy=None):
        while self.index < self.steps:
            action = None if learned_policy is None else learned_policy(self.features())
            self.step_decision(action)
        return self.result()

    def result(self):
        cfg = self.cfg
        rows = [r for r in self.logs if r["t"] >= cfg.warmup_s]
        result = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]
                  if key not in ("t", "energy_wh", "action", "severity")}
        eligible = self.target_eligible
        deadline = self.target_crossing - cfg.required_lead_s
        result.update(early_report_rate=float(np.mean(self.first_report[eligible] <= deadline[eligible])) if np.any(eligible) else None,
                      early_local_rate=float(np.mean(self.first_local[eligible] <= deadline[eligible])) if np.any(eligible) else None,
                      eligible_targets=int(eligible.sum()),
                      early_reports=int(np.sum(self.first_report[eligible] <= deadline[eligible])),
                      energy_wh=float(self.fleet.energy.sum()), distance_km=float(self.fleet.distance.sum() / 1000),
                      min_separation_m=float(self.fleet.min_separation), collision_steps=self.fleet.collision_steps,
                      switches=self.policy.switches, calibration_fraction=float(self.policy.action_counts[7] / max(1, self.policy.decisions)),
                      observations=int(sum(b.observations for b in self.beliefs)),
                      scenario=self.scenario, seed=self.seed, policy=self.policy_name,
                      config=cfg.dict(), source_sha256=source_hash(), exogenous_sha256=self.exogenous_sha256,
                      run_id=run_id(cfg, self.scenario, self.seed, self.policy_name),
                      runtime_s=time.perf_counter() - self.start_clock,
                      action_counts=self.policy.action_counts.tolist())
        return result

    def save_trace(self, path):
        if not self.trace:
            return
        np.savez_compressed(path, **{k: np.array([r[k] for r in self.trace]) for k in self.trace[0]},
                            target_crossing=self.target_crossing, first_report=self.first_report,
                            first_local=self.first_local, target_eligible=self.target_eligible)
