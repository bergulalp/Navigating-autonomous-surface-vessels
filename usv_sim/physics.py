"""Phenomenological sensor, channel and planar vessel models; not fitted hardware."""
import numpy as np
from scipy.special import expit, ndtr, logit


def severity_at(t, duration, scenario):
    x = t / duration
    if scenario == "calm":
        return 0.12
    if scenario == "storm":
        return 0.18 + 0.70 * np.clip((x - 0.22) / 0.22, 0, 1)
    if scenario == "intermittent":
        return 0.38 + 0.48 * float((t % 420) < 100 and x > 0.2)
    if scenario == "sensor_fault":
        return 0.48
    if scenario == "moving_front":
        return 0.48 + 0.22 * np.sin(2 * np.pi * x)
    if scenario == "blackout":
        return 0.48
    raise ValueError(scenario)


def nominal_sensor(severity, cfg):
    radar = cfg.radar_r50_calm_m * np.exp(-cfg.radar_weather_slope * severity)
    camera = cfg.camera_r50_calm_m * np.exp(-0.8 * severity)
    availability = 0.94 / (1 + 1.1 * severity ** 2)
    return radar, camera, availability


def footprints(positions, points, severity, range_scale, availability_offset, cfg,
               wind=0.0, camera_scale=None, model=None):
    """Return conditional fused Pd, radar Pd and shared visibility probability.

    Visibility is ONE common latent variable per observer/scan for both sensors.
    Conditional radar and camera detector noises are independent. The same
    equations feed the actual sampler and the planner (which has estimated inputs).
    """
    delta = points[None, :, :] - positions[:, None, :]
    distance = np.linalg.norm(delta, axis=-1)
    angle = np.arctan2(delta[..., 1], delta[..., 0])
    rad, cam, a0 = nominal_sensor(severity, cfg)
    anisotropy = np.exp(-0.20 * severity * np.cos(2 * (angle - wind)))
    radar_range = rad * np.asarray(range_scale)[:, None] * anisotropy
    camera_range = cam * (np.ones(len(positions)) if camera_scale is None else camera_scale)[:, None]
    zrad = (radar_range - distance) / (0.20 * radar_range)
    zcam = (camera_range - distance) / (0.22 * camera_range)
    curve = ndtr if (model or cfg.sensor_model) == "probit" else expit
    # Probit uses an approximately slope-matched scaling, but different tails.
    factor = 0.6267 if curve is ndtr else 1.0
    pr = 0.94 * curve(factor * zrad)
    pc = 0.86 * curve(factor * zcam)
    fused = 1 - (1 - pr) * (1 - pc)
    a = expit(logit(a0) + np.asarray(availability_offset))
    return fused, pr, a


def radio_prob(positions, severity, cfg, scale=1.0, fade=1.0):
    """Reciprocal packet probability, including the base station at index n."""
    p = np.vstack((positions, np.zeros((1, 2))))
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    r50 = cfg.radio_r50_calm_m * np.exp(-cfg.radio_weather_slope * severity) * scale
    q = 0.995 * fade / (1 + (d / r50) ** 6)
    np.fill_diagonal(q, 1)
    return q


def reachable(adjacency, max_hops):
    """Reachability to final node, supporting arbitrary leading batch dimensions."""
    reach = np.zeros(adjacency.shape[:-1], dtype=bool)
    reach[..., -1] = True
    for _ in range(max_hops):
        reach = np.any(adjacency & reach[..., None, :], axis=-1) | reach
    return reach


def forward_latest(timestamps, payload, adjacency):
    """One synchronous hop. Receiver gets the newest reachable cached record.

    timestamps: [node, record]; payload: [node, record, feature] or None.
    A record cannot travel two hops in this call. Diagonal edges keep local copies.
    """
    offered = np.where(adjacency[:, :, None], timestamps[None, :, :], -np.inf)
    sender = np.argmax(offered, axis=1)
    latest = np.max(offered, axis=1)
    if payload is None:
        return latest, None
    data = payload[sender, np.arange(timestamps.shape[1])[None, :]].copy()
    return latest, data


class Fleet:
    def __init__(self, positions, cfg):
        self.p = positions.copy()
        self.psi = np.arctan2(positions[:, 1], positions[:, 0]) + np.pi / 2
        self.u = np.zeros(cfg.n)
        self.v = np.zeros(cfg.n)
        self.r = np.zeros(cfg.n)
        self.energy = np.zeros(cfg.n)
        self.distance = np.zeros(cfg.n)
        self.cfg = cfg
        self.min_separation = np.inf
        self.collision_steps = 0

    def step(self, goals, current, wave_force, severity):
        substeps = int(np.ceil(self.cfg.dt_s / self.cfg.motion_dt_s))
        for _ in range(substeps):
            self._substep(goals, current, wave_force, severity, self.cfg.dt_s / substeps)

    def _substep(self, goals, current, wave_force, severity, dt):
        cfg = self.cfg
        delta = goals - self.p
        # Simple onboard velocity guidance with current compensation.
        wanted = 0.025 * delta - current
        dp = self.p[:, None, :] - self.p[None, :, :]
        dist = np.linalg.norm(dp, axis=-1)
        np.fill_diagonal(dist, np.inf)
        self.min_separation = min(self.min_separation, float(np.min(dist)))
        self.collision_steps += int(np.any(dist < 30))
        # Local near-field avoidance. This is not a certified safety filter.
        repulsion = np.clip((180 - dist) / 60, 0, 2.0)
        wanted += np.sum(dp / np.maximum(dist[..., None], 1) * repulsion[..., None], axis=1)
        psi_cmd = np.arctan2(wanted[:, 1], wanted[:, 0])
        err = np.arctan2(np.sin(psi_cmd - self.psi), np.cos(psi_cmd - self.psi))
        r_cmd = np.clip(err / 5, -0.25, 0.25)
        self.r += (r_cmd - self.r) * (1 - np.exp(-dt / 2.5))
        self.psi += self.r * dt
        u_cmd = np.minimum(np.linalg.norm(wanted, axis=1), cfg.max_speed_mps)
        u_cmd *= np.clip(np.cos(err), 0, 1)
        u_cmd[self.energy >= cfg.battery_wh] = 0
        self.u += np.clip((u_cmd - self.u) * (1 - np.exp(-dt / 4)), -0.25 * dt, 0.25 * dt)
        self.v += (0.16 * severity * wave_force - self.v) * (1 - np.exp(-dt / 4))
        velocity = np.stack((self.u * np.cos(self.psi) - self.v * np.sin(self.psi),
                             self.u * np.sin(self.psi) + self.v * np.cos(self.psi)), axis=1) + current
        move = velocity * dt
        self.p += move
        self.distance += np.linalg.norm(move, axis=1)
        power_w = 60 + 18 * np.abs(self.u) ** 3 + 8 * self.v ** 2 + 4 * self.r ** 2
        self.energy += power_w * dt / 3600
