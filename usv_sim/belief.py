"""Two-parameter range/availability filter with time-based process uncertainty."""
import numpy as np
from scipy.special import expit, logit
from .physics import nominal_sensor


class SensorBelief:
    def __init__(self):
        self.rs, self.off = np.meshgrid(np.linspace(0.25, 1.65, 21),
                                       np.linspace(-2.8, 1.2, 21), indexing="ij")
        prior = np.exp(-0.5 * ((self.rs - 1) / 0.23) ** 2 - 0.5 * (self.off / 0.65) ** 2)
        self.posterior = prior / prior.sum()
        self.innovation = 0.0
        self.observations = 0

    def predict(self, dt, cfg=None):
        # Parameter diffusion continues even when no reference is observed.
        sr = cfg.range_process_std if cfg else 0.004
        sa = cfg.availability_process_std if cfg else 0.008
        weights = np.array([sr ** 2 / 0.07 ** 2, sa ** 2 / 0.20 ** 2]) * dt / 2
        substeps = max(1, int(np.ceil(weights.max() / 0.45)))
        p = self.posterior.copy()
        # Conservative nearest-neighbour diffusion works even below one grid cell.
        for _ in range(substeps):
            for axis, weight in enumerate(weights / substeps):
                flux = np.diff(p, axis=axis) * weight
                lower, upper = [slice(None)] * 2, [slice(None)] * 2
                lower[axis], upper[axis] = slice(None, -1), slice(1, None)
                p[tuple(lower)] += flux
                p[tuple(upper)] -= flux
        hazard = 1 - np.exp(-dt / 14400)
        self.posterior = (1 - hazard) * p / p.sum() + hazard / p.size
        self.innovation *= np.exp(-dt / 120)

    def update(self, distances, angles, detections, severity, wind, cfg):
        if len(distances) == 0:
            return
        radar, _, a0 = nominal_sensor(severity, cfg)
        ll = np.zeros_like(self.posterior)
        residual = []
        for d, angle, y in zip(distances, angles, detections):
            r = radar * self.rs * np.exp(-0.20 * severity * np.cos(2 * (angle - wind)))
            prob = np.clip(expit(logit(a0) + self.off) * 0.94 * expit((r - d) / (0.2 * r)), 1e-7, 1 - 1e-7)
            pred = float(np.sum(self.posterior * prob))
            residual.append(abs(float(y) - pred) / np.sqrt(pred * (1 - pred) + 0.04))
            ll += np.log(prob if y else 1 - prob)
        # Shared visibility and persistence make observations dependent. Tempering
        # limits false confidence; it is an approximation, not an exact HMM filter.
        temper = min(1.0, cfg.dt_s / cfg.wave_memory_s) / max(1, len(distances) / 2)
        logp = np.log(self.posterior + 1e-300) + temper * ll
        logp -= logp.max()
        self.posterior = np.exp(logp)
        self.posterior /= self.posterior.sum()
        self.innovation = 0.8 * self.innovation + 0.2 * float(np.mean(residual))
        self.observations += len(distances)

    def moments(self):
        p = self.posterior
        mr, ma = np.sum(p * self.rs), np.sum(p * self.off)
        vr, va = np.sum(p * (self.rs - mr) ** 2), np.sum(p * (self.off - ma) ** 2)
        cov = np.sum(p * (self.rs - mr) * (self.off - ma))
        return np.array([mr, np.sqrt(vr), ma, np.sqrt(va), cov, self.innovation, self.observations])

    def interval(self, array, tail=0.05):
        ids = np.argsort(array.ravel())
        vals, probs = array.ravel()[ids], self.posterior.ravel()[ids]
        cdf = np.cumsum(probs)
        return np.interp([tail, 1 - tail], cdf, vals)
