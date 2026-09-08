from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path

SCENARIOS = ("calm", "storm", "intermittent", "sensor_fault", "moving_front", "blackout")
POLICIES = ("fixed_ring", "fixed_robust", "model", "adaptive", "active", "periodic", "hysteresis",
            "bio", "bio_no_habituation", "truth_informed", "random")


@dataclass(frozen=True)
class Config:
    n: int = 6
    duration_s: float = 3600.0
    dt_s: float = 5.0
    motion_dt_s: float = 1.0
    decision_s: float = 60.0
    warmup_s: float = 300.0
    report_ttl_s: float = 30.0
    telemetry_ttl_s: float = 40.0
    command_timeout_s: float = 180.0
    mission_radius_m: float = 2500.0
    crossing_radius_m: float = 1800.0
    required_lead_s: float = 60.0
    grid_angles: int = 64
    target_count: int = 36
    priority_turns: float = 1.0
    max_speed_mps: float = 3.0
    battery_wh: float = 1800.0
    radio_r50_calm_m: float = 4600.0
    radar_r50_calm_m: float = 1650.0
    camera_r50_calm_m: float = 650.0
    radio_weather_slope: float = 1.05
    radar_weather_slope: float = 1.4
    wave_correlation_m: float = 350.0
    wave_memory_s: float = 15.0
    range_process_std: float = 0.004
    availability_process_std: float = 0.008
    sensor_model: str = "logistic"
    full_blackout: bool = False
    information_weight: float = 0.035
    movement_weight: float = 0.025
    switch_threshold: float = 0.006
    minimum_dwell_s: float = 120.0
    evidence_memory_s: float = 90.0
    habituation_memory_s: float = 480.0
    habituation_gain: float = 2.0
    planner_samples: int = 32
    temporal_planner: bool = True
    transit_lookahead_s: float = 300.0

    def validate(self):
        if self.n < 3 or self.grid_angles < 8 or self.target_count < 1:
            raise ValueError("Use n >= 3, grid_angles >= 8 and target_count >= 1")
        if not 0 < self.dt_s <= self.decision_s or self.duration_s <= self.warmup_s:
            raise ValueError("Invalid time settings")
        if not 0 < self.motion_dt_s <= self.dt_s:
            raise ValueError("Require 0 < motion_dt_s <= dt_s")
        for t in (self.decision_s, self.duration_s):
            if abs(t / self.dt_s - round(t / self.dt_s)) > 1e-8:
                raise ValueError("Duration and decision interval must be multiples of dt_s")
        if self.sensor_model not in ("logistic", "probit"):
            raise ValueError("sensor_model must be logistic or probit")
        if self.planner_samples < 2 or self.report_ttl_s < self.dt_s:
            raise ValueError("Use at least two planner samples and TTL >= dt_s")
        return self

    @classmethod
    def read(cls, path=None, **overrides):
        data = json.loads(Path(path).read_text()) if path else {}
        return cls(**(data | {k: v for k, v in overrides.items() if v is not None})).validate()

    def dict(self):
        return asdict(self)


def source_hash():
    root = Path(__file__).parent
    h = hashlib.sha256()
    for p in sorted(root.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def run_id(config, scenario, seed, policy, checkpoint_hash=""):
    blob = dict(config=config.dict(), scenario=scenario, seed=int(seed), policy=policy,
                source_sha256=source_hash(), checkpoint_sha256=checkpoint_hash)
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:20]
