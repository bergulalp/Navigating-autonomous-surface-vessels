"""Restartable paired experiments; extension and physical hashes are both saved."""
import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from usv_sim.config import Config, SCENARIOS, POLICIES, source_hash
from usv_sim.world import World
from usv_sim.planning import mission_weights
from .control import ControlConfig, LocalPolicy

LOCAL = ("static_tuned", "economic", "economic_no_cost", "fungal", "deadline_fusion", "fungal_no_fusion",
         "fungal_no_stop", "recruitment", "greedy_sparse", "recruitment_no_social")


def extension_hash():
    h = hashlib.sha256()
    for name in ("control.py", "experiment.py"):
        h.update(name.encode()); h.update(Path(__file__).with_name(name).read_bytes())
    return h.hexdigest()


def identity(cfg, settings, scenario, seed, policy):
    return hashlib.sha256(json.dumps(dict(config=cfg, control=settings, scenario=scenario, seed=seed,
                                         policy=policy, physical=source_hash(), extension=extension_hash()),
                                     sort_keys=True).encode()).hexdigest()[:24]


class ResearchWorld(World):
    def step(self):
        row = super().step()
        age = np.minimum(self.t - self.probe_times[-1], 10 * self.cfg.report_ttl_s)
        w = mission_weights(self.angles, self.t, self.cfg.duration_s, self.scenario, self.cfg.priority_turns)
        row["capped_aoi_s"] = float(w @ age)
        # Clipped AoI explicitly includes unobserved cells at the cap.
        row["aoi_cap_fraction"] = float(w @ (age >= 10 * self.cfg.report_ttl_s))
        return row


def job_run(job):
    cfg_dict, settings_dict, scenario, seed, policy, trace = job
    cfg, settings = Config(**cfg_dict).validate(), ControlConfig(**settings_dict)
    world = ResearchWorld(cfg, scenario, seed, policy, bool(trace))
    if policy in LOCAL:
        world.policy = LocalPolicy(policy, cfg, settings)
    result = world.run()
    result.update(extension_sha256=extension_hash(), control=settings.dict(),
                  run_id=identity(cfg_dict, settings_dict, scenario, seed, policy))
    data = np.array([r["delivered"] for r in world.logs if r["t"] >= cfg.warmup_s])
    result["delivered_p10"] = float(np.quantile(data, .10))
    result["delivered_below_half_fraction"] = float(np.mean(data < .5))
    if hasattr(world.policy, "diagnostics"):
        diag = world.policy.diagnostics
        result["accepted_moves"] = sum(d["accepted"] for d in diag)
        result["mean_recruited_vessels"] = float(np.mean([d.get("moving_vessels", 0) for d in diag])) if diag else 0.
        if trace:
            Path(trace).with_suffix(".decisions.json").write_text(json.dumps(diag, indent=2))
    if trace:
        world.save_trace(trace)
    return result


def main():
    p = argparse.ArgumentParser(description="USV v3 paired, restartable CPU experiments")
    p.add_argument("--config", default="configs/benchmark.json")
    p.add_argument("--control")
    p.add_argument("--out", required=True)
    p.add_argument("--policies", nargs="+", choices=LOCAL + POLICIES,
                   default=["fixed_ring", "static_tuned", "economic", "fungal", "greedy_sparse", "recruitment"])
    p.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--seed-start", type=int, default=7000)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--trace", action="store_true")
    p.add_argument("--duration", type=float)
    p.add_argument("--energy-price", type=float)
    p.add_argument("--step", type=float)
    p.add_argument("--static-radius", type=float)
    args = p.parse_args()
    if args.seeds < 1 or args.workers < 1:
        p.error("Use positive seeds/workers")
    cfg = Config.read(args.config, duration_s=args.duration)
    cc = json.loads(Path(args.control).read_text()) if args.control else {}
    cc.update({k: v for k, v in dict(energy_price=args.energy_price, step_m=args.step,
                                   static_radius=args.static_radius).items() if v is not None})
    settings = ControlConfig(**cc)
    if not 0 <= settings.energy_price or settings.step_m <= 0 or not .5 <= settings.static_radius <= 1.4:
        p.error("Invalid control parameters")
    dest = Path(args.out); dest.mkdir(parents=True, exist_ok=True)
    provenance = dict(config=cfg.dict(), control=settings.dict(), physical_sha256=source_hash(),
                      extension_sha256=extension_hash())
    manifest = dest / "manifest.json"
    if manifest.exists():
        prev = json.loads(manifest.read_text())
        if any(prev[k] != v for k, v in provenance.items()):
            raise ValueError("Existing output has different code/configuration; use a new --out")
    else:
        manifest.write_text(json.dumps(provenance | dict(python=sys.version, platform=platform.platform(),
                                                        arguments=vars(args)), indent=2))
    jobs = []
    for scenario in args.scenarios:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            for policy in args.policies:
                key = identity(cfg.dict(), settings.dict(), scenario, seed, policy)
                if (dest / (key + ".json")).exists():
                    continue
                trace = str(dest / (key + ".npz")) if args.trace and seed == args.seed_start else None
                jobs.append((cfg.dict(), settings.dict(), scenario, seed, policy, trace))
    started = time.perf_counter()
    print(f"{len(jobs)} pending episodes, workers={args.workers}; simulator hours per episode={cfg.duration_s/3600:g}", flush=True)
    def save(row, count):
        path = dest / (row["run_id"] + ".json")
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(row, indent=2, allow_nan=False)); tmp.replace(path)
        if count % 10 == 0 or count == len(jobs):
            print(f"{count}/{len(jobs)} in {time.perf_counter()-started:.1f}s; {row['scenario']} {row['policy']}: C={row['delivered']:.4f}, E={row['energy_wh']:.1f} Wh", flush=True)
    if args.workers == 1:
        for i, job in enumerate(jobs, 1): save(job_run(job), i)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            fs = [pool.submit(job_run, j) for j in jobs]
            for i, f in enumerate(as_completed(fs), 1): save(f.result(), i)
    elapsed = time.perf_counter() - started
    timing = dict(completed=len(jobs), wall_s=elapsed, workers=args.workers,
                  wall_s_per_episode=elapsed / len(jobs) if jobs else None,
                  cpu_count=os.cpu_count(), config=cfg.dict())
    if jobs:
        # Multiple resume invocations retain their actual throughput measurements.
        with (dest / "timing.jsonl").open("a") as f: f.write(json.dumps(timing) + "\n")
    records = [json.loads(f.read_text()) for f in dest.glob("*.json")
               if f.name != "manifest.json" and not f.name.endswith(".decisions.json")]
    if records:
        keys = sorted({k for r in records for k, v in r.items() if not isinstance(v, (list, dict))})
        with (dest / "episodes.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, keys, extrasaction="ignore"); w.writeheader(); w.writerows(records)
    print(json.dumps(timing), flush=True)


if __name__ == "__main__":
    main()
