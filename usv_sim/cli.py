import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import scipy
from .config import Config, SCENARIOS, POLICIES, run_id, source_hash
from .world import World


def episode_job(job):
    data, scenario, seed, policy, trace_path, checkpoint = job
    cfg = Config(**data)
    learned = None
    checkpoint_hash = ""
    world_policy = policy
    if checkpoint:
        from .learning import DoubleQ
        agent = DoubleQ.load(checkpoint)
        learned = agent.action
        world_policy = "active"
        checkpoint_hash = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    world = World(cfg, scenario, seed, world_policy, bool(trace_path))
    result = world.run(learned)
    result["policy"] = policy
    result["run_id"] = run_id(cfg, scenario, seed, policy, checkpoint_hash)
    result["checkpoint_sha256"] = checkpoint_hash
    if trace_path:
        world.save_trace(trace_path)
    return result


def main():
    parser = argparse.ArgumentParser(description="USV swarm research: simulate, benchmark or train")
    sub = parser.add_subparsers(dest="command", required=True)
    for cmd in ("run", "benchmark", "train"):
        p = sub.add_parser(cmd)
        p.add_argument("--config")
        p.add_argument("--duration", type=float)
        p.add_argument("--n", type=int)
        p.add_argument("--out", required=True)
        if cmd == "train":
            p.add_argument("--episodes", type=int, default=200)
            p.add_argument("--training-seed", type=int, default=0)
            p.add_argument("--resume", action="store_true")
            continue
        p.add_argument("--policies", nargs="+", default=["bio"] if cmd == "run" else ["fixed_ring", "fixed_robust", "adaptive", "active", "hysteresis", "bio"])
        p.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=["storm"] if cmd == "run" else list(SCENARIOS))
        p.add_argument("--seeds", type=int, default=1 if cmd == "run" else 20)
        p.add_argument("--seed-start", type=int, default=1000)
        p.add_argument("--workers", type=int, default=1)
        p.add_argument("--trace", action="store_true")
        p.add_argument("--checkpoint", help="Evaluate a native DoubleQ checkpoint; use --policies double_q")
    args = parser.parse_args()
    cfg = Config.read(args.config, duration_s=args.duration, n=args.n)
    if args.command == "train":
        from .learning import train
        train(cfg, args.out, args.episodes, args.training_seed, args.resume)
        return
    for p in args.policies:
        if p not in POLICIES and not (p == "double_q" and args.checkpoint):
            parser.error(f"Unknown policy {p}; use {POLICIES} or double_q with --checkpoint")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config_blob = dict(config=cfg.dict(), source_sha256=source_hash(), python=sys.version,
                       platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__,
                       arguments=vars(args))
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["config"] != cfg.dict() or previous["source_sha256"] != source_hash():
            raise ValueError("Output directory has different source/config; choose a new --out")
    else:
        manifest_path.write_text(json.dumps(config_blob, indent=2))
    checkpoint_hash = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest() if args.checkpoint else ""
    jobs = []
    skipped = 0
    for scenario in args.scenarios:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            for policy in args.policies:
                checkpoint = args.checkpoint if policy == "double_q" else None
                ident = run_id(cfg, scenario, seed, policy, checkpoint_hash if checkpoint else "")
                dest = out / (ident + ".json")
                if dest.exists():
                    skipped += 1
                    continue
                trace_path = str(out / (ident + ".npz")) if args.trace and seed == args.seed_start else None
                jobs.append((cfg.dict(), scenario, seed, policy, trace_path, checkpoint))
    print(f"{len(jobs)} episodes to run; {skipped} already complete.", flush=True)
    def save(result, count):
        dest = out / (result["run_id"] + ".json")
        temporary = dest.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
        temporary.replace(dest)
        if count % 10 == 0 or count == len(jobs):
            print(f"Completed {count}/{len(jobs)}: {result['scenario']} {result['policy']} seed {result['seed']} delivered={result['delivered']:.3f}", flush=True)
    if args.workers == 1:
        for count, job in enumerate(jobs, 1):
            save(episode_job(job), count)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(episode_job, j) for j in jobs]
            for count, future in enumerate(as_completed(futures), 1):
                save(future.result(), count)
    records = [json.loads(p.read_text()) for p in out.glob("*.json") if p.name != "manifest.json"]
    if records:
        columns = [k for k in records[0] if k not in ("config", "action_counts")]
        with (out / "episodes.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, columns, extrasaction="ignore")
            writer.writeheader(); writer.writerows(records)
    print(f"Saved {len(records)} episodes to {out}", flush=True)


if __name__ == "__main__":
    main()
