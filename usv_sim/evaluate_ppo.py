import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
from .config import Config, SCENARIOS, source_hash


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--config")
    p.add_argument("--out", required=True)
    p.add_argument("--stack", type=int, default=4)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--seed-start", type=int, default=3000)
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    args = p.parse_args()
    from stable_baselines3 import PPO
    from gymnasium.wrappers import FrameStackObservation
    from .gym_env import USVEnv
    cfg = Config.read(args.config)
    model = PPO.load(args.model, device="cpu")
    model_path = Path(args.model)
    if not model_path.exists():
        model_path = model_path.with_suffix(".zip")
    checkpoint_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for scenario in args.scenarios:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            env = USVEnv(cfg, domain_randomization=False)
            wrapped = FrameStackObservation(env, stack_size=args.stack) if args.stack > 1 else env
            obs, _ = wrapped.reset(seed=seed, options={"scenario": scenario, "world_seed": seed})
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = wrapped.step(int(action))
                done = terminated or truncated
            result = env.world.result()
            result.update(policy="ppo", checkpoint_sha256=checkpoint_hash, source_sha256=source_hash())
            result["run_id"] = hashlib.sha256((result["run_id"] + checkpoint_hash).encode()).hexdigest()[:20]
            (out / (result["run_id"] + ".json")).write_text(json.dumps(result, indent=2, allow_nan=False))
            print(scenario, seed, round(result["delivered"], 4), flush=True)
            wrapped.close()


if __name__ == "__main__":
    main()
