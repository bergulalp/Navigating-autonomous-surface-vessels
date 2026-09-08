"""Optional deep RL workflow; keep checkpoint evaluation separate from training."""
import argparse
import json
from pathlib import Path
from .config import Config, source_hash


def make_env(cfg, seed, stack=4):
    def build():
        from gymnasium.wrappers import FrameStackObservation
        from stable_baselines3.common.monitor import Monitor
        from .gym_env import USVEnv
        env = USVEnv(cfg)
        env.reset(seed=seed)
        if stack > 1:
            env = FrameStackObservation(env, stack_size=stack)
        return Monitor(env)
    return build


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--out", default="results/ppo_seed0")
    p.add_argument("--steps", type=int, default=300000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--stack", type=int, default=4)
    p.add_argument("--resume")
    args = p.parse_args()
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        from stable_baselines3.common.callbacks import CheckpointCallback
        from .gym_env import USVEnv
    except ImportError as e:
        raise SystemExit("Install optional dependencies: python -m pip install -r requirements-rl.txt") from e
    cfg = Config.read(args.config)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    check_env(USVEnv(cfg), warn=True)
    constructors = [make_env(cfg, 500000 + args.seed * 1000 + i, args.stack) for i in range(args.workers)]
    vec = SubprocVecEnv(constructors) if args.workers > 1 else DummyVecEnv(constructors)
    model = (PPO.load(args.resume, env=vec, device="cpu") if args.resume else
             PPO("MlpPolicy", vec, seed=args.seed, n_steps=256, batch_size=128, gamma=.97,
                 learning_rate=3e-4, policy_kwargs={"net_arch": [64, 64]}, verbose=1, device="cpu"))
    (out / "manifest.json").write_text(json.dumps(dict(config=cfg.dict(), arguments=vars(args),
                                                     source_sha256=source_hash()), indent=2))
    callback = CheckpointCallback(save_freq=max(1, 10000 // args.workers), save_path=str(out), name_prefix="ppo")
    model.learn(args.steps, callback=callback, reset_num_timesteps=not bool(args.resume))
    model.save(out / "final")
    vec.close()


if __name__ == "__main__":
    main()
