"""Episode-level paired inference; never use scans as independent replicates."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = ("delivered", "early_report_rate", "energy_wh", "range_rmse", "switches", "calibration_fraction")


def ci(values):
    x = np.array(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float(x.mean()) if len(x) else np.nan, np.nan, np.nan, len(x))
    mean = x.mean()
    half = student_t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return mean, mean - half, mean + half, len(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--reference", default="fixed_ring")
    args = p.parse_args()
    records = []
    for folder in args.inputs:
        for path in Path(folder).glob("*.json"):
            d = json.loads(path.read_text())
            if "run_id" in d:
                records.append(d)
    if not records:
        raise SystemExit("No episode JSON files found")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    if df.run_id.duplicated().any():
        raise ValueError("Duplicate run IDs in inputs")
    # Cross-policy pairing requires identical configuration and exogenous tape.
    df["config_key"] = df.config.map(lambda x: json.dumps(x, sort_keys=True))
    if df.config_key.nunique() > 1 or df.source_sha256.nunique() > 1:
        raise ValueError("Mixed source/configuration; analyze each experiment separately")
    pairing = ["scenario", "seed", "config_key"]
    if (df.groupby(pairing).exogenous_sha256.nunique() > 1).any():
        raise ValueError("Mismatched exogenous tapes")
    if df.duplicated(["scenario", "seed", "policy"]).any():
        raise ValueError("Multiple training checkpoints with same policy label; compare separately")
    summaries, paired = [], []
    for (scenario, policy), group in df.groupby(["scenario", "policy"]):
        for metric in METRICS:
            mean, low, high, n = ci(group[metric])
            summaries.append(dict(scenario=scenario, policy=policy, metric=metric, mean=mean, low=low, high=high, n=n))
            reference = df[(df.scenario == scenario) & (df.policy == args.reference)][["seed", metric]]
            match = group[["seed", metric]].merge(reference, on="seed", suffixes=("", "_reference"), validate="one_to_one")
            if len(match):
                m, lo, hi, count = ci(match[metric] - match[metric + "_reference"])
                paired.append(dict(scenario=scenario, policy=policy, reference=args.reference, metric=metric,
                                   difference=m, low=lo, high=hi, n=count))
    summary, pairs = pd.DataFrame(summaries), pd.DataFrame(paired)
    summary.to_csv(out / "summary.csv", index=False)
    pairs.to_csv(out / "paired_differences.csv", index=False)
    df.drop(columns=["config", "config_key", "action_counts"]).to_csv(out / "episodes.csv", index=False)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    selected = [p for p in ("fixed_ring", "fixed_robust", "adaptive", "active", "hysteresis", "bio", "bio_no_habituation", "double_q") if p in df.policy.unique()]
    scenarios = sorted(df.scenario.unique())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    for ax, scenario in zip(axes.flat, scenarios):
        g = summary[(summary.scenario == scenario) & (summary.metric == "delivered")].set_index("policy").reindex(selected)
        y = g["mean"].to_numpy()
        ax.errorbar(np.arange(len(selected)), y, yerr=np.array([y - g.low, g.high - y]), fmt="o", capsize=3, color="#145b70")
        ax.set_xticks(np.arange(len(selected)), [s.replace("_", "\n") for s in selected], fontsize=8)
        ax.set_title(scenario.replace("_", " "))
        ax.set_ylabel("Delivered fresh coverage")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=.2)
    for ax in axes.flat[len(scenarios):]:
        ax.set_visible(False)
    fig.suptitle("Held-out episodes: mean and 95% t intervals (synthetic model)", fontsize=15)
    fig.tight_layout()
    fig.savefig(out / "benchmark.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for policy in selected:
        g = df[df.policy == policy]
        ax.scatter(g.energy_wh.mean(), g.delivered.mean(), s=70, label=policy.replace("_", " "))
    ax.set_xlabel("Fleet energy over the full mission [Wh; assumed power model]")
    ax.set_ylabel("Delivered fresh coverage after warm-up")
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "energy_coverage.png", dpi=180); plt.close(fig)
    print(summary[summary.metric == "delivered"].to_string(index=False))


if __name__ == "__main__":
    main()
