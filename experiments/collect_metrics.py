#!/usr/bin/env python
"""
Collect the evaluation metrics of a purification sweep into one tidy CSV.

Each run writes <logdir>/config.json (the full argument namespace) and
<logdir>/eval/metrics.json (per-episode lists plus their `_mean` summaries). This
script pairs them up and emits one row per run.

    python experiments/collect_metrics.py results/purification -o summary.csv

Add --per-episode to additionally emit one row per evaluation episode, which is what
the plotting script uses for confidence intervals.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

# Metrics kept in the summary. Anything not listed is still written when --all is used.
METRICS = [
    # resource efficiency - the quantity the experiment is about
    "resources_per_success_mean",
    "resources_for_and_per_success_mean",
    "resources_per_packet_mean",
    "total_links_per_success_mean",
    "purification_links_agent_mean",
    "purification_links_auto_mean",
    "purification_links_auto_per_success_mean",
    "purification_links_agent_per_success_mean",
    "purification_rounds_requested_mean",
    # service quality - resource savings only count if these hold up
    "episode_success_rate_mean",
    "total_episode_fidelity_mean",
    "average_episode_packets_mean",
    "average_reward_mean",
    "average_episode_path_length_0_mean",
]

# Run settings that identify an arm.
CONFIG_KEYS = [
    "method",
    "seed",
    "max_purification_rounds",
    "purification_cost_weight",
    "auto_distillation_threshold",
    "keep_auto_distillation",
    "policy",
    "n_router",
    "neighbors",
    "n_quantum",
    "ttl",
    "total_steps",
    "episode_steps",
    "netmon",
    "action_mask",
    "enable_action_mask",
]


def arm_label(config):
    """Short human-readable name for the method of a run."""
    method = config.get("method", "reliq") or "reliq"  # runs predating --method are RELiQ
    rounds = int(config.get("max_purification_rounds", 0) or 0)
    auto = float(config.get("auto_distillation_threshold", 0) or 0)
    cost = float(config.get("purification_cost_weight", 0) or 0)
    policy = config.get("policy", "?")

    if policy in ("qpath", "qleap"):
        return policy
    if method == "rt-reliq":
        suffix = " +auto" if config.get("keep_auto_distillation") else ""
        return f"RT-RELiQ P={rounds} c={cost:g}{suffix}"
    return "RELiQ" if auto > 0 else "RELiQ (no purif.)"


def find_runs(root: Path):
    """Yield (config_path, metrics_path) for every finished run under root."""
    for metrics_path in sorted(root.rglob("eval/metrics.json")):
        logdir = metrics_path.parent.parent
        config_path = logdir / "config.json"
        if not config_path.exists():
            # some layouts keep config.json next to metrics.json
            alt = metrics_path.parent / "config.json"
            config_path = alt if alt.exists() else None
        yield config_path, metrics_path


def load_json(path):
    with open(path) as f:
        data = json.load(f)
    # main.py writes the args dict as a JSON string in some code paths
    if isinstance(data, str):
        data = json.loads(data)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="directory containing the runs")
    parser.add_argument("-o", "--output", type=Path, default=Path("summary.csv"))
    parser.add_argument(
        "--per-episode",
        action="store_true",
        help="also write <output stem>_episodes.csv with one row per eval episode",
    )
    parser.add_argument(
        "--all", action="store_true", help="keep every scalar metric, not just METRICS"
    )
    args = parser.parse_args()

    rows = []
    episode_rows = []

    for config_path, metrics_path in find_runs(args.root):
        metrics = load_json(metrics_path)
        config = load_json(config_path) if config_path else {}

        row = {
            "run": str(metrics_path.parent.parent.relative_to(args.root)),
            "arm": arm_label(config),
        }
        for key in CONFIG_KEYS:
            if key in config:
                row[key] = config[key]

        if args.all:
            row.update({k: v for k, v in metrics.items() if not isinstance(v, list)})
        else:
            for key in METRICS:
                if key in metrics:
                    row[key] = metrics[key]
        rows.append(row)

        if args.per_episode:
            # per-episode lists are stored under the un-suffixed key
            lists = {
                k[: -len("_mean")]: metrics[k[: -len("_mean")]]
                for k in METRICS
                if k.endswith("_mean")
                and k[: -len("_mean")] in metrics
                and isinstance(metrics[k[: -len("_mean")]], list)
            }
            if lists:
                n = max(len(v) for v in lists.values())
                for i in range(n):
                    episode_row = dict(row)
                    episode_row["episode"] = i
                    for key, values in lists.items():
                        if i < len(values):
                            episode_row[key] = values[i]
                    episode_rows.append(episode_row)

    if not rows:
        raise SystemExit(f"no finished runs found under {args.root}")

    df = pd.DataFrame(rows).sort_values(["arm", "seed"], na_position="last")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"wrote {args.output}  ({len(df)} runs, {len(df.columns)} columns)")
    print(df.get("arm", pd.Series(dtype=str)).value_counts().to_string())

    if args.per_episode and episode_rows:
        ep_path = args.output.with_name(args.output.stem + "_episodes.csv")
        pd.DataFrame(episode_rows).to_csv(ep_path, index=False)
        print(f"wrote {ep_path}  ({len(episode_rows)} episodes)")


if __name__ == "__main__":
    main()
