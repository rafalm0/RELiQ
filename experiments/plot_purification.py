#!/usr/bin/env python
"""
Figures for the purification-control sweep.

Takes the CSVs written by collect_metrics.py and produces:

  fig1_resource_efficiency.png  links consumed per successful end-to-end entanglement,
                                split into routing links, agent purification and
                                automatic purification
  fig2_cost_quality.png         the trade-off plane: resource cost against delivered
                                fidelity and success rate, so a saving is only visible
                                as an improvement if quality is held
  fig3_cost_weight_sweep.png    effect of the reward penalty lambda on how much the
                                agent purifies and what it gets for it

Usage:
    python experiments/collect_metrics.py results/purification -o summary.csv --per-episode
    python experiments/plot_purification.py summary.csv -o figures/
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# arm ordering and colours, baselines first
ARM_ORDER = ["RELiQ", "RELiQ (no purif.)", "qpath", "qleap"]
BASELINE_COLORS = {
    "RELiQ": "#d95f02", "RELiQ (no purif.)": "#9e9e9e",
    "qpath": "#7570b3", "qleap": "#e7298a",
}
RL_CMAP = plt.get_cmap("viridis")


def order_arms(arms):
    """Baselines first, then RT-RELiQ arms sorted by rounds then cost weight."""
    baselines = [a for a in ARM_ORDER if a in arms]
    rl = sorted(
        [a for a in arms if a.startswith("RT-RELiQ")],
        key=lambda a: (
            float(a.split("P=")[1].split()[0]),
            float(a.split("c=")[1].split()[0]) if "c=" in a else 0.0,
        ),
    )
    rest = [a for a in arms if a not in baselines and a not in rl]
    return baselines + rl + sorted(rest)


def arm_colors(arms):
    colors = {}
    rl = [a for a in arms if a.startswith("RT-RELiQ")]
    for i, a in enumerate(rl):
        colors[a] = RL_CMAP(0.15 + 0.7 * (i / max(len(rl) - 1, 1)))
    for a in arms:
        colors.setdefault(a, BASELINE_COLORS.get(a, "#4575b4"))
    return colors


def agg(df, value, by="arm"):
    """Mean and standard error across seeds."""
    g = df.groupby(by)[value]
    out = pd.DataFrame({"mean": g.mean(), "sem": g.std(ddof=1) / np.sqrt(g.count())})
    return out.fillna(0.0)


def fig_resource_efficiency(df, out_dir):
    """Stacked links-per-success, separating routing from the two purification paths."""
    arms = order_arms(df["arm"].unique().tolist())

    # resources_per_success is the upstream metric (one link per hop); purification
    # links are reported separately per mechanism and stacked on top.
    d = df.copy()
    d["routing"] = d["resources_per_success_mean"]
    d["purif_agent"] = d.get("purification_links_agent_per_success_mean", 0.0)
    d["purif_auto"] = d.get("purification_links_auto_per_success_mean", 0.0)

    parts = [
        ("routing", "routing links", "#4575b4"),
        ("purif_agent", "agent purification (RT-RELiQ)", "#2ca25f"),
        ("purif_auto", "automatic purification (RELiQ)", "#d95f02"),
    ]

    x = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(max(7.0, 1.0 * len(arms)), 4.4))
    bottom = np.zeros(len(arms))
    for col, label, color in parts:
        stats = agg(d, col).reindex(arms).fillna(0.0)
        vals = stats["mean"].to_numpy()
        ax.bar(x, vals, 0.68, bottom=bottom, label=label, color=color,
               edgecolor="white", linewidth=0.6)
        bottom += vals

    total = agg(d, "total_links_per_success_mean").reindex(arms)
    if total["mean"].notna().any():
        ax.errorbar(x, total["mean"], yerr=total["sem"], fmt="none",
                    ecolor="black", capsize=3, linewidth=1.0)

    # "links per success" is undefined for an arm that never completed an
    # entanglement; mark those instead of silently drawing an empty slot.
    missing = d.groupby("arm")["resources_per_success_mean"].apply(
        lambda s: s.isna().all()
    ).reindex(arms, fill_value=True)
    for xi, arm in zip(x, arms):
        if bool(missing.loc[arm]):
            ax.text(xi, 0.02 * max(bottom.max(), 1.0), "no success",
                    ha="center", va="bottom", rotation=90, fontsize=8, color="grey")

    ax.set_xticks(x)
    ax.set_xticklabels(arms, rotation=30, ha="right")
    ax.set_ylabel("quantum links per successful entanglement")
    ax.set_title("Resource cost of one end-to-end entanglement")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = out_dir / "fig1_resource_efficiency.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig_cost_quality(df, out_dir):
    """Cost against quality: a saving only counts if fidelity and success hold."""
    arms = order_arms(df["arm"].unique().tolist())
    colors = arm_colors(arms)

    cost_col = (
        "total_links_per_success_mean"
        if "total_links_per_success_mean" in df
        else "resources_per_success_mean"
    )
    panels = [
        ("total_episode_fidelity_mean", "mean delivered fidelity"),
        ("episode_success_rate_mean", "success rate"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, (ycol, ylabel) in zip(axes, panels):
        if ycol not in df:
            ax.set_visible(False)
            continue
        xs = agg(df, cost_col).reindex(arms)
        ys = agg(df, ycol).reindex(arms)
        for arm in arms:
            ax.errorbar(
                xs.loc[arm, "mean"], ys.loc[arm, "mean"],
                xerr=xs.loc[arm, "sem"], yerr=ys.loc[arm, "sem"],
                fmt="o", markersize=7, color=colors[arm], capsize=2,
                label=arm, linewidth=1.0,
            )
        ax.set_xlabel("quantum links per successful entanglement")
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)

    # arrow showing the direction of improvement: cheaper and at least as good
    axes[0].annotate(
        "better", xy=(0.06, 0.92), xytext=(0.30, 0.92),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="grey"), color="grey", va="center",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(arms), 5),
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Resource cost against service quality")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path = out_dir / "fig2_cost_quality.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig_cost_weight_sweep(df, out_dir):
    """How the reward penalty changes purification intensity and what it buys."""
    rl = df[df["max_purification_rounds"].fillna(0) > 0].copy()
    if rl.empty or "purification_cost_weight" not in rl:
        return None

    metrics = [
        ("purification_links_agent_per_success_mean", "purification links per success"),
        ("total_links_per_success_mean", "total links per success"),
        ("total_episode_fidelity_mean", "mean delivered fidelity"),
    ]
    metrics = [(c, l) for c, l in metrics if c in rl]

    rounds = sorted(rl["max_purification_rounds"].unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.0))
    axes = np.atleast_1d(axes)

    for ax, (col, label) in zip(axes, metrics):
        for i, p in enumerate(rounds):
            sub = rl[rl["max_purification_rounds"] == p]
            stats = agg(sub, col, by="purification_cost_weight").sort_index()
            ax.errorbar(
                stats.index, stats["mean"], yerr=stats["sem"],
                marker="o", capsize=3, linewidth=1.6,
                color=RL_CMAP(0.15 + 0.7 * (i / max(len(rounds) - 1, 1))),
                label=f"P = {int(p)}",
            )
        ax.set_xlabel(r"purification cost weight $\lambda$")
        ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(frameon=False, title="max rounds")
    fig.suptitle("Effect of the purification reward penalty")
    fig.tight_layout()
    path = out_dir / "fig3_cost_weight_sweep.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="CSV from collect_metrics.py")
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    df = pd.read_csv(args.summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for col in ("max_purification_rounds", "purification_cost_weight"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for fn in (fig_resource_efficiency, fig_cost_quality, fig_cost_weight_sweep):
        path = fn(df, args.out_dir)
        print(f"wrote {path}" if path else f"skipped {fn.__name__} (missing columns)")


if __name__ == "__main__":
    main()
