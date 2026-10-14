#!/usr/bin/env python
"""
Model selection and final report for experiments/train_best.sh.

Uses only the standard library, numpy and matplotlib, so it runs in the repository venv.

    python experiments/best_models.py select        RESULTS_DIR [--metric M] [--minimize]
    python experiments/best_models.py list-selected RESULTS_DIR
    python experiments/best_models.py report        RESULTS_DIR

Layout expected under RESULTS_DIR (written by train_best.sh):

    runs/<method>/seed<k>/model_best.pt, model_last.pt      training output
    runs/<method>/seed<k>/val_model_best/metrics.json       validation of each checkpoint
    runs/<method>/seed<k>/val_model_last/metrics.json
    runs/<method>/seed<k>/test/metrics.json                 test of the selected checkpoint

Selection is done on validation topologies only. The test set is never used to choose a
model, so the reported test numbers are not inflated by selection.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

METHOD_ORDER = ["reliq", "rt-reliq", "reliq-ratio"]
METHOD_LABEL = {
    "reliq": "RELiQ",
    "rt-reliq": "RT-RELiQ",
    "reliq-ratio": "RELiQ + ratio reward",
}
METHOD_COLOR = {"reliq": "#d95f02", "rt-reliq": "#1b9e77", "reliq-ratio": "#7570b3"}
CHECKPOINTS = ["model_best", "model_last"]

# (metric key, label, higher is better)
REPORT_METRICS = [
    ("average_episode_packets_mean", "entanglements per episode", True),
    ("episode_success_rate_mean", "success rate", True),
    ("total_episode_fidelity_mean", "mean delivered fidelity", True),
    ("total_links_per_success_mean", "quantum links per success", False),
]
LINK_PARTS = [
    ("resources_per_success_mean", "routing", "#4575b4"),
    ("purification_links_agent_per_success_mean", "agent purification", "#1b9e77"),
    ("purification_links_auto_per_success_mean", "automatic purification", "#d95f02"),
]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def runs(root):
    """Yield (method, seed, run_dir) for every run directory, in a stable order."""
    base = root / "runs"
    if not base.exists():
        return
    methods = sorted(
        (p.name for p in base.iterdir() if p.is_dir()),
        key=lambda m: (METHOD_ORDER.index(m) if m in METHOD_ORDER else 99, m),
    )
    for method in methods:
        for run_dir in sorted((base / method).glob("seed*"), key=lambda p: int(p.name[4:])):
            yield method, int(run_dir.name[4:]), run_dir


def score(metrics, metric):
    value = metrics.get(metric)
    if value is None or not np.isfinite(value):
        return None
    return float(value)


# --------------------------------------------------------------------------------------
def cmd_select(args):
    root = Path(args.results_dir)
    sign = -1.0 if args.minimize else 1.0
    selection = {"metric": args.metric, "minimize": args.minimize, "methods": {}}

    for method, seed, run_dir in runs(root):
        candidates = []
        for ckpt in CHECKPOINTS:
            mpath = run_dir / f"val_{ckpt}" / "metrics.json"
            cpath = run_dir / f"{ckpt}.pt"
            if not (mpath.exists() and cpath.exists()):
                continue
            s = score(load_json(mpath), args.metric)
            if s is not None:
                candidates.append((s, ckpt))
        entry = selection["methods"].setdefault(method, {"runs": {}, "best": None})
        if not candidates:
            print(f"[select] {method} seed={seed}: no validated checkpoint, skipped")
            continue
        s, ckpt = max(candidates, key=lambda c: sign * c[0])
        entry["runs"][str(seed)] = {
            "run_dir": str(run_dir),
            "checkpoint": ckpt,
            "checkpoint_path": str(run_dir / f"{ckpt}.pt"),
            "validation_score": s,
            "validation_scores": {c: v for v, c in candidates},
        }

    best_dir = root / "best_models"
    best_dir.mkdir(parents=True, exist_ok=True)
    for method, entry in selection["methods"].items():
        if not entry["runs"]:
            continue
        seed, run = max(entry["runs"].items(), key=lambda kv: sign * kv[1]["validation_score"])
        entry["best"] = {"seed": int(seed), **run}
        target = best_dir / f"{method}.pt"
        shutil.copy2(run["checkpoint_path"], target)
        entry["best"]["copied_to"] = str(target)
        print(
            f"[select] {METHOD_LABEL.get(method, method)}: seed {seed} {run['checkpoint']} "
            f"({args.metric} = {run['validation_score']:.4g} on validation) -> {target}"
        )

    with open(root / "selection.json", "w") as f:
        json.dump(selection, f, indent=2)
    print(f"[select] wrote {root / 'selection.json'}")


def cmd_list_selected(args):
    root = Path(args.results_dir)
    selection = load_json(root / "selection.json")
    for entry in selection["methods"].values():
        for run in entry["runs"].values():
            print(run["run_dir"], run["checkpoint_path"])


# --------------------------------------------------------------------------------------
def collect_test(root, selection):
    """method -> {"best": metrics or None, "seeds": {seed: metrics}}"""
    out = {}
    for method, entry in selection["methods"].items():
        seeds = {}
        for seed, run in entry["runs"].items():
            mpath = Path(run["run_dir"]) / "test" / "metrics.json"
            if mpath.exists():
                seeds[int(seed)] = load_json(mpath)
        best = None
        if entry.get("best") is not None:
            best = seeds.get(entry["best"]["seed"])
        if seeds:
            out[method] = {"best": best, "seeds": seeds, "best_seed": entry["best"]["seed"]}
    return out


def fmt(x, digits=3):
    return "–" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


def write_table(results, selection, path):
    lines = [
        "# Best model per method (test topologies)",
        "",
        f"Selection: `{selection['metric']}` on validation topologies "
        f"({'lower' if selection['minimize'] else 'higher'} is better). "
        "Test topologies were not used for selection.",
        "",
        "| method | best seed / checkpoint | "
        + " | ".join(label for _, label, _ in REPORT_METRICS)
        + " |",
        "| --- | --- | " + " | ".join("---" for _ in REPORT_METRICS) + " |",
    ]
    for method in sorted(results, key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99):
        r = results[method]
        run = selection["methods"][method]["best"]
        cells = []
        for key, _, _ in REPORT_METRICS:
            best = r["best"].get(key) if r["best"] else None
            vals = [m.get(key) for m in r["seeds"].values() if m.get(key) is not None]
            spread = f" (seeds {np.mean(vals):.3f} ± {np.std(vals):.3f}, n={len(vals)})" if len(vals) > 1 else ""
            cells.append(f"**{fmt(best)}**{spread}")
        lines.append(
            f"| {METHOD_LABEL.get(method, method)} | {run['seed']} / {run['checkpoint']} | "
            + " | ".join(cells)
            + " |"
        )
    lines += [
        "",
        "Bold: the selected best model. In brackets: mean ± sd over every seed's selected "
        "checkpoint.",
        "`quantum links per success` = routing + agent purification + automatic "
        "purification (`total_links_per_success`).",
    ]
    path.write_text("\n".join(lines) + "\n")


def plot(results, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = sorted(results, key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)
    x = np.arange(len(methods))
    labels = [METHOD_LABEL.get(m, m) for m in methods]
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, len(REPORT_METRICS), figsize=(3.6 * len(REPORT_METRICS), 3.8))
    for ax, (key, label, higher) in zip(axes, REPORT_METRICS):
        if key == "total_links_per_success_mean":
            # composition of the cost for the best model
            bottom = np.zeros(len(methods))
            for part, part_label, color in LINK_PARTS:
                vals = np.array([
                    (results[m]["best"] or {}).get(part, 0.0) or 0.0 for m in methods
                ])
                ax.bar(x, vals, 0.6, bottom=bottom, color=color, label=part_label,
                       edgecolor="white", linewidth=0.5)
                bottom += vals
            ax.legend(frameon=False, fontsize=7, loc="upper right")
        else:
            vals = [(results[m]["best"] or {}).get(key, np.nan) for m in methods]
            ax.bar(x, vals, 0.6, color=[METHOD_COLOR.get(m, "#888888") for m in methods])
        # every seed's selected model as a dot
        for xi, m in zip(x, methods):
            seed_vals = [s.get(key) for s in results[m]["seeds"].values() if s.get(key) is not None]
            jitter = rng.uniform(-0.12, 0.12, len(seed_vals))
            ax.scatter(xi + jitter, seed_vals, s=14, color="black", zorder=3,
                       label="each seed" if (xi == 0 and key == REPORT_METRICS[0][0]) else None)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(f"{label} ({'higher' if higher else 'lower'} is better)", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("Best model per method on the test topologies (bars); dots: every seed", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def cmd_report(args):
    root = Path(args.results_dir)
    selection = load_json(root / "selection.json")
    results = collect_test(root, selection)
    if not results:
        sys.exit("[report] no test results found; run the test phase first")
    out = root / "report"
    out.mkdir(parents=True, exist_ok=True)
    write_table(results, selection, out / "results.md")
    plot(results, out / "best_models.png")
    with open(out / "test_metrics_best.json", "w") as f:
        json.dump(
            {
                m: {k: v for k, v in (r["best"] or {}).items() if not isinstance(v, list)}
                for m, r in results.items()
            },
            f, indent=2, sort_keys=True,
        )
    print(f"[report] wrote {out / 'results.md'}, {out / 'best_models.png'}, {out / 'test_metrics_best.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select")
    p.add_argument("results_dir")
    p.add_argument("--metric", default="average_episode_packets_mean")
    p.add_argument("--minimize", action="store_true")
    p.set_defaults(fn=cmd_select)
    p = sub.add_parser("list-selected")
    p.add_argument("results_dir")
    p.set_defaults(fn=cmd_list_selected)
    p = sub.add_parser("report")
    p.add_argument("results_dir")
    p.set_defaults(fn=cmd_report)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
