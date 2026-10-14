"""Export benchmark quality plots and a flat run table (optional Matplotlib)."""

from __future__ import annotations

import argparse
import csv
import json
from math import log10
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_results(directory: Path) -> None:
    report = json.loads((directory / "report.json").read_text())
    output = directory / "plots"
    output.mkdir(exist_ok=True)
    names = list(report["instances"])
    profiles = list(report["profiles"])
    labels = [name.replace("_", " ") for name in profiles]
    entries = report["instances"]
    values = [
        [
            log10(entries[name]["tuning"][profile]["median_overhead"])
            - log10(entries[name]["best_initial_overhead"])
            for name in names
        ]
        for profile in profiles
    ]
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 180})
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    heatmap = ax.imshow(values, cmap="YlOrRd", vmin=0, aspect="auto")
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(profiles)), labels)
    ax.set_title("Tuning quality: median of 8 starts / best initializer")
    for row, profile in enumerate(profiles):
        for col, name in enumerate(names):
            ratio = (
                entries[name]["tuning"][profile]["median_overhead"]
                / entries[name]["best_initial_overhead"]
            )
            ax.text(col, row, f"{ratio:.2g}x", ha="center", va="center", color="black")
    fig.colorbar(heatmap, ax=ax, label="log10(cost ratio); lower is better")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"hyperparameter_quality.{suffix}")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    rows = []
    for name in names:
        baseline = entries[name]["best_initial_overhead"]
        runs = json.loads((directory / name / "final_runs.json").read_text())
        best = baseline
        history, ratios = [], []
        for index, run in enumerate(runs, 1):
            value = run["best"]["sampling_overhead"] if run["best"] else None
            if value is not None:
                best = min(best, value)
                ratios.append(value / baseline)
            else:
                ratios.append(float("nan"))
            history.append(best / baseline)
            rows.append(
                {
                    "instance": name,
                    "profile": entries[name]["selected_profile"],
                    "seed": run["seed"],
                    "overhead": value,
                    "ratio_to_best_initializer": value / baseline if value else None,
                    "widths": json.dumps(run["best"]["block_widths"]) if value else None,
                    "evaluations": run["evaluations"],
                    "proposals": run["proposals"],
                    "seconds": run["elapsed_seconds"],
                }
            )
        axes[0].step(range(1, len(runs) + 1), history, where="post", label=name)
        axes[1].scatter(range(1, len(runs) + 1), ratios, s=13, alpha=0.65, label=name)
    axes[0].set_title("Aggregate incumbent includes best initializer")
    axes[0].set_ylabel("Best overhead / initializer overhead")
    axes[0].set_ylim(0.85, 1.02)
    axes[1].set_title("Individual final-start quality")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Run overhead / best initializer overhead")
    axes[1].axhline(1, color="gray", linestyle="--", linewidth=1)
    for ax in axes:
        ax.set_xlabel("Final restart index (seeds 1000–1049)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"restart_quality.{suffix}")
    plt.close(fig)
    with (directory / "final_runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved plots to {output} and {len(rows)} final runs to CSV.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("validation/sa_benchmarks"))
    plot_results(parser.parse_args().directory)
