"""Scientific cost/quality plots and the exact contracted winning QFT12 graph."""

from __future__ import annotations

import argparse
import json
from math import asin, pi
from pathlib import Path


def export_graph(directory: Path):
    from qpartition import HybridPlan, PartitionGraph, load_problem

    folder = directory / "qft12"
    graph = PartitionGraph(load_problem(folder))
    plan = HybridPlan.load(folder / "best_plan.json")
    evaluation = plan.validate(graph)
    blocks = graph.labels_from_evaluation(evaluation)
    vertices = [
        {
            "index": i,
            "qubit": node["qubit"],
            "operation": int(node["operation_id"][2:]),
            "block": blocks[i],
        }
        for i, node in enumerate(graph.nodes)
    ]
    gates = [
        {"operation": int(v.operation_id[2:]), "ports": v.ports, "cut": bool(value)}
        for v, value in zip(graph.variables, plan.state)
        if v.kind == "gate"
    ]
    wires = [
        {
            "qubit": v.qubit,
            "operation": int(v.operation_id[2:]),
            "old_block": blocks[v.ports[0]],
            "new_block": blocks[v.ports[1]],
        }
        for v, value in zip(graph.variables, plan.state)
        if v.kind == "wire" and value
    ]
    data = {
        "fingerprint": graph.problem.circuit_fingerprint,
        "vertices": vertices,
        "gates": gates,
        "wire_cuts": wires,
        "initial_blocks": [blocks[v] for v in graph.input_nodes],
        "widths": evaluation.block_widths,
        "overhead": evaluation.sampling_overhead,
        "graph_schema": graph.graph_schema,
    }
    (folder / "graph_figure_data.json").write_text(json.dumps(data, indent=2) + "\n")


def render(directory: Path):
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    report = json.loads((directory / "report.json").read_text())
    entries = report["instances"]
    output = directory / "plots"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 180})
    colors = ("#1673A8", "#D57719")
    pink = "#B72D70"

    def save(fig, stem):
        for suffix in ("png", "svg", "pdf"):
            path = output / f"{stem}.{suffix}"
            fig.savefig(path, facecolor="white")
            if suffix == "svg":
                path.write_text(
                    "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
                )
        plt.close(fig)

    theta = np.linspace(0, pi, 500)
    six_phases = (1 + 2 * np.abs(np.sin(theta / 2))) ** 12
    crossover = 2 * asin((16 ** (1 / 12) - 1) / 2)
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    ax.semilogy(
        theta, six_phases, color=colors[0], lw=2, label="Six CP cuts: [1+2|sin(theta/2)|]^12"
    )
    ax.axhline(16, color=colors[1], lw=2, label="One wire cut: 16")
    ax.axvline(crossover, color="gray", ls="--", label=f"Equal cost at {crossover:.3f} rad")
    for name in ("bridge_weak", "bridge_strong"):
        entry = entries[name]
        angle = entry["provenance"]["phase_angle"]
        cost = entry["final"]["best_overhead"]
        ax.scatter(angle, cost, color=pink, s=45, zorder=4)
        ax.annotate(
            f"{name}: {cost:.3g}",
            (angle, cost),
            xytext=(0, 15),
            textcoords="offset points",
            ha="right" if angle > 1 else "left",
        )
    ax.set_xlabel("Controlled-phase angle theta (radians)")
    ax.set_ylabel("Physical sampling-overhead factor")
    ax.set_title("Identical topology and capacity: gate/wire preference changes with angle")
    ax.set_xlim(-0.1, pi + 0.15)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")
    save(fig, "gate_wire_cost_crossover")

    names = list(entries)
    positions = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(13, 5), layout="constrained")
    for shift, key, label, color in (
        (-0.2, "gate_only_seed_overhead", "Spatial gate-only seed", colors[0]),
        (0, "deterministic_refinement_overhead", "Deterministic interval refinement", "#6C995A"),
        (0.2, None, "SA + polishing, best of 50", colors[1]),
    ):
        values = [entries[n][key] if key else entries[n]["final"]["best_overhead"] for n in names]
        ax.bar(positions + shift, values, width=0.2, label=label, color=color)
    ax.set_yscale("log")
    ax.set_xticks(positions, names, rotation=30, ha="right")
    ax.set_ylabel("Physical sampling-overhead factor (log scale)")
    ax.set_title(
        "Separate initializer/refinement gains from SA gains; controls have no Table-I claim"
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper left")
    save(fig, "benchmark_attribution")

    profiles = list(report["profiles"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, name in zip(axes, ("qft6", "qft12")):
        baseline = entries[name]["deterministic_refinement_overhead"]
        distributions = [
            [
                run["best"]["sampling_overhead"] / baseline
                for run in json.loads((directory / name / f"tuning_{profile}.json").read_text())
                if run["best"]
            ]
            for profile in profiles
        ]
        ax.boxplot(distributions, showfliers=True)
        ax.set_xticks(
            range(1, len(profiles) + 1),
            [profile.replace("_", "\n") for profile in profiles],
            fontsize=8,
        )
        ax.set_yscale("log")
        ax.axhline(1, color="gray", ls="--")
        ax.set_title(f"{name}: six-start tuning distributions")
        ax.set_ylabel("Run cost / deterministic-refinement cost")
        ax.grid(axis="y", alpha=0.2)
    save(fig, "proposal_quality")

    data = json.loads((directory / "qft12/graph_figure_data.json").read_text())
    vertices = {v["index"]: v for v in data["vertices"]}
    fig, axes = plt.subplots(2, 1, figsize=(18, 9), layout="constrained")
    fig.suptitle(
        "Winning contracted QFT12 graph: 144 incidence vertices, 204 edges\n"
        f"24 gate cuts + 2 wire cuts | widths 5 + 9 (limit 9) | overhead {data['overhead']:,.2f}",
        fontsize=16,
    )
    for panel, ax in enumerate(axes):
        first, last = 42 * panel, 42 * panel + 41
        left, right = first - 0.65, last + 0.65
        for q, initial in enumerate(data["initial_blocks"]):
            cursor, block = left, initial
            for wire in data["wire_cuts"]:
                if wire["qubit"] != q:
                    continue
                boundary = wire["operation"] - 0.7
                if boundary < left:
                    block = wire["new_block"]
                elif boundary <= right:
                    ax.plot([cursor, boundary], [q, q], color=colors[block], lw=1.5)
                    ax.plot(
                        [boundary - 0.15, boundary + 0.05], [q + 0.18, q - 0.18], color=pink, lw=2.8
                    )
                    ax.plot(
                        [boundary + 0.03, boundary + 0.23], [q + 0.18, q - 0.18], color=pink, lw=2.8
                    )
                    cursor, block = boundary, wire["new_block"]
            ax.plot([cursor, right], [q, q], color=colors[block], lw=1.5)
        for gate in data["gates"]:
            op = gate["operation"]
            if not first <= op <= last:
                continue
            a, b = (vertices[index] for index in gate["ports"])
            color = pink if gate["cut"] else colors[a["block"]]
            ax.plot(
                [op, op],
                [a["qubit"], b["qubit"]],
                color=color,
                ls="--" if gate["cut"] else "-",
                lw=1,
                alpha=0.85,
            )
            for vertex in (a, b):
                ax.plot(
                    op,
                    vertex["qubit"],
                    "o",
                    color=colors[vertex["block"]],
                    ms=4,
                    mec="white",
                    mew=0.35,
                    zorder=4,
                )
        ax.set_xlim(left, right)
        ax.set_ylim(11.6, -0.65)
        ax.set_yticks(range(12), [f"q{q}" for q in range(12)])
        ax.set_xticks(range(first, last + 1), [str(i) for i in range(first, last + 1)], fontsize=7)
        ax.tick_params(length=0)
        ax.set_title(
            f"Original operation positions {first}–{last}; single-qubit gates omitted", fontsize=11
        )
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[-1].legend(
        handles=[
            Line2D([0], [0], color=colors[0], lw=3, label="A: q0, q10, q11, early q9, late q1"),
            Line2D([0], [0], color=colors[1], lw=3, label="B: q2–q8, late q9, early q1"),
            Line2D([0], [0], color=pink, ls="--", label="Cut gate edge"),
            Line2D([0], [0], color=pink, marker="x", ls="", label="Wire cuts before #24 / #76"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        fontsize=10,
    )
    axes[-1].set_xlabel(
        "Topology view: terminal boundaries alias first/last incidence; not a QPD branch."
    )
    save(fig, "qft12_contracted_graph")
    print(f"Saved scientific figures in {output} (PNG, SVG, PDF).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("validation/sa_benchmarks_v2"))
    parser.add_argument("--mode", choices=("export", "render", "both"), default="both")
    args = parser.parse_args()
    if args.mode in ("export", "both"):
        export_graph(args.directory)
    if args.mode in ("render", "both"):
        render(args.directory)
