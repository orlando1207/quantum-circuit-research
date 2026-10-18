"""Draw exact QFT12 plan annotations; export and render may use separate environments."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from math import pi, prod
from pathlib import Path


def export_data(directory: Path, output: Path) -> dict:
    from qpartition import HybridPlan, load_problem
    from qpartition.legacy_graph import LegacyPartitionGraph as PartitionGraph

    folder = directory / "qft12"
    graph = PartitionGraph(load_problem(folder))
    plan = HybridPlan.load(folder / "best_plan.json")
    best = plan.validate(graph)
    initialization = json.loads((folder / "initialization.json").read_text())
    initial_index = min(
        range(len(initialization["overheads"])), key=initialization["overheads"].__getitem__
    )
    initial = tuple(initialization["states"][initial_index])
    spatial = graph.evaluate(initial, max_qubits=plan.max_fragment_qubits)
    report = json.loads((directory / "report.json").read_text())["instances"]["qft12"]

    def location(evaluation, port):
        segment = evaluation.node_segments[port]
        return evaluation.component_blocks[evaluation.segment_components[segment]]

    operations, wire_cuts, removed = [], [], []
    for index, (op_id, item) in enumerate(
        zip(graph.problem.operation_ids, graph.problem.circuit.data)
    ):
        if item.operation.name == "barrier":
            continue
        variable = graph.gate_variables[op_id]
        selected = graph.variables[variable].options[plan.state[variable]]
        old = graph.variables[variable].options[initial[variable]]
        record = {
            "index": index,
            "id": op_id,
            "name": item.operation.name,
            "qubits": [graph.problem.circuit.find_bit(q).index for q in item.qubits],
            "params": [float(value) for value in item.operation.params],
            "action": selected.kind if selected else "retain",
            "input_blocks": [location(best, pin) for pin, _ in graph.gate_ports[op_id]],
            "output_blocks": [location(best, pout) for _, pout in graph.gate_ports[op_id]],
        }
        operations.append(record)
        if old is not None and old.kind == "gate" and selected is None:
            removed.append({**record, "old_overhead": old.gamma**2})
        for wire in graph.incoming_wires[op_id]:
            if wire is not None and plan.state[wire]:
                candidate = graph.variables[wire]
                left, right = candidate.ports
                wire_cuts.append(
                    {
                        "before_index": index,
                        "before_id": op_id,
                        "qubit": candidate.qubit,
                        "old_block": location(best, left),
                        "new_block": location(best, right),
                        "overhead": candidate.options[plan.state[wire]].gamma ** 2,
                    }
                )
    # The comparison inset is specific to this verified first-version result.
    assert len(removed) == 2 and len(wire_cuts) == 1
    assert best.action_counts == {"wire": 1, "gate": 30, "excision": 0}
    assert all(
        (not old and not new)
        or old == new
        or (var.kind == "wire" and old == 0)
        or var.operation_id in {op["id"] for op in removed}
        for var, old, new in zip(graph.variables, initial, plan.state)
    ), "The inset assumes all other decisions are unchanged."
    ratio = prod(w["overhead"] for w in wire_cuts) / prod(op["old_overhead"] for op in removed)
    assert abs(ratio - best.sampling_overhead / spatial.sampling_overhead) < 1e-10
    data = {
        "schema_version": 1,
        "fingerprint": graph.problem.circuit_fingerprint,
        "qubits": graph.problem.circuit.num_qubits,
        "max_qubits": plan.max_fragment_qubits,
        "operations": operations,
        "wire_cuts": wire_cuts,
        "removed_gate_cuts": removed,
        "initial_blocks": [location(best, p) for p in graph.input_nodes],
        "baseline_overhead": spatial.sampling_overhead,
        "baseline_widths": spatial.block_widths,
        "best_overhead": best.sampling_overhead,
        "best_widths": best.block_widths,
        "ratio": ratio,
        "best_seed": report["best_seed"],
        "final_summary": report["final"],
        "comparison_scope": "Count/spatial-cost-matched QFT reconstruction, not author artifact.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "qft12_figure_data.json").write_text(json.dumps(data, indent=2) + "\n")
    return data


def render(data: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    colors = ("#1673A8", "#D57719")
    cut_color = "#B72D70"
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 180})

    def phase_label(angle):
        value = Fraction(angle / pi).limit_denominator(4096)
        if value.numerator == 1:
            return r"$\pi$" if value.denominator == 1 else rf"$\pi/{value.denominator}$"
        return rf"${value.numerator}\pi/{value.denominator}$"

    def phase_gate(ax, x, control, target, angle, color, cut=False, size=0.8, fontsize=6.8):
        ax.plot([x, x], [control, target], color=color, ls="--" if cut else "-", lw=1.1, zorder=2)
        ax.plot(x, control, "o", color=color, mfc="white" if cut else color, ms=4, zorder=4)
        ax.add_patch(
            Rectangle(
                (x - size / 2, target - 0.36),
                size,
                0.72,
                facecolor="white",
                edgecolor=color,
                linewidth=1.2,
                zorder=4,
            )
        )
        ax.text(
            x,
            target,
            "P\n" + phase_label(angle),
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            zorder=5,
        )

    def save(fig, stem):
        for suffix in ("png", "svg", "pdf"):
            path = output / f"{stem}.{suffix}"
            fig.savefig(path, facecolor="white")
            if suffix == "svg":
                path.write_text(
                    "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
                )
        plt.close(fig)

    # Full circuit: each column is one original operation, preserving exact order.
    fig, axes = plt.subplots(3, 1, figsize=(16, 13), layout="constrained")
    fig.suptitle(
        "Optimized QFT12: exact cut positions from the saved plan\n"
        f"Overhead {data['best_overhead']:,.2f} | widths 5 + 8 (limit 9) | "
        "30 CP cuts + 1 wire cut | seed 1016",
        fontsize=15,
    )
    for panel, ax in enumerate(axes):
        first, last = panel * 28, (panel + 1) * 28 - 1
        left, right = first - 0.65, last + 0.65
        for q, initial_block in enumerate(data["initial_blocks"]):
            cursor, block = left, initial_block
            for wire in data["wire_cuts"]:
                if wire["qubit"] != q:
                    continue
                boundary = wire["before_index"] - 0.7
                if boundary < left:
                    block = wire["new_block"]
                elif boundary <= right:
                    ax.plot([cursor, boundary], [q, q], color=colors[block], lw=1.6)
                    ax.plot(
                        [boundary - 0.12, boundary + 0.12],
                        [q + 0.2, q - 0.2],
                        color=cut_color,
                        lw=2.4,
                        zorder=6,
                    )
                    ax.plot(
                        [boundary + 0.08, boundary + 0.32],
                        [q + 0.2, q - 0.2],
                        color=cut_color,
                        lw=2.4,
                        zorder=6,
                    )
                    cursor, block = boundary, wire["new_block"]
            ax.plot([cursor, right], [q, q], color=colors[block], lw=1.6)
        for op in data["operations"]:
            x = op["index"]
            if not first <= x <= last:
                continue
            qs = op["qubits"]
            color = cut_color if op["action"] != "retain" else colors[op["input_blocks"][0]]
            if op["name"] == "cp":
                phase_gate(ax, x, *qs, op["params"][0], color, op["action"] == "gate")
            elif op["name"] == "h":
                q = qs[0]
                ax.add_patch(
                    Rectangle((x - 0.3, q - 0.23), 0.6, 0.46, fc="white", ec=color, zorder=4)
                )
                ax.text(x, q, "H", ha="center", va="center", color=color, fontsize=8, zorder=5)
            elif op["name"] == "swap":
                ax.plot([x, x], qs, color=color, lw=1.3)
                ax.plot([x, x], qs, "x", color=color, ms=7, mew=1.5)
            else:
                raise ValueError(f"Unsupported figure gate: {op['name']}")
        ax.set_xlim(left, right)
        ax.set_ylim(11.6, -0.75)
        ax.set_yticks(range(12), [f"q{q}" for q in range(12)])
        ax.set_xticks(range(first, last + 1), [str(i) for i in range(first, last + 1)])
        ax.tick_params(axis="x", labelsize=7, length=0)
        ax.tick_params(axis="y", length=0)
        ax.set_title(f"Original operation indices {first}–{last} (left to right)", fontsize=11)
        for spine in ax.spines.values():
            spine.set_visible(False)
    legend = [
        Line2D([0], [0], color=colors[0], lw=3, label="A: 5 static qubits"),
        Line2D([0], [0], color=colors[1], lw=3, label="B: 8 static qubits"),
        Line2D([0], [0], color=cut_color, ls="--", marker="o", mfc="white", label="Gate cut"),
        Line2D([0], [0], color=cut_color, marker="x", ls="", label="Wire cut before H(q9), #23"),
    ]
    axes[-1].legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4)
    axes[-1].set_xlabel(
        "Logical circuit with cut annotations; physical QPD branches replace marked operations."
    )
    save(fig, "qft12_optimized_circuit")

    # Comparison inset: all omitted gates are explicitly identified as omitted.
    fig = plt.figure(figsize=(14, 7.5), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=(0.8, 2, 1.3))
    header = fig.add_subplot(grid[0, :])
    header.axis("off")
    improvement = 100 * (1 - data["ratio"])
    header.text(0.5, 0.8, "QFT12: what the first SA version improves", ha="center", fontsize=19)
    header.text(
        0.5,
        0.26,
        f"Spatial seed: {data['baseline_overhead']:,.2f}    →    "
        f"Hybrid: {data['best_overhead']:,.2f}    |    {improvement:.2f}% lower overhead",
        ha="center",
        fontsize=14,
        color="#145638",
    )
    for panel, hybrid in enumerate((False, True)):
        ax = fig.add_subplot(grid[1, panel])
        qs = (11, 10, 9)
        for row, q in enumerate(qs):
            if q == 9:
                ax.plot([0.3, 5.2], [row, row], color=colors[0 if hybrid else 1], lw=3)
                ax.plot([5.2, 7.5], [row, row], color=colors[1], lw=3)
            else:
                ax.plot([0.3, 7.5], [row, row], color=colors[0], lw=3)
        for x, op in zip((1.7, 3.8), data["removed_gate_cuts"]):
            phase_gate(
                ax,
                x,
                qs.index(op["qubits"][0]),
                qs.index(op["qubits"][1]),
                op["params"][0],
                colors[0] if hybrid else cut_color,
                cut=not hybrid,
                size=1,
                fontsize=11,
            )
            ax.text(x, -0.6, f"#{op['index']}", ha="center", fontsize=10)
        if hybrid:
            ax.plot([5.03, 5.25], [2.18, 1.82], color=cut_color, lw=3)
            ax.plot([5.23, 5.45], [2.18, 1.82], color=cut_color, lw=3)
            ax.text(5.2, 2.6, "Wire cut\ncost 16", ha="center", color=cut_color, fontsize=10)
        ax.add_patch(Rectangle((6.2, 1.75), 0.6, 0.5, fc="white", ec=colors[1], zorder=5))
        ax.text(6.5, 2, "H", ha="center", va="center", color=colors[1], zorder=6)
        ax.text(6.5, -0.6, "#23", ha="center", fontsize=10)
        ax.set_xlim(0, 8)
        ax.set_ylim(3.2, -0.9)
        ax.set_yticks(range(3), [f"q{q}" for q in qs])
        ax.set_xticks([])
        ax.set_title(
            "Hybrid: keep both early gates local" if hybrid else "Spatial: cut both early gates",
            fontsize=13,
        )
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("Inset omits other gates; original operation indices shown.", fontsize=9)
    footer = fig.add_subplot(grid[2, :])
    footer.axis("off")
    factors = [op["old_overhead"] for op in data["removed_gate_cuts"]]
    footer.text(
        0.5,
        0.95,
        f"Replace two gate-cut factors {factors[0]:.4f} × {factors[1]:.4f} = "
        f"{prod(factors):.4f} with one wire-cut factor 16.\n"
        "The other 30 gate cuts are unchanged; no SWAP cuts or excisions.",
        ha="center",
        va="top",
        fontsize=12,
    )
    footer.text(
        0.5,
        0.48,
        "A = q0, q1, q10, q11, early q9 (5 qubits)\n"
        "B = q2–q8, late q9 (8 qubits); both within limit 9",
        ha="center",
        va="top",
        fontsize=11,
    )
    footer.text(
        0.5,
        0.07,
        "Quality limit: best found in 1/50 final starts; median overhead 34.6 million.\n"
        "Reconstructed benchmark; no optimality certificate or hardware execution.",
        ha="center",
        va="top",
        fontsize=11,
        color="#843E12",
    )
    save(fig, "qft12_optimization_summary")
    print(f"Saved QFT12 circuit and comparison figures to {output} (PNG, SVG, PDF).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("validation/sa_benchmarks"))
    parser.add_argument("--output", type=Path, default=Path("docs/images"))
    parser.add_argument("--mode", choices=("export", "render", "both"), default="both")
    args = parser.parse_args()
    if args.mode in ("export", "both"):
        figure_data = export_data(args.directory, args.output)
    else:
        figure_data = json.loads((args.output / "qft12_figure_data.json").read_text())
    if args.mode in ("render", "both"):
        render(figure_data, args.output)
