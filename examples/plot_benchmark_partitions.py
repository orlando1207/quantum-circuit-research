"""Draw PNG-only logical fragment templates from every saved benchmark winner.

Export with the research conda Python, then render with a matplotlib Python.
The temporary JSON separates those environments; it is not a deliverable.
QPD endpoints denote families of local operations, not a single sampled branch.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from math import ceil, isclose, pi
from pathlib import Path


def export(directory: Path, destination: Path, names: list[str] | None = None) -> None:
    from qpartition import HybridPlan, PartitionGraph, compile_hybrid, load_problem

    report = json.loads((directory / "report.json").read_text())
    records = []
    for name, entry in report["instances"].items():
        if names and name not in names:
            continue
        folder = directory / name
        graph = PartitionGraph(load_problem(folder))
        plan = HybridPlan.load(folder / "best_plan.json")
        compiled = compile_hybrid(graph, plan)
        evaluation = compiled.evaluation
        size = len(graph.problem.circuit.data)

        def location(vertex):
            return compiled.segment_locations[evaluation.node_segments[vertex]]

        segments = [None] * evaluation.segment_count
        for vertex, metadata in enumerate(graph.nodes):
            segment = evaluation.node_segments[vertex]
            if segments[segment] is None:
                segments[segment] = {
                    "qubit": metadata["qubit"],
                    "location": location(vertex),
                    "start": -1.0,
                    "end": size - 0.1,
                }
            elif segments[segment]["qubit"] != metadata["qubit"]:
                raise ValueError("A temporal segment spans different source qubits.")
        operations, cuts = [], []
        for index, (op_id, item) in enumerate(
            zip(graph.problem.operation_ids, graph.problem.circuit.data, strict=True)
        ):
            if item.operation.name == "barrier":
                continue
            variable = graph.gate_variables.get(op_id)
            ports = [pair[0] for pair in graph.gate_ports[op_id]]
            cut = variable is not None and bool(plan.state[variable])
            locations = [location(vertex) for vertex in ports]
            if len(ports) == 2 and not cut and locations[0][0] != locations[1][0]:
                raise ValueError("A retained gate crosses fragments.")
            operations.append(
                {
                    "index": index,
                    "name": item.operation.name,
                    "params": [float(value) for value in item.operation.params],
                    "qubits": [graph.problem.circuit.find_bit(q).index for q in item.qubits],
                    "locations": locations,
                    "cut": cut,
                }
            )
            for wire in graph.incoming_wires[op_id]:
                if wire is None or not plan.state[wire]:
                    continue
                candidate = graph.variables[wire]
                left, right = candidate.ports
                segments[evaluation.node_segments[left]]["end"] = index - 0.55
                segments[evaluation.node_segments[right]]["start"] = index - 0.47
                cuts.append(
                    {
                        "id": len(cuts) + 1,
                        "index": index,
                        "qubit": candidate.qubit,
                        "old": location(left),
                        "new": location(right),
                    }
                )
        if not isclose(evaluation.sampling_overhead, entry["final"]["best_overhead"]):
            raise ValueError(f"{name}: saved plan and report overhead disagree.")
        if sum(evaluation.block_widths) != graph.problem.circuit.num_qubits + len(cuts):
            raise ValueError("Static register count does not conserve cut wire segments.")
        if sum(op["cut"] for op in operations) != evaluation.action_counts["gate"]:
            raise ValueError("Gate annotation count does not match the plan.")
        records.append(
            {
                "name": name,
                "qubits": graph.problem.circuit.num_qubits,
                "size": size,
                "operations": operations,
                "wire_cuts": cuts,
                "segments": segments,
                "widths": evaluation.block_widths,
                "limit": plan.max_fragment_qubits,
                "overhead": evaluation.sampling_overhead,
                "counts": evaluation.action_counts,
                "fingerprint": graph.problem.circuit_fingerprint,
            }
        )
    destination.write_text(json.dumps(records))
    print(f"Validated and exported {len(records)} saved benchmark plans.")


def angle(value: float) -> str:
    fraction = Fraction(value / pi).limit_denominator(4096)
    if isclose(float(fraction) * pi, value, abs_tol=1e-9):
        if not fraction:
            return "0"
        numerator, denominator = fraction.numerator, fraction.denominator
        prefix = "−" if numerator < 0 else ""
        coefficient = "" if abs(numerator) == 1 else str(abs(numerator))
        return f"{prefix}{coefficient}π" + (f"/{denominator}" if denominator != 1 else "")
    return f"{value:.3g}"


def render(data: Path, output: Path, columns: int, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    colors = ("#1673A8", "#D57719", "#34865E", "#7660A4", "#BA5555", "#658696")
    magenta = "#B72D70"
    ink = "#28313D"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    output.mkdir(parents=True, exist_ok=True)

    def box(ax, x, y, label, color, *, cut=False, fontsize=7):
        box_height = 0.86 if label.count("\n") > 2 else 0.64
        ax.add_patch(
            Rectangle(
                (x - 0.41, y - box_height / 2),
                0.82,
                box_height,
                facecolor="white",
                edgecolor=color,
                linewidth=1.2,
                linestyle="--" if cut else "-",
                zorder=4,
            )
        )
        ax.text(
            x,
            y,
            label,
            fontsize=fontsize,
            ha="center",
            va="center",
            color=color,
            linespacing=1.05,
            zorder=5,
        )

    def gate(ax, operation, rows, color):
        x = operation["index"]
        name, params = operation["name"], operation["params"]
        if len(rows) == 1:
            label = name.upper()
            if params:
                label += "\n" + (
                    angle(params[0]) if len(params) == 1 else "\n".join(angle(p) for p in params)
                )
            box(ax, x, rows[0], label, color, fontsize=5 if len(params) > 1 else 7)
            return
        a, b = rows
        ax.plot([x, x], [a, b], color=color, lw=1.2, zorder=2)
        if name in ("cx", "cy", "ch", "cp", "cz"):
            ax.plot(x, a, "o", color=color, markersize=4, zorder=5)
            if name == "cx":
                ax.plot(
                    x,
                    b,
                    "o",
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markersize=9,
                    zorder=4,
                )
                ax.plot([x - 0.19, x + 0.19], [b, b], color=color, lw=1.1, zorder=5)
                ax.plot([x, x], [b - 0.21, b + 0.21], color=color, lw=1.1, zorder=5)
            elif name == "cz":
                ax.plot(x, b, "o", color=color, markersize=4, zorder=5)
            else:
                label = {"cy": "Y", "ch": "H", "cp": "P"}[name]
                if params:
                    label += "\n" + angle(params[0])
                box(ax, x, b, label, color)
        elif name == "swap":
            for y in rows:
                ax.plot([x - 0.17, x + 0.17], [y - 0.19, y + 0.19], color=color, lw=1.4, zorder=5)
                ax.plot([x - 0.17, x + 0.17], [y + 0.19, y - 0.19], color=color, lw=1.4, zorder=5)
        else:
            label = name.upper() + ("\n" + angle(params[0]) if params else "")
            for y in rows:
                box(ax, x, y, label, color)

    for record in json.loads(data.read_text()):
        n, size = record["qubits"], record["size"]
        widths, cuts = record["widths"], record["wire_cuts"]
        fragments = len(widths)
        if fragments > len(colors):
            colors = tuple(
                plt.get_cmap("turbo")(i / max(1, fragments - 1)) for i in range(fragments)
            )
        names = [chr(65 + b) if b < 26 else f"F{b}" for b in range(fragments)]
        plot_columns = min(3, fragments)
        groups = ceil(fragments / plot_columns)
        folds = ceil(size / columns)
        overview_height = 0.19 * n + 0.8
        panel_height = 0.35 * max(widths) + 0.95
        header_height = 1.65 + 0.3 * (ceil((fragments + 2) / 4) - 1)
        height = overview_height + folds * groups * panel_height + header_height + 1.05
        fig = plt.figure(figsize=(21, height), facecolor="white")
        grid = fig.add_gridspec(
            folds * groups + 1,
            plot_columns,
            height_ratios=[overview_height] + [panel_height] * (folds * groups),
            left=0.07,
            right=0.98,
            top=1 - header_height / height,
            bottom=1.05 / height,
            hspace=0.4,
            wspace=0.18,
        )
        fig.text(
            0.07,
            1 - 0.32 / height,
            f"{record['name']}  |  optimized {fragments}-way partition",
            fontsize=20,
            fontweight="bold",
            color=ink,
            va="top",
        )
        counts = record["counts"]
        fig.text(
            0.07,
            1 - 0.76 / height,
            f"{n} source qubits   •   fragment widths: {widths}"
            f"   •   limit: {record['limit']} per fragment   •   "
            f"{counts['gate']} gate cuts + {counts['wire']} wire cuts   •   "
            f"sampling overhead Ω = {record['overhead']:.6g}",
            fontsize=11,
            color=ink,
            va="top",
        )
        handles = [
            Line2D([], [], color=colors[b], lw=3, label=f"Fragment {names[b]}")
            for b in range(fragments)
        ] + [
            Line2D(
                [],
                [],
                color=magenta,
                ls="--",
                marker="s",
                markerfacecolor="white",
                label="g#: matched gate-cut QPD ports",
            ),
            Line2D(
                [],
                [],
                color=magenta,
                marker="|",
                linestyle="None",
                markersize=11,
                label="w#: wire-cut out / in ports",
            ),
        ]
        fig.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.066, 1 - 1.03 / height),
            ncol=4,
            frameon=False,
            fontsize=10,
            handlelength=2,
        )

        # All source operations are included; this compact overview shows assignment.
        ax = fig.add_subplot(grid[0, :])
        for segment in record["segments"]:
            q = segment["qubit"]
            ax.plot(
                [segment["start"], segment["end"]],
                [q, q],
                color=colors[segment["location"][0]],
                lw=1.8,
                zorder=1,
            )
        for op in record["operations"]:
            x, qs = op["index"], op["qubits"]
            color = magenta if op["cut"] else colors[op["locations"][0][0]]
            if len(qs) == 2:
                ax.plot(
                    [x, x],
                    qs,
                    color=color,
                    lw=0.7,
                    linestyle="--" if op["cut"] else "-",
                    alpha=0.7,
                    zorder=2,
                )
            ax.scatter(
                [x] * len(qs),
                qs,
                s=10 if len(qs) == 2 else 4,
                facecolors="white" if op["cut"] else color,
                edgecolors=color,
                linewidths=0.6,
                zorder=3,
            )
        for cut in cuts:
            ax.plot(
                cut["index"] - 0.47,
                cut["qubit"],
                "|",
                color=magenta,
                markersize=11,
                markeredgewidth=2,
                zorder=5,
            )
            ax.annotate(
                f"w{cut['id']}",
                (cut["index"] - 0.47, cut["qubit"]),
                xytext=(0, 7),
                textcoords="offset points",
                fontsize=7,
                color=magenta,
                ha="center",
                zorder=6,
            )
        ax.set_title(
            "Source circuit assignment overview  ·  detailed local gates below",
            loc="left",
            fontsize=11,
            pad=9,
            color=ink,
        )
        ax.set_yticks(range(n), [f"q{q}" for q in range(n)], fontsize=8)
        ax.set_ylim(n - 0.55, -0.75)
        ax.set_xlim(-1, size)
        ax.set_xlabel("Source operation index (zero based)", fontsize=8)
        ax.tick_params(length=0, labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for fold in range(folds):
            start, stop = fold * columns, min((fold + 1) * columns, size)
            for block in range(fragments):
                ax = fig.add_subplot(
                    grid[1 + fold * groups + block // plot_columns, block % plot_columns]
                )
                color = colors[block]
                segments = sorted(
                    (s for s in record["segments"] if s["location"][0] == block),
                    key=lambda s: s["location"][1],
                )
                for segment in segments:
                    y = segment["location"][1]
                    ax.plot(
                        [start - 0.95, stop - 0.1],
                        [y, y],
                        color="#DDE1E6",
                        lw=0.8,
                        ls=":",
                        zorder=0,
                    )
                    left, right = (
                        max(start - 0.95, segment["start"]),
                        min(stop - 0.1, segment["end"]),
                    )
                    if left < right:
                        ax.plot([left, right], [y, y], color=color, lw=1.2, zorder=1)
                for op in record["operations"]:
                    if not start <= op["index"] < stop:
                        continue
                    if op["cut"]:
                        for owner, row in op["locations"]:
                            if owner == block:
                                box(
                                    ax,
                                    op["index"],
                                    row,
                                    f"g{op['index']}\n{op['name'].upper()}",
                                    magenta,
                                    cut=True,
                                )
                    elif op["locations"][0][0] == block:
                        gate(ax, op, [loc[1] for loc in op["locations"]], color)
                for cut in cuts:
                    if not start <= cut["index"] < stop:
                        continue
                    for key, shift, marker, direction in (
                        ("old", 0.55, ">", "out"),
                        ("new", 0.47, ">", "in"),
                    ):
                        owner, row = cut[key]
                        if owner != block:
                            continue
                        x = cut["index"] - shift
                        ax.plot(x, row, marker, color=magenta, markersize=6, zorder=6)
                        ax.annotate(
                            f"w{cut['id']} {direction}",
                            (x, row),
                            xytext=(0, -14),
                            textcoords="offset points",
                            color=magenta,
                            fontsize=6.5,
                            ha="center",
                            zorder=7,
                            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.1},
                        )
                ax.set_yticks(
                    range(widths[block]),
                    [f"{names[block]}{s['location'][1]} · q{s['qubit']}" for s in segments],
                    fontsize=8,
                )
                ax.set_ylim(max(widths) - 0.35, -0.85)
                ax.set_xlim(start - 1, start + columns - 0.05)
                ax.set_xticks(range(start, stop))
                ax.tick_params(length=0, labelsize=7, pad=4)
                ax.set_title(
                    f"Fragment {names[block]}  |  {widths[block]} static registers  |  "
                    f"operations {start}–{stop - 1}",
                    color=color,
                    loc="left",
                    fontsize=11,
                    pad=9,
                )
                for spine in ax.spines.values():
                    spine.set_visible(False)
        fig.text(
            0.07,
            0.24 / height,
            "Logical fragment templates: g# ports share the source gate index; w# ports "
            "share a wire-cut ID. Local QPD basis choices / branches are not expanded.\n"
            "Row labels give local register · original qubit. Dotted wires are allocated "
            "but inactive; colored wires carry that source segment. Angles are in radians.",
            fontsize=9,
            color=ink,
            va="bottom",
        )
        path = output / f"{record['name']}_partition.png"
        fig.savefig(
            path,
            dpi=dpi,
            facecolor="white",
            metadata={
                "Title": f"{record['name']} optimized logical fragments",
                "Description": f"Source fingerprint: {record['fingerprint']}; saved best_plan.json",
            },
        )
        plt.close(fig)
        print(path, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("export", "render"))
    parser.add_argument("--directory", type=Path, default=Path("validation/sa_benchmarks_v2"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--only", nargs="*")
    parser.add_argument(
        "--output", type=Path, default=Path("validation/sa_benchmarks_v2/partition_png")
    )
    parser.add_argument("--columns", type=int, default=24)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    if args.columns < 1 or args.dpi < 1:
        parser.error("columns and dpi must be positive")
    if args.mode == "export":
        export(args.directory, args.data, args.only)
    else:
        render(args.data, args.output, args.columns, args.dpi)


if __name__ == "__main__":
    main()
