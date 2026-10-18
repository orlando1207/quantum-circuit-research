"""Audit the QFT6 2500 -> 42 comparison; optional source-circuit PNG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def audit(directory):
    from dataclasses import asdict
    from math import prod, sin

    import numpy as np
    from qiskit.quantum_info import Operator

    from qpartition import GroupingConfig, HybridPlan, PartitionGraph, load_problem

    old = Path("validation/grouping_benchmarks_v1")
    problem = load_problem(old / "qft6")
    state = HybridPlan.load(old / "qft6/grouped_multiway/winner/best_plan.json").state
    variants = {
        "compressed_first_3_local": GroupingConfig(
            max_gates=3, anchor_strategy="first", wire_windows=False
        ),
        "compressed_both_3_wire": GroupingConfig(max_gates=3),
        "compressed_first_4_wire": GroupingConfig(anchor_strategy="first"),
        "compressed_both_4_wire": GroupingConfig(),
    }
    records = {}
    for name, config in variants.items():
        graph = PartitionGraph(problem, grouping=config)
        evaluation = graph.evaluate(state, max_qubits=4, partition_mode="multiway")
        records[name] = {
            "config": asdict(config),
            "overhead": evaluation.sampling_overhead,
            "widths": evaluation.block_widths,
            "groups": [
                {
                    **asdict(selected),
                    "operations": graph.grouping.by_id[selected.group_id].operations,
                    "anchor": graph.grouping.by_id[selected.group_id].anchor,
                }
                for selected in evaluation.groups
            ],
        }
    angles = np.pi / 2.0 ** np.arange(1, 5)
    overlap = float(prod(np.cos(angles / 2)))
    choi = 1 + 2 * np.sqrt(1 - overlap**2)
    restriction = 1 + 2 * sin(float(sum(angles)) / 2)
    target = Operator(problem.circuit).data
    order = [0, 5, 1, 2, 3, 4]
    indices = [sum(((k >> j) & 1) << q for j, q in enumerate(order)) for k in range(64)]
    reshuffled = (
        target[np.ix_(indices, indices)]
        .reshape(16, 4, 16, 4)
        .transpose(1, 3, 0, 2)
        .reshape(16, 256)
    )
    schmidt = np.linalg.svd(reshuffled, compute_uv=False) / 8
    schmidt = schmidt[schmidt > 1e-12]
    report = {
        "source_fingerprint": problem.circuit_fingerprint,
        "partition": [[0, 5], [1, 2, 3, 4]],
        "paper_no_group": 2500,
        "paper_group": 42,
        "authors_group_manifest_available": False,
        "independent_cost": prod(1 + 2 * sin(a / 2) for a in angles) ** 4,
        "angles_each_star": angles.tolist(),
        "star_choi_overlap": overlap,
        "star_choi_gamma_lower_bound": float(choi),
        "two_star_choi_product_overhead": float(choi**4),
        "star_extremal_subspace_gamma_lower_bound": restriction,
        "independently_decomposed_two_stars_overhead_lower_bound": restriction**4,
        "whole_partition_choi_schmidt_coefficients": schmidt.tolist(),
        "whole_partition_choi_overhead_lower_bound": float((2 * sum(schmidt) ** 2 - 1) ** 2),
        "interpretation": "41.6666 rounds to 42, a numerical signature only. The authors do not "
        "specify their CP group cost. Each star Choi bound is strictly below "
        "a valid extremal-subspace lower bound, so it is not an achievable "
        "star cost certificate. This does not rule out overhead 42 for a "
        "different joint decomposition of the entire circuit.",
        "variants_same_edge_mask": records,
        "operations": [
            {
                "id": op,
                "name": item.operation.name,
                "qubits": [problem.circuit.find_bit(q).index for q in item.qubits],
                "angle": float(item.operation.params[0]) if item.operation.params else None,
            }
            for op, item in zip(problem.operation_ids, problem.circuit.data, strict=True)
        ],
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"variants_same_edge_mask", "operations"}},
            indent=2,
        )
    )
    print({k: v["overhead"] for k, v in records.items()})


def map_lp_audit(directory):
    """Numerical LP certificate restricted to the existing local signed-map pool."""
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from scipy.optimize import linprog

    from qpartition import HybridPlan, PartitionGraph, load_problem

    folder = directory.parent / "qft6"
    graph = PartitionGraph(load_problem(folder), grouping=True)
    plan = HybridPlan.load(folder / "enhanced/winner/best_plan.json")
    selected = plan.validate(graph).groups[0]
    candidate = graph.grouping.by_id[selected.group_id]
    basis = graph.grouping.basis(selected)
    sides = tuple(
        tuple(q for i, q in enumerate(candidate.qubits) if ((selected.pattern >> i) & 1) == side)
        for side in (0, 1)
    )

    def schur(steps, n):
        dimension = 2**n
        kraus = [(1.0, np.eye(dimension, dtype=complex))]
        for step in steps:
            if step.operation.name == "measure":
                bits = (np.arange(dimension) >> step.qubits[0]) & 1
                projectors = [np.diag(bits == value) for value in (0, 1)]
                kraus = [
                    (weight * (-1 if value and step.scored else 1), projectors[value] @ k)
                    for weight, k in kraus
                    for value in (0, 1)
                ]
            else:
                qc = QuantumCircuit(n)
                qc.append(step.operation, step.qubits)
                matrix = Operator(qc).data
                kraus = [(w, matrix @ k) for w, k in kraus]
        for _, k in kraus:
            assert np.max(abs(k - np.diag(np.diag(k)))) < 1e-12
        return sum(w * np.outer(np.diag(k), np.diag(k).conj()) for w, k in kraus)

    local = [{}, {}]
    columns = []
    for maps in basis.maps:
        fs = [schur(maps[i], len(sides[i])) for i in (0, 1)]
        for i, f in enumerate(fs):
            rounded = np.round(f, 12)
            rounded[abs(rounded) < 1e-12] = 0
            local[i][rounded.tobytes()] = f
        columns.append(np.kron(fs[1], fs[0]).reshape(-1))
    original = np.array(columns).T
    order = sides[0] + sides[1]
    indices = [
        sum(((bits >> i) & 1) << candidate.qubits.index(q) for i, q in enumerate(order))
        for bits in range(2 ** len(order))
    ]
    target_unitary = graph.grouping._target(candidate)[np.ix_(indices, indices)]
    assert np.max(abs(target_unitary - np.diag(np.diag(target_unitary)))) < 1e-12
    vector = np.diag(target_unitary)
    target = np.outer(vector, vector.conj()).reshape(-1)
    assert np.max(abs(original @ np.array(basis.coefficients) - target)) < 1e-11
    pools = {
        "original_product_maps": original,
        "all_products_of_existing_local_maps": np.array(
            [np.kron(b, a).reshape(-1) for a in local[0].values() for b in local[1].values()]
        ).T,
    }
    checks = {}
    for name, complex_matrix in pools.items():
        matrix = np.concatenate((complex_matrix.real, complex_matrix.imag))
        rhs = np.concatenate((target.real, target.imag))
        count = matrix.shape[1]
        solution = linprog(
            np.ones(2 * count),
            A_eq=np.concatenate((matrix, -matrix), axis=1),
            b_eq=rhs,
            bounds=(0, None),
            method="highs",
            options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9},
        )
        assert solution.success, solution.message
        weights = solution.x[:count] - solution.x[count:]
        dual = solution.eqlin.marginals
        dual /= max(1.0, float(np.max(abs(matrix.T @ dual))))
        lower = float(dual @ rhs)
        upper = float(sum(abs(weights)))
        residual = float(np.max(abs(matrix @ weights - rhs)))
        assert residual < 1e-8 and upper - lower < 1e-7
        checks[name] = {
            "product_maps": count,
            "coefficient_norm": upper,
            "dual_lower_bound_in_this_finite_pool": lower,
            "constraint_residual": residual,
            "primal_dual_gap": upper - lower,
        }
    value = {
        "status": "passed",
        "group_id": selected.group_id,
        "source_fingerprint": graph.problem.circuit_fingerprint,
        "constructed_norm": basis.gamma,
        "local_map_counts": [len(x) for x in local],
        "checks": checks,
        "scope": "Numerical finite-pool LP optimum only. All local Kraus matrices "
        "were verified diagonal before using Schur-channel constraints. This does not optimize "
        "over arbitrary LO instruments or over larger joint source blocks.",
    }
    (directory / "map_pool_lp.json").write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value, indent=2))


def render(directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    report = json.loads((directory / "audit.json").read_text())
    fig, ax = plt.subplots(figsize=(18, 6))
    a, b = "#1167b1", "#ce4e22"
    first = {f"op{i:06d}" for i in (1, 2, 3, 4)}
    last = {f"op{i:06d}" for i in (10, 14, 17, 19)}
    for q in range(6):
        y = 5 - q
        bg = "#e8f1fa" if q in (0, 5) else "#f2f3f4"
        ax.add_patch(Rectangle((-0.8, y - 0.35), 25.2, 0.7, facecolor=bg, edgecolor="none"))
        ax.plot([-0.5, 24], [y, y], color="#777", lw=1, zorder=1)
        ax.text(-1, y, f"q{q}  " + ("A" if q in (0, 5) else "B"), ha="right", va="center")
    for x, op in enumerate(report["operations"]):
        ys = [5 - q for q in op["qubits"]]
        color = a if op["id"] in first else b if op["id"] in last else "#777"
        if op["name"] == "h":
            ax.text(
                x,
                ys[0],
                "H",
                ha="center",
                va="center",
                bbox=dict(boxstyle="square,pad=.2", fc="white", ec="#444"),
                zorder=4,
            )
        elif op["name"] == "swap":
            ax.plot([x, x], ys, color=color, lw=1.5)
            ax.scatter([x, x], ys, marker="x", color=color, s=45, zorder=4)
        else:
            ax.plot([x, x], ys, color=color, lw=2 if op["id"] in first | last else 1, zorder=2)
            ax.scatter([x, x], ys, color=color, s=24, zorder=3)
            ax.text(x, max(ys) + 0.12, f"{op['angle']:.3f}", ha="center", fontsize=7, color=color)
    ax.annotate(
        "Group 1: four CP gates collected at first gate",
        xy=(1, -0.1),
        xytext=(0, -1.05),
        color=a,
        arrowprops=dict(arrowstyle="->", color=a),
        ha="left",
    )
    ax.annotate(
        "Group 2: four CP gates collected at last gate",
        xy=(19, 5.1),
        xytext=(10, 6.2),
        color=b,
        arrowprops=dict(arrowstyle="->", color=b),
        ha="left",
    )
    ax.set_title(
        "QFT6, M = 4: eight cut CP gates; physical fragment widths (2, 4)\n"
        "Independent overhead 2521.75; implemented joint maps 440.17; "
        "paper target 42 remains unverified",
        pad=20,
    )
    ax.set_xlim(-2, 25)
    ax.set_ylim(-1.5, 6.7)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(directory / "qft6_grouping.png", dpi=170)
    plt.close(fig)
    print("Saved", directory / "qft6_grouping.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("validation/grouping_benchmarks_v2/qft6_audit")
    )
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--map-lp", action="store_true")
    args = parser.parse_args()
    if args.render_only:
        render(args.output)
    else:
        audit(args.output)

    if args.map_lp:
        map_lp_audit(args.output)
