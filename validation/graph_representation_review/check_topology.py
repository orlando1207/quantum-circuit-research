"""Review-only topology checks; not a production partition implementation."""

import hashlib
import json
from itertools import product
from pathlib import Path

import networkx as nx
from qiskit import QuantumCircuit

from qpartition import CircuitProblem, CutPlan, WireCut, compile_cut, temporal_graph


def build(circuit, compact):
    graph = nx.Graph()
    previous = {}
    candidates = []
    internals = {}
    couplings = {}
    dummies = {}
    for q in range(circuit.num_qubits):
        node = f"input:{q}"
        graph.add_node(node, q=q)
        previous[q] = node
    arities = []
    for gi, item in enumerate(circuit.data):
        qubits = [circuit.find_bit(q).index for q in item.qubits]
        arities.append(len(qubits))
        internals[gi] = []
        dummies[gi] = []
        centers = []
        for operand, q in enumerate(qubits):
            pin, pout = f"g{gi}:in:{operand}", f"g{gi}:out:{operand}"
            graph.add_node(pin, q=q)
            graph.add_node(pout, q=q)
            edge = (previous[q], pin)
            graph.add_edge(*edge, kind="wire")
            if previous[q].startswith("g"):
                candidates.append(edge)
            if compact or len(qubits) == 1:
                graph.add_edge(pin, pout, kind="continuity")
                internals[gi].append((pin, pout))
                centers.append(pout)
            else:
                center = f"dummy:{gi}:{operand}"
                graph.add_node(center, q=q)
                dummies[gi].append(center)
                for local in ((pin, center), (center, pout)):
                    graph.add_edge(*local, kind="continuity")
                    internals[gi].append(local)
                centers.append(center)
            previous[q] = pout
        if len(centers) == 2:
            coupling = tuple(centers)
            graph.add_edge(*coupling, kind="coupling")
            internals[gi].append(coupling)
            couplings[gi] = coupling
    for q in range(circuit.num_qubits):
        node = f"output:{q}"
        graph.add_node(node, q=q)
        graph.add_edge(previous[q], node, kind="wire")
    return graph, candidates, internals, couplings, dummies, arities


def transform(data, modes, breaks):
    original, candidates, internals, couplings, dummies, arities = data
    graph = original.copy()
    for edge, cut in zip(candidates, breaks):
        if cut:
            graph.remove_edge(*edge)
    for gi, mode in enumerate(modes):
        if mode == "gate":
            graph.remove_edge(*couplings[gi])
        elif mode == "excision":
            graph.remove_edges_from(internals[gi])
            graph.remove_nodes_from(dummies[gi])
    return graph


def signature(graph):
    components = list(nx.connected_components(graph))
    ownership = {node: fi for fi, nodes in enumerate(components) for node in nodes}
    wire_graph = graph.copy()
    wire_graph.remove_edges_from(
        [(u, v) for u, v, info in graph.edges(data=True) if info["kind"] == "coupling"]
    )
    widths = [0] * len(components)
    for segment in nx.connected_components(wire_graph):
        widths[ownership[next(iter(segment))]] += 1
    return sorted(
        (tuple(sorted(node for node in nodes if not node.startswith("dummy:"))), width)
        for nodes, width in zip(components, widths)
    )


circuits = {}
c = QuantumCircuit(2)
c.h(0)
c.cx(0, 1)
c.t(0)
c.cx(0, 1)
circuits["rejoining_two_qubit_circuit"] = c
c = QuantumCircuit(3)
c.h(0)
c.cx(0, 1)
c.cx(1, 2)
circuits["three_qubit_chain"] = c
c = QuantumCircuit(1)
c.h(0)
c.t(0)
circuits["single_qubit_chain"] = c
c = QuantumCircuit(3)
c.cx(0, 1)
circuits["idle_wire"] = c

results = []
for name, circuit in circuits.items():
    dummy = build(circuit, False)
    compact = build(circuit, True)
    count = 0
    for modes in product(
        *[
            ("retain", "gate", "excision") if arity == 2 else ("retain", "excision")
            for arity in dummy[-1]
        ]
    ):
        for breaks in product((False, True), repeat=len(dummy[1])):
            left = signature(transform(dummy, modes, breaks))
            right = signature(transform(compact, modes, breaks))
            assert left == right, (name, modes, breaks)
            expected = (
                circuit.num_qubits
                + sum(breaks)
                + sum(arity for arity, mode in zip(dummy[-1], modes) if mode == "excision")
            )
            assert sum(width for _, width in right) == expected
            count += 1
    results.append(
        {
            "circuit": name,
            "qubits": circuit.num_qubits,
            "assignments_checked": count,
            "topology_and_width_agree": True,
        }
    )

single = QuantumCircuit(2)
single.cx(0, 1)
data = build(single, True)
truth = {}
for mode in ("retain", "gate", "excision"):
    sig = signature(transform(data, (mode,), ()))
    truth[mode] = {"components": len(sig), "widths": sorted(width for _, width in sig)}
assert truth == {
    "retain": {"components": 1, "widths": [2]},
    "gate": {"components": 2, "widths": [1, 1]},
    "excision": {"components": 4, "widths": [1, 1, 1, 1]},
}

single_problem = CircuitProblem.create(single, "ZZ")
star_graph = temporal_graph(single_problem).to_undirected()
star_graph.remove_node(single_problem.operation_ids[0])
star_components_after_center_deletion = nx.number_connected_components(star_graph)
assert star_components_after_center_deletion == 4
single_compiled = compile_cut(
    single_problem,
    CutPlan.create(single_problem, gate_cuts=(single_problem.operation_ids[0],)),
)
assert sorted(single_compiled.cost.fragment_widths.values()) == [1, 1]

circuit = circuits["rejoining_two_qubit_circuit"]
problem = CircuitProblem.create(circuit, "ZZ")
compiled = compile_cut(
    problem,
    CutPlan.create(problem, wire_cuts=(WireCut(0, problem.operation_ids[2]),)),
)
assert list(compiled.cost.fragment_widths.values()) == [3]
exact_error = abs(compiled.run_exact().expectation - problem.exact_expectation())
assert exact_error < 1e-12

report = {
    "scope": "Topology and static allocation, not a new production compiler or optimality proof",
    "plan_sha256": hashlib.sha256(Path("plan.md").read_bytes()).hexdigest(),
    "circuits": results,
    "total_assignments_checked": sum(row["assignments_checked"] for row in results),
    "single_cx_truth_table": truth,
    "current_star_graph": {
        "components_after_deleting_gate_center": star_components_after_center_deletion,
        "actual_gate_cut_fragment_widths": single_compiled.cost.fragment_widths,
    },
    "rejoining_wire_cut": {
        "original_qubits": 2,
        "fragments": 1,
        "materialized_widths": compiled.cost.fragment_widths,
        "exact_reconstruction_error": exact_error,
    },
}
target = Path("validation/graph_representation_review")
target.mkdir(parents=True, exist_ok=True)
(target / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
