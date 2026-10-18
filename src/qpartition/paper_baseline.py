"""Auditable reconstruction of arXiv:2608.05287v1, not the authors' source code.

The paper omits initialization sizes, weighted-KL mechanics, and Table-I manual
group manifests. Every supplied choice is explicit in PaperConfig. Grouped
results are cut-placement cost models, not executable grouped decompositions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, log
from time import perf_counter
from types import SimpleNamespace

import networkx as nx
import numpy as np

from .templates import TemplateRegistry


@dataclass(frozen=True)
class GateGroup:
    name: str
    kind: str
    operations: tuple[str, ...]
    provenance: str


@dataclass(frozen=True)
class PaperConfig:
    max_qubits: int
    seed: int = 0
    max_iter: int = 10
    spatial_sizes: str = "balanced"
    capacity_model: str = "physical"
    group_gain: str = "objective"
    zero_weight_moves: bool = True


class GroupCost:
    def __init__(self, vertices, costs):
        self.vertices = tuple(vertices)
        self.costs = np.asarray(costs, dtype=float)

    def state(self, labels):
        return sum(int(labels[v]) << i for i, v in enumerate(self.vertices))

    def masks(self, size):
        masks = np.zeros(size, dtype=np.int64)
        for i, vertex in enumerate(self.vertices):
            masks[vertex] = 1 << i
        return masks


class PaperTopology:
    """Literal input/output and gate-incidence graph from Section V-B."""

    schema = "paper-topology-reconstruction-v1"

    def __init__(self, problem, groups=()):
        problem.validate_snapshot()
        self.problem = problem
        self.registry = TemplateRegistry()
        self.groups = tuple(groups)
        n = problem.circuit.num_qubits
        self.nodes = [{"qubit": q, "kind": kind} for kind in ("input", "output") for q in range(n)]
        self.inputs = tuple(range(n))
        self.outputs = tuple(range(n, 2 * n))
        self.gates, self.wires = {}, []
        current = list(self.inputs)
        for op_id, item in zip(problem.operation_ids, problem.circuit.data, strict=True):
            if item.operation.name == "barrier":
                continue
            if item.operation.num_qubits not in (1, 2):
                raise ValueError("Normalize input to one-/two-qubit unitary gates.")
            if item.operation.num_qubits == 1:
                continue
            qubits = tuple(problem.circuit.find_bit(q).index for q in item.qubits)
            vertices = []
            for q in qubits:
                vertex = len(self.nodes)
                self.nodes.append({"qubit": q, "kind": "gate", "operation_id": op_id})
                self.wires.append((current[q], vertex, q, op_id))
                current[q] = vertex
                vertices.append(vertex)
            template, _ = self.registry.for_gate(item.operation)
            if template is None:
                raise ValueError("Baseline requires a supported decomposition for every gate.")
            self.gates[op_id] = {
                "ports": tuple(vertices),
                "qubits": qubits,
                "name": item.operation.name,
                "gamma": template.gamma,
            }
        self.wires.extend((current[q], self.outputs[q], q, None) for q in range(n))
        self.size = len(self.nodes)
        self.starts = np.zeros(self.size, dtype=int)
        self.starts[list(self.inputs)] = 1
        self.endpoint_sizes = np.zeros(self.size, dtype=int)
        self.endpoint_sizes[: 2 * n] = 1
        self.wire_adjacency = np.zeros((self.size, self.size), dtype=int)
        self.incident_wires = [[] for _ in self.nodes]
        for i, (a, b, _, _) in enumerate(self.wires):
            self.wire_adjacency[a, b] = self.wire_adjacency[b, a] = 1
            self.incident_wires[a].append(i)
            self.incident_wires[b].append(i)
        self.group_costs = self._build_groups()
        grouped = {op for group in self.groups for op in group.operations}
        self.matrix = np.zeros((self.size, self.size))
        for a, b, _, _ in self.wires:
            self.matrix[a, b] = self.matrix[b, a] = log(4)
        for op_id, gate in self.gates.items():
            if op_id not in grouped:
                a, b = gate["ports"]
                self.matrix[a, b] = self.matrix[b, a] = log(gate["gamma"])

    def _build_groups(self):
        costs, occupied = [], set()
        indices = {op: i for i, op in enumerate(self.problem.operation_ids)}
        for group in self.groups:
            if not group.provenance or len(group.operations) < 2:
                raise ValueError("Groups require at least two gates and explicit provenance.")
            if occupied.intersection(group.operations):
                raise ValueError("Overlapping group manifests would double count gate costs.")
            occupied.update(group.operations)
            gates = [self.gates[op] for op in group.operations]
            if any(g["name"] != "cx" for g in gates):
                raise ValueError("Only the paper's explicit CX grouping rules are implemented.")
            if group.kind not in ("parallel", "cascade_control", "cascade_target"):
                raise ValueError("Unsupported group: supply a verified rule rather than guessing.")
            if group.kind == "parallel":
                operands = [q for g in gates for q in g["qubits"]]
                if len(set(operands)) != len(operands):
                    raise ValueError("Parallel groups must have disjoint qubit supports.")
                shared = None
            else:
                side = int(group.kind == "cascade_target")
                shared = gates[0]["qubits"][side]
                if any(g["qubits"][side] != shared for g in gates):
                    raise ValueError("Cascade gates must have the specified common operand.")
                others = [g["qubits"][1 - side] for g in gates]
                if len(set(others)) != len(others):
                    raise ValueError("Repeated operands require a separate composed-channel rule.")
            selected = set(group.operations)
            first, last = min(indices[op] for op in selected), max(indices[op] for op in selected)
            support = {q for g in gates for q in g["qubits"]}
            for op_id, item in zip(
                self.problem.operation_ids[first : last + 1],
                self.problem.circuit.data[first : last + 1],
                strict=True,
            ):
                if op_id in selected or item.operation.name == "barrier":
                    continue
                qs = {self.problem.circuit.find_bit(q).index for q in item.qubits}
                if (len(qs) > 1 and qs & support) or (shared is not None and shared in qs):
                    raise ValueError("Intervening operations invalidate this canonical group rule.")
            vertices = tuple(v for g in gates for v in g["ports"])
            if len(vertices) > 16:
                raise ValueError("Explicit group-pattern tables are bounded to 16 incidence bits.")
            values = []
            for pattern in range(1 << len(vertices)):
                bits = [(pattern >> i) & 1 for i in range(len(vertices))]
                cut_count = sum(bits[2 * i] != bits[2 * i + 1] for i in range(len(gates)))
                if group.kind == "parallel":
                    gamma = 2 ** (cut_count + 1) - 1
                else:
                    side = int(group.kind == "cascade_target")
                    continuous = len({bits[2 * i + side] for i in range(len(gates))}) == 1
                    gamma = (3 if cut_count else 1) if continuous else 3**cut_count
                values.append(log(gamma))
            costs.append(GroupCost(vertices, values))
        return tuple(costs)

    def widths(self, labels):
        result = np.bincount(np.asarray(labels)[list(self.inputs)], minlength=2)
        for a, b, _, _ in self.wires:
            if labels[a] != labels[b]:
                result[labels[b]] += 1
        return result

    def flip_width_deltas(self, labels):
        result = np.zeros((self.size, 2), dtype=int)
        for v in range(self.size):
            if self.starts[v]:
                result[v, labels[v]] -= 1
                result[v, 1 - labels[v]] += 1
            for i in self.incident_wires[v]:
                a, b, _, _ = self.wires[i]
                if labels[a] != labels[b]:
                    result[v, labels[b]] -= 1
                xa, xb = int(labels[a]) ^ int(a == v), int(labels[b]) ^ int(b == v)
                if xa != xb:
                    result[v, xb] += 1
        return result

    def cost(self, labels):
        labels = np.asarray(labels)
        different = labels[:, None] != labels[None, :]
        base = float(np.sum(self.matrix * different) / 2)
        return base + sum(group.costs[group.state(labels)] for group in self.group_costs)

    def spatial_labels(self, colors):
        return np.array([colors[node["qubit"]] for node in self.nodes], dtype=int)

    def materialize(self, labels):
        """Independent SDK validation of individual cuts; GG remains cost-only."""
        from qiskit_addon_cutting import cut_gates, cut_wires, partition_problem
        from qiskit_addon_cutting.instructions import CutWire

        boundaries, trailing = {}, []
        for a, b, q, before in self.wires:
            if labels[a] != labels[b]:
                (trailing if before is None else boundaries.setdefault(before, [])).append(q)
        marked = self.problem.circuit.copy_empty_like()
        gate_indices = []
        for op_id, item in zip(self.problem.operation_ids, self.problem.circuit.data, strict=True):
            for q in boundaries.get(op_id, []):
                marked.append(CutWire(), [q])
            gate = self.gates.get(op_id)
            if gate and labels[gate["ports"][0]] != labels[gate["ports"][1]]:
                gate_indices.append(len(marked.data))
            marked.append(item.operation, item.qubits, item.clbits)
        for q in trailing:
            marked.append(CutWire(), [q])
        marked, _ = cut_gates(marked, gate_indices)
        expanded = cut_wires(marked)
        partitioned = partition_problem(expanded)
        from math import prod

        individual_overhead = float(prod(b.kappa for b in partitioned.bases) ** 2)
        widths = [c.num_qubits for c in partitioned.subcircuits.values()]
        return marked, {
            "sdk_component_widths": widths,
            "sdk_individual_sampling_overhead": individual_overhead,
            "labeled_block_widths": self.widths(labels).tolist(),
            "grouped_physical_compilation": False if self.groups else None,
        }


def _kl_temporal(graph, initial, config):
    """KL pair passes; width and true objective are checked explicitly."""
    labels = np.asarray(initial, dtype=int).copy()
    rng = np.random.default_rng(config.seed)
    size = graph.size
    matrix = graph.matrix
    group_masks = [g.masks(size) for g in graph.group_costs]
    initial_cost = graph.cost(labels)
    trace = []
    pair_evaluations = 0
    for iteration in range(config.max_iter):
        before = labels.copy()
        before_cost = graph.cost(before)
        best_labels, best_cost = before.copy(), before_cost
        locked = np.zeros(size, dtype=bool)
        widths = graph.widths(labels)
        for _ in range(size):
            a = rng.permutation(np.flatnonzero((labels == 0) & ~locked))
            b = rng.permutation(np.flatnonzero((labels == 1) & ~locked))
            if not len(a) or not len(b):
                break
            different = labels[:, None] != labels[None, :]
            d = np.sum(np.where(different, matrix, -matrix), axis=1)
            gains = d[a, None] + d[b][None, :] - 2 * matrix[np.ix_(a, b)]
            for group, masks in zip(graph.group_costs, group_masks, strict=True):
                state = group.state(labels)
                ma, mb = masks[a, None], masks[b][None, :]
                swapped = group.costs[state ^ ma ^ mb]
                if config.group_gain == "objective":
                    gains += group.costs[state] - swapped
                else:
                    singles = (
                        2 * group.costs[state] - group.costs[state ^ ma] - group.costs[state ^ mb]
                    )
                    gains += singles + np.where(
                        (ma != 0) & (mb != 0), swapped - group.costs[state], 0
                    )
            if config.capacity_model == "physical":
                delta = graph.flip_width_deltas(labels)
                correction = graph.wire_adjacency[np.ix_(a, b)]
                first = widths[0] + delta[a, 0, None] + delta[b, 0][None, :] + correction
                second = widths[1] + delta[a, 1, None] + delta[b, 1][None, :] + correction
                feasible = (first <= config.max_qubits) & (second <= config.max_qubits)
            else:
                load = int(np.sum(graph.endpoint_sizes[labels == 0]))
                first = load - graph.endpoint_sizes[a, None] + graph.endpoint_sizes[b][None, :]
                second = int(np.sum(graph.endpoint_sizes)) - first
                feasible = (first <= 2 * config.max_qubits) & (second <= 2 * config.max_qubits)
            pair_evaluations += gains.size
            gains = np.where(feasible, gains, -np.inf)
            ia, ib = np.unravel_index(np.argmax(gains), gains.shape)
            pair_gain = gains[ia, ib]
            single = None
            single_gain = -np.inf
            if config.zero_weight_moves:
                eligible = np.flatnonzero(~locked & (graph.endpoint_sizes == 0))
                if len(eligible):
                    single_gains = d[eligible].copy()
                    for group, masks in zip(graph.group_costs, group_masks, strict=True):
                        state = group.state(labels)
                        single_gains += group.costs[state] - group.costs[state ^ masks[eligible]]
                    if config.capacity_model == "physical":
                        delta = graph.flip_width_deltas(labels)
                        valid = np.all(widths + delta[eligible] <= config.max_qubits, axis=1)
                        single_gains = np.where(valid, single_gains, -np.inf)
                    best_single = int(np.argmax(single_gains))
                    single_gain = single_gains[best_single]
                    single = int(eligible[best_single])
            if np.isfinite(single_gain) and single_gain > pair_gain:
                labels[single] ^= 1
                locked[single] = True
            elif np.isfinite(pair_gain):
                left, right = int(a[ia]), int(b[ib])
                labels[left], labels[right] = 1, 0
                locked[left] = locked[right] = True
            else:
                break
            widths = graph.widths(labels)
            cost = graph.cost(labels)
            if cost < best_cost - 1e-12:
                best_labels, best_cost = labels.copy(), cost
        labels = best_labels
        trace.append(
            {
                "pass": iteration + 1,
                "log_gamma": best_cost,
                "physical_widths": graph.widths(labels).tolist(),
            }
        )
        if best_cost >= before_cost - 1e-12:
            break
    assert graph.cost(labels) <= initial_cost + 1e-10
    return labels, trace, pair_evaluations


def _grouped_spatial_view(topology, interaction):
    """Explicit count-unit interpretation of the unspecified grouped spatial KL."""
    n = topology.problem.circuit.num_qubits
    matrix = nx.to_numpy_array(interaction, nodelist=range(n), weight="weight")
    grouped = {op for group in topology.groups for op in group.operations}
    for op in grouped:
        a, b = topology.gates[op]["qubits"]
        matrix[a, b] -= 1
        matrix[b, a] -= 1
    costs = []
    for group in topology.group_costs:
        qubits = sorted({topology.nodes[v]["qubit"] for v in group.vertices})
        values = []
        for pattern in range(1 << len(qubits)):
            colors = [0] * n
            for i, q in enumerate(qubits):
                colors[q] = (pattern >> i) & 1
            labels = topology.spatial_labels(colors)
            values.append(group.costs[group.state(labels)] / log(3))
        costs.append(GroupCost(qubits, values))

    def widths(labels):
        return np.bincount(labels, minlength=2)

    def deltas(labels):
        result = np.zeros((n, 2), dtype=int)
        for i in range(n):
            result[i, labels[i]] = -1
            result[i, 1 - labels[i]] = 1
        return result

    def cost(labels):
        labels = np.asarray(labels)
        return float(np.sum(matrix * (labels[:, None] != labels[None, :])) / 2) + sum(
            group.costs[group.state(labels)] for group in costs
        )

    return SimpleNamespace(
        size=n,
        matrix=matrix,
        group_costs=costs,
        endpoint_sizes=np.ones(n, dtype=int),
        wire_adjacency=np.zeros((n, n), dtype=int),
        widths=widths,
        flip_width_deltas=deltas,
        cost=cost,
    )


def run_paper_reconstruction(problem, config: PaperConfig, groups=()):
    """Run stated two-stage architecture, recording all reconstruction choices.

    GG uses explicit manifests and a documented count-unit spatial surrogate;
    the paper does not specify those units or provide the original manifests.
    """
    if config.max_qubits < 1 or config.max_iter < 1:
        raise ValueError("Capacity and max_iter must be positive.")
    if config.spatial_sizes not in ("balanced", "capacity"):
        raise ValueError("Unknown spatial initialization policy.")
    if config.capacity_model not in ("physical", "endpoint_surrogate"):
        raise ValueError("Unknown capacity model.")
    if config.group_gain not in ("objective", "literal_eq3"):
        raise ValueError("Unknown grouped gain policy.")
    n = problem.circuit.num_qubits
    if n > 2 * config.max_qubits:
        return None, {"status": "proven-infeasible-two-way", "config": asdict(config)}
    started = perf_counter()
    topology = PaperTopology(problem, groups)
    interaction = nx.Graph()
    interaction.add_nodes_from(range(n))
    for gate in topology.gates.values():
        a, b = gate["qubits"]
        interaction.add_edge(a, b, weight=interaction.get_edge_data(a, b, {}).get("weight", 0) + 1)
    rng = np.random.default_rng(config.seed)
    count = (
        n // 2
        if config.spatial_sizes == "balanced"
        else int(rng.integers(max(1, n - config.max_qubits), min(n - 1, config.max_qubits) + 1))
    )
    order = rng.permutation(n).tolist()
    partition = (set(order[:count]), set(order[count:]))
    a, b = nx.algorithms.community.kernighan_lin_bisection(
        interaction,
        partition=partition,
        seed=config.seed,
        max_iter=config.max_iter,
        weight="weight",
    )
    colors = tuple(int(q in b) for q in range(n))
    if groups:
        view = _grouped_spatial_view(topology, interaction)
        colors, _, _ = _kl_temporal(view, np.array(colors), config)
        colors = tuple(int(c) for c in colors)
    initial = topology.spatial_labels(colors)
    stage_one = topology.cost(initial)
    labels, trace, pairs = _kl_temporal(topology, initial, config)
    widths = topology.widths(labels)
    gate_cuts = [
        op
        for op, gate in topology.gates.items()
        if labels[gate["ports"][0]] != labels[gate["ports"][1]]
    ]
    wire_cuts = [
        {"qubit": q, "before_operation_id": before}
        for left, right, q, before in topology.wires
        if labels[left] != labels[right]
    ]
    assert int(sum(widths)) == n + len(wire_cuts)
    return topology, {
        "status": "feasible" if max(widths) <= config.max_qubits else "physically-infeasible",
        "method": "paper two-stage KL reconstruction",
        "fidelity": "paper architecture with explicit missing-detail choices; not author code",
        "grouped_status": "explicit manifest in both stages; not authors' Table-I manifests"
        if groups
        else "no grouping",
        "graph_schema": topology.schema,
        "source_fingerprint": problem.circuit_fingerprint,
        "config": asdict(config),
        "labels": labels.tolist(),
        "spatial_qubit_labels": colors,
        "stage_one_sampling_overhead": exp(2 * stage_one),
        "stage_one_cut_gate_count": sum(
            colors[g["qubits"][0]] != colors[g["qubits"][1]] for g in topology.gates.values()
        ),
        "sampling_overhead": exp(2 * topology.cost(labels)),
        "widths": [int(w) for w in widths if w],
        "fragments": int(sum(widths > 0)),
        "gate_cuts": gate_cuts,
        "wire_cuts": wire_cuts,
        "groups": [asdict(g) for g in groups],
        "trace": trace,
        "pair_evaluations": pairs,
        "elapsed_seconds": perf_counter() - started,
        "assumptions": [
            "Spatial stage uses gate counts, as Section V-A states.",
            "Initialization-size policy and KL max_iter are not pinned in the paper.",
            "Temporal KL uses locked pair swaps and optional zero-size single moves, "
            "with best actual-cost prefixes.",
            "Physical capacity guard is additional to underspecified endpoint-size handling.",
            "Grouped joint QPD execution is not implemented by this placement baseline.",
            "Grouped spatial costs use ln(gamma)/ln(3) in gate-count units; temporal "
            "costs use ln(gamma). This convention is not pinned in the paper.",
        ],
    }
