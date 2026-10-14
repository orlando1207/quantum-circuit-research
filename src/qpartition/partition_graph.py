"""Typed four-port graph, legal action states, and exact static-width evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

import networkx as nx

from .circuits import CircuitProblem
from .templates import ActionTemplate, TemplateRegistry


@dataclass(frozen=True)
class ActionVariable:
    kind: str
    operation_id: str
    qubit: int | None
    ports: tuple[int, ...]
    options: tuple[ActionTemplate | None, ...]


@dataclass(frozen=True)
class GraphEvaluation:
    state: tuple[int, ...]
    log_overhead: float
    component_widths: tuple[int, ...]
    block_widths: tuple[int, ...]
    width_penalty: int
    feasible: bool
    node_segments: tuple[int, ...]
    segment_components: tuple[int, ...]
    component_blocks: tuple[int, ...]
    action_counts: dict[str, int]

    @property
    def sampling_overhead(self) -> float:
        return exp(self.log_overhead) if self.log_overhead < 709 else float("inf")

    @property
    def segment_count(self) -> int:
        return len(self.segment_components)


def _root(parent: list[int], vertex: int) -> int:
    while parent[vertex] != vertex:
        parent[vertex] = parent[parent[vertex]]
        vertex = parent[vertex]
    return vertex


def _union(parent: list[int], left: int, right: int) -> None:
    left, right = _root(parent, left), _root(parent, right)
    if left != right:
        parent[right] = left


def pack_two(widths: tuple[int, ...], limit: int) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    """Exact minimum squared overflow with a subset-sum packing witness.

    Disconnected components share a static circuit only by summing registers;
    this does not schedule or reuse qubits between separate executions.
    """
    total = sum(widths)
    reachable = 1
    previous: dict[int, tuple[int, int]] = {}
    for index, width in enumerate(widths):
        added = (reachable << width) & ~reachable
        while added:
            bit = added & -added
            value = bit.bit_length() - 1
            previous[value] = (value - width, index)
            added -= bit
        reachable |= reachable << width
    best = (float("inf"), float("inf"), 0)
    remaining = reachable
    while remaining:
        bit = remaining & -remaining
        value = bit.bit_length() - 1
        penalty = max(0, value - limit) ** 2 + max(0, total - value - limit) ** 2
        rank = (penalty, abs(total - 2 * value), value)
        if rank < best:
            best = rank
        remaining -= bit
    subtotal = best[2]
    left = set()
    cursor = subtotal
    while cursor:
        cursor, index = previous[cursor]
        left.add(index)
    blocks = tuple(0 if index in left else 1 for index in range(len(widths)))
    raw_widths = (subtotal, total - subtotal)
    if 0 in raw_widths:
        blocks = (0,) * len(widths)
        block_widths = (total,)
    else:
        block_widths = raw_widths
    return int(best[0]), blocks, block_widths


class PartitionGraph:
    def __init__(self, problem: CircuitProblem, registry: TemplateRegistry | None = None):
        problem.validate_snapshot()
        self.problem = problem
        self.registry = registry or TemplateRegistry()
        self.variables: list[ActionVariable] = []
        self.nodes: list[dict] = []
        # role: wire (disabled by wire cut), continuity (disabled by excision),
        # coupling (retained only in retain mode), fixed (always retained).
        self.edges: list[tuple[int, int, str, int]] = []
        self.input_nodes = []
        self.output_nodes = []
        self.gate_ports: dict[str, tuple[tuple[int, int], ...]] = {}
        self.gate_variables: dict[str, int] = {}
        self.incoming_wires: dict[str, tuple[int | None, ...]] = {}
        previous = []
        previous_operation: list[str | None] = []
        for q in range(problem.circuit.num_qubits):
            vertex = self._node(qubit=q, kind="input")
            self.input_nodes.append(vertex)
            previous.append(vertex)
            previous_operation.append(None)
        for op_id, item in zip(problem.operation_ids, problem.circuit.data, strict=True):
            if item.operation.name == "barrier":
                continue
            if item.operation.num_qubits not in (1, 2):
                raise ValueError("Normalize the circuit to one-/two-qubit unitary gates.")
            qubits = [problem.circuit.find_bit(q).index for q in item.qubits]
            ports = []
            wires = []
            for q in qubits:
                pin = self._node(qubit=q, kind="port", direction="in", operation_id=op_id)
                pout = self._node(qubit=q, kind="port", direction="out", operation_id=op_id)
                ports.append((pin, pout))
                if previous_operation[q] is not None:
                    variable = len(self.variables)
                    self.variables.append(
                        ActionVariable(
                            "wire", op_id, q, (previous[q], pin), (None, self.registry.wire)
                        )
                    )
                    self.edges.append((previous[q], pin, "wire", variable))
                    wires.append(variable)
                else:
                    self.edges.append((previous[q], pin, "fixed", -1))
                    wires.append(None)
                previous[q], previous_operation[q] = pout, op_id
            cut, excision = self.registry.for_gate(item.operation)
            options = (None, cut, excision) if cut else (None, excision)
            variable = len(self.variables)
            self.variables.append(
                ActionVariable(
                    "gate", op_id, None, tuple(v for pair in ports for v in pair), options
                )
            )
            self.gate_variables[op_id] = variable
            self.gate_ports[op_id] = tuple(ports)
            self.incoming_wires[op_id] = tuple(wires)
            for pin, pout in ports:
                self.edges.append((pin, pout, "continuity", variable))
            if len(ports) == 2:
                self.edges.append((ports[0][1], ports[1][1], "coupling", variable))
        for q, last in enumerate(previous):
            vertex = self._node(qubit=q, kind="output")
            self.output_nodes.append(vertex)
            self.edges.append((last, vertex, "fixed", -1))
        self.adjacency = [set() for _ in self.variables]
        # Consecutive scopes include gate--wire--gate relationships.
        for op_id, gate_index in self.gate_variables.items():
            for wire in self.incoming_wires[op_id]:
                if wire is not None:
                    self._adjacent(gate_index, wire)
                    predecessor_port = self.variables[wire].ports[0]
                    predecessor = self.nodes[predecessor_port]["operation_id"]
                    self._adjacent(wire, self.gate_variables[predecessor])
        self.cost_upper_bound = sum(
            max(option.log_overhead if option else 0 for option in variable.options)
            for variable in self.variables
        )
        neighborhoods = set()
        for index in self.gate_variables.values():
            neighborhoods.add(tuple(sorted({index} | self.adjacency[index])))
        # Wire boundaries with nearest two-qubit gates form static rewrite motifs.
        # Single-qubit gates remain present; they need not inflate every block proposal.
        for q in range(problem.circuit.num_qubits):
            gate_sequence = [
                op_id
                for op_id, ports in self.gate_ports.items()
                if any(self.nodes[pin]["qubit"] == q for pin, _ in ports)
            ]
            positions = {op: i for i, op in enumerate(gate_sequence)}
            wire_sequence = []
            for index, variable in enumerate(self.variables):
                if variable.kind != "wire" or variable.qubit != q:
                    continue
                wire_sequence.append(index)
                position = positions[variable.operation_id]
                before = next(
                    (
                        self.gate_variables[op]
                        for op in reversed(gate_sequence[:position])
                        if len(self.gate_ports[op]) == 2
                    ),
                    None,
                )
                after = next(
                    (
                        self.gate_variables[op]
                        for op in gate_sequence[position:]
                        if len(self.gate_ports[op]) == 2
                    ),
                    None,
                )
                neighborhoods.add(tuple(sorted(v for v in (index, before, after) if v is not None)))
            neighborhoods.update(
                tuple(sorted(pair)) for pair in zip(wire_sequence, wire_sequence[1:])
            )
        self.macro_neighborhoods = tuple(sorted(neighborhoods))

    def _node(self, **metadata) -> int:
        self.nodes.append(metadata)
        return len(self.nodes) - 1

    def _adjacent(self, left: int, right: int) -> None:
        self.adjacency[left].add(right)
        self.adjacency[right].add(left)

    def validate_state(self, state: tuple[int, ...]) -> None:
        self.problem.validate_snapshot()
        if len(state) != len(self.variables) or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < len(variable.options)
            for value, variable in zip(state, self.variables)
        ):
            raise ValueError("State must select exactly one available option per action variable.")

    def evaluate(
        self,
        state: tuple[int, ...],
        *,
        max_qubits: int,
        partition_mode: str = "two_blocks",
        max_components: int | None = None,
        validate: bool = True,
    ) -> GraphEvaluation:
        if validate:
            self.validate_state(state)
        if isinstance(max_qubits, bool) or not isinstance(max_qubits, int) or max_qubits < 1:
            raise ValueError("max_qubits must be a positive integer.")
        if partition_mode not in ("two_blocks", "components"):
            raise ValueError("partition_mode must be 'two_blocks' or 'components'.")
        if max_components is not None and (
            isinstance(max_components, bool)
            or not isinstance(max_components, int)
            or max_components < 1
        ):
            raise ValueError("max_components must be a positive integer.")
        options = [variable.options[value] for variable, value in zip(self.variables, state)]
        component_parent = list(range(len(self.nodes)))
        segment_parent = component_parent.copy()
        for left, right, role, variable in self.edges:
            option = options[variable] if variable >= 0 else None
            enabled = (
                role == "fixed"
                or (role == "wire" and option is None)
                or (role == "continuity" and (option is None or option.kind != "excision"))
                or (role == "coupling" and option is None)
            )
            if enabled:
                _union(component_parent, left, right)
                if role != "coupling":
                    _union(segment_parent, left, right)
        component_ids, segment_ids = {}, {}
        node_segments = []
        segment_components = []
        widths = []
        for vertex in range(len(self.nodes)):
            component_root = _root(component_parent, vertex)
            if component_root not in component_ids:
                component_ids[component_root] = len(widths)
                widths.append(0)
            component = component_ids[component_root]
            segment_root = _root(segment_parent, vertex)
            if segment_root not in segment_ids:
                segment_ids[segment_root] = len(segment_components)
                segment_components.append(component)
                widths[component] += 1
            node_segments.append(segment_ids[segment_root])
        counts = {"wire": 0, "gate": 0, "excision": 0}
        log_overhead = 0.0
        expected_segments = self.problem.circuit.num_qubits
        for option in options:
            if option:
                counts[option.kind] += 1
                log_overhead += option.log_overhead
                expected_segments += (
                    1 if option.kind == "wire" else option.arity if option.kind == "excision" else 0
                )
        if expected_segments != len(segment_components):
            raise RuntimeError("Static segment conservation check failed.")
        if partition_mode == "two_blocks":
            penalty, component_blocks, block_widths = pack_two(tuple(widths), max_qubits)
        else:
            component_blocks = tuple(range(len(widths)))
            block_widths = tuple(widths)
            penalty = sum(max(0, width - max_qubits) ** 2 for width in widths)
        if max_components is not None:
            penalty += max(0, len(widths) - max_components) ** 2
        return GraphEvaluation(
            state,
            log_overhead,
            tuple(widths),
            block_widths,
            penalty,
            penalty == 0,
            tuple(node_segments),
            tuple(segment_components),
            component_blocks,
            counts,
        )

    def networkx_graph(self, state: tuple[int, ...] | None = None) -> nx.Graph:
        """Inspectable typed graph; search uses the same edges with integer arrays."""
        if state is None:
            state = (0,) * len(self.variables)
        self.validate_state(state)
        graph = nx.Graph()
        graph.add_nodes_from((i, metadata.copy()) for i, metadata in enumerate(self.nodes))
        for left, right, role, index in self.edges:
            option = self.variables[index].options[state[index]] if index >= 0 else None
            enabled = (
                role == "fixed"
                or (role == "wire" and option is None)
                or (role == "continuity" and (option is None or option.kind != "excision"))
                or (role == "coupling" and option is None)
            )
            if enabled:
                graph.add_edge(left, right, kind=role, action_variable=index)
        return graph
