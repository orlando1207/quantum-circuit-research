"""Contracted temporal incidence graph with optional cut-realization factors."""

from __future__ import annotations

from dataclasses import replace

import networkx as nx

from .circuits import CircuitProblem
from .graph_common import ActionVariable, GraphEvaluation, _root, _union, pack_two
from .templates import TemplateRegistry


class PartitionGraph:
    """Two vertices per two-qubit gate; fixed local continuations are contracted.

    One-qubit chains stay in the execution IR. Equivalent wire boundaries are
    represented canonically immediately before the next two-qubit operation.
    Input/output boundaries alias their first/last incidence, preserving registers.
    """

    graph_schema = "temporal-incidence-v2"

    def __init__(
        self,
        problem: CircuitProblem,
        registry: TemplateRegistry | None = None,
        *,
        grouping=None,
        excision: bool = False,
        excision_gamma: float | None = None,
    ):
        problem.validate_snapshot()
        self.problem = problem
        from .grouping import GroupingConfig, GroupLibrary

        if not isinstance(excision, bool):
            raise ValueError("excision must be boolean.")
        config = GroupingConfig() if grouping is True else grouping
        config = config if config is not False else None
        if config is not None and "excision" in config.families:
            excision = True
        if excision:
            config = (
                GroupingConfig(families=("excision",))
                if config is None
                else replace(config, families=tuple(dict.fromkeys((*config.families, "excision"))))
            )
        if excision_gamma is not None and not excision:
            raise ValueError("excision_gamma requires excision=True.")
        self.registry = registry or TemplateRegistry(
            include_excision=excision, excision_gamma=excision_gamma
        )
        if self.registry.include_excision != excision:
            raise ValueError("Registry excision policy must match the graph's excision setting.")
        if registry is not None and excision_gamma is not None:
            if registry.excision_gamma != excision_gamma:
                raise ValueError("Registry and graph excision cost assumptions differ.")
        self.variables: list[ActionVariable] = []
        self.nodes: list[dict] = []
        self.edges: list[tuple[int, int, str, int]] = []
        self.gate_variables: dict[str, int] = {}
        self.wire_variables: dict[tuple[str, int], int] = {}
        self.gate_ports: dict[str, tuple[tuple[int, int], ...]] = {}
        self.incoming_wires: dict[str, tuple[int | None, ...]] = {}
        self.wire_nodes: list[list[int]] = [[] for _ in range(problem.circuit.num_qubits)]
        gate_vertices = {}
        for op_id, item in zip(problem.operation_ids, problem.circuit.data, strict=True):
            if item.operation.name == "barrier":
                continue
            if item.operation.num_qubits not in (1, 2):
                raise ValueError("Normalize the circuit to one-/two-qubit unitary gates.")
            if item.operation.num_qubits != 2:
                continue
            vertices = []
            for operand, qubit in enumerate(item.qubits):
                q = problem.circuit.find_bit(qubit).index
                vertex = len(self.nodes)
                self.nodes.append(
                    {"qubit": q, "kind": "incidence", "operation_id": op_id, "operand": operand}
                )
                if self.wire_nodes[q]:
                    wire = len(self.variables)
                    previous = self.wire_nodes[q][-1]
                    self.variables.append(
                        ActionVariable(
                            "wire", op_id, q, (previous, vertex), (None, self.registry.wire)
                        )
                    )
                    self.wire_variables[op_id, q] = wire
                    self.edges.append((previous, vertex, "wire", wire))
                self.wire_nodes[q].append(vertex)
                vertices.append(vertex)
            cut, _ = self.registry.for_gate(item.operation)
            variable = len(self.variables)
            self.variables.append(
                ActionVariable(
                    "gate", op_id, None, tuple(vertices), (None, cut) if cut else (None,)
                )
            )
            self.gate_variables[op_id] = variable
            self.edges.append((*vertices, "coupling", variable))
            gate_vertices[op_id] = tuple(vertices)
        for q, vertices in enumerate(self.wire_nodes):
            if not vertices:
                vertices.append(len(self.nodes))
                self.nodes.append({"qubit": q, "kind": "local-wire"})
        self.input_nodes = [vertices[0] for vertices in self.wire_nodes]
        self.output_nodes = [vertices[-1] for vertices in self.wire_nodes]
        current = self.input_nodes.copy()
        for op_id, item in zip(problem.operation_ids, problem.circuit.data, strict=True):
            if item.operation.name == "barrier":
                continue
            qs = [problem.circuit.find_bit(q).index for q in item.qubits]
            if item.operation.num_qubits == 2:
                vertices = gate_vertices[op_id]
                self.incoming_wires[op_id] = tuple(self.wire_variables.get((op_id, q)) for q in qs)
                for q, vertex in zip(qs, vertices):
                    current[q] = vertex
            else:
                vertices = tuple(current[q] for q in qs)
                self.incoming_wires[op_id] = (None,)
            self.gate_ports[op_id] = tuple((vertex, vertex) for vertex in vertices)
        incident = [set() for _ in self.nodes]
        for left, right, _, variable in self.edges:
            incident[left].add(variable)
            incident[right].add(variable)
        self.adjacency = [set() for _ in self.variables]
        for neighbors in incident:
            for variable in neighbors:
                self.adjacency[variable].update(neighbors - {variable})
        self.macro_neighborhoods = tuple(
            sorted({tuple(sorted({v} | neighbors)) for v, neighbors in enumerate(self.adjacency)})
        )
        self.cost_upper_bound = sum(
            max(option.log_overhead if option else 0 for option in variable.options)
            for variable in self.variables
        )
        self.grouping = None
        self.grouping_fingerprint = None
        self.group_regions = ()
        if config is not None:
            self.grouping = GroupLibrary(self, config)
            self.grouping_fingerprint = self.grouping.fingerprint
            self.group_regions = self.grouping.regions
            self.graph_schema = "temporal-incidence-grouped-v3"

    def validate_state(self, state: tuple[int, ...]) -> None:
        self.problem.validate_snapshot()
        if len(state) != len(self.variables) or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < len(variable.options)
            for value, variable in zip(state, self.variables)
        ):
            raise ValueError("State must select exactly one available option per action variable.")

    def state_from_labels(self, labels: tuple[int, ...]) -> tuple[int, ...]:
        if len(labels) != len(self.nodes) or any(
            isinstance(label, bool) or not isinstance(label, int) or label < 0 for label in labels
        ):
            raise ValueError("Provide one nonnegative integer label per incidence vertex.")
        state = [0] * len(self.variables)
        for left, right, _, variable in self.edges:
            value = int(labels[left] != labels[right])
            if value >= len(self.variables[variable].options):
                raise ValueError("Vertex labels require an unavailable gate cut.")
            state[variable] = value
        return tuple(state)

    def labels_from_evaluation(self, evaluation: GraphEvaluation) -> tuple[int, ...]:
        return tuple(
            evaluation.component_blocks[evaluation.segment_components[segment]]
            for segment in evaluation.node_segments
        )

    def evaluate(
        self,
        state: tuple[int, ...],
        *,
        max_qubits: int,
        partition_mode: str = "two_blocks",
        max_components: int | None = None,
        validate: bool = True,
        forced_groups: tuple[str, ...] | None = None,
    ) -> GraphEvaluation:
        if validate:
            self.validate_state(state)
        if isinstance(max_qubits, bool) or not isinstance(max_qubits, int) or max_qubits < 1:
            raise ValueError("max_qubits must be a positive integer.")
        if partition_mode not in ("two_blocks", "components", "multiway"):
            raise ValueError("Unknown partition mode.")
        if max_components is not None and (
            isinstance(max_components, bool)
            or not isinstance(max_components, int)
            or max_components < 1
        ):
            raise ValueError("max_components must be a positive integer.")
        components = list(range(len(self.nodes)))
        segments = components.copy()
        for left, right, role, variable in self.edges:
            if state[variable] == 0:
                _union(components, left, right)
                if role == "wire":
                    _union(segments, left, right)
        component_ids, segment_ids = {}, {}
        widths, node_segments, segment_components = [], [], []
        for node in range(len(self.nodes)):
            root = _root(components, node)
            if root not in component_ids:
                component_ids[root] = len(widths)
                widths.append(0)
            component = component_ids[root]
            root = _root(segments, node)
            if root not in segment_ids:
                segment_ids[root] = len(segment_components)
                segment_components.append(component)
                widths[component] += 1
            node_segments.append(segment_ids[root])
        counts = {"wire": 0, "gate": 0}
        cost = 0.0
        for variable, value in zip(self.variables, state):
            if value:
                option = variable.options[value]
                counts[option.kind] += 1
                cost += option.log_overhead
        if len(segment_components) != self.problem.circuit.num_qubits + counts["wire"]:
            raise RuntimeError("Static segment conservation check failed.")
        if partition_mode == "two_blocks":
            penalty, blocks, block_widths = pack_two(tuple(widths), max_qubits)
        else:
            blocks = tuple(range(len(widths)))
            block_widths = tuple(widths)
            penalty = sum(max(0, width - max_qubits) ** 2 for width in widths)
        if max_components is not None:
            penalty += max(0, len(widths) - max_components) ** 2
        independent_cost, groups, complete = cost, (), None
        excision_wires = 0
        if self.grouping is not None and "excision" in self.grouping.config.families:
            counts["excision"] = 0
        if self.grouping is not None:
            node_components = tuple(segment_components[s] for s in node_segments)
            if forced_groups is not None:
                selection = self.grouping.forced_selection(
                    state, forced_groups, tuple(widths), max_qubits, partition_mode, max_components
                )
            else:
                selection = self.grouping.select(
                    state,
                    tuple(widths),
                    node_components,
                    max_qubits,
                    partition_mode,
                    max_components,
                )
            if selection is not None:
                cost = max(0.0, cost - selection.savings)
                widths, blocks, block_widths = (
                    selection.component_widths,
                    selection.component_blocks,
                    selection.block_widths,
                )
                segment_components = [selection.old_to_new[c] for c in segment_components]
                penalty, groups, complete = (
                    selection.width_penalty,
                    selection.groups,
                    selection.complete,
                )
                for selected in groups:
                    candidate = self.grouping.by_id[selected.group_id]
                    if candidate.family == "excision":
                        counts["excision"] = counts.get("excision", 0) + 1
                        counts["gate"] -= state[self.gate_variables[candidate.anchor]]
                        counts["wire"] -= len(candidate.excision_wires)
                        excision_wires += len(candidate.excision_wires)
        return GraphEvaluation(
            state,
            cost,
            tuple(widths),
            block_widths,
            penalty,
            penalty == 0,
            tuple(node_segments),
            tuple(segment_components),
            blocks,
            counts,
            independent_cost if self.grouping is not None else None,
            groups,
            complete,
            excision_wires,
        )

    def networkx_graph(self, state: tuple[int, ...] | None = None) -> nx.Graph:
        state = (0,) * len(self.variables) if state is None else state
        self.validate_state(state)
        graph = nx.Graph()
        graph.add_nodes_from((i, metadata.copy()) for i, metadata in enumerate(self.nodes))
        for left, right, role, variable in self.edges:
            if not state[variable]:
                graph.add_edge(
                    left,
                    right,
                    kind=role,
                    action_variable=variable,
                    weight=self.variables[variable].options[1].log_overhead
                    if len(self.variables[variable].options) > 1
                    else float("inf"),
                )
        return graph
