"""Lazy physical compilation and reconstruction for all three cutting actions."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import product
from math import prod
from pathlib import Path
from time import perf_counter

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Measure
from qiskit.circuit.library import HGate, Reset, SdgGate, SGate, XGate

from .execution import execute_exact, execute_shots
from .partition_graph import GraphEvaluation, PartitionGraph


@dataclass(frozen=True)
class HybridPlan:
    circuit_fingerprint: str
    state: tuple[int, ...]
    template_ids: tuple[str | None, ...]
    max_fragment_qubits: int
    partition_mode: str = "two_blocks"
    max_components: int | None = None
    graph_schema: str = "temporal-incidence-v2"
    grouping_fingerprint: str | None = None
    forced_groups: tuple[str, ...] | None = None

    @classmethod
    def create(
        cls,
        graph: PartitionGraph,
        state: tuple[int, ...],
        *,
        max_qubits: int,
        partition_mode: str = "two_blocks",
        max_components: int | None = None,
        forced_groups: tuple[str, ...] | None = None,
    ) -> HybridPlan:
        evaluation = graph.evaluate(
            state,
            max_qubits=max_qubits,
            partition_mode=partition_mode,
            max_components=max_components,
            forced_groups=forced_groups,
        )
        if not evaluation.feasible:
            raise ValueError("Cannot create an executable plan from an infeasible state.")
        return cls(
            graph.problem.circuit_fingerprint,
            state,
            tuple(
                variable.options[value].template_id if value else None
                for variable, value in zip(graph.variables, state)
            ),
            max_qubits,
            partition_mode,
            max_components,
            graph.graph_schema,
            getattr(graph, "grouping_fingerprint", None),
            forced_groups,
        )

    def validate(self, graph: PartitionGraph) -> GraphEvaluation:
        if self.graph_schema != graph.graph_schema:
            raise ValueError("Plan graph schema differs; explicitly migrate historical plans.")
        if self.circuit_fingerprint != graph.problem.circuit_fingerprint:
            raise ValueError("Hybrid plan belongs to a different circuit snapshot.")
        if self.grouping_fingerprint != getattr(graph, "grouping_fingerprint", None):
            raise ValueError("Hybrid plan grouping LUT or selection policy differs.")
        graph.validate_state(self.state)
        expected = tuple(
            variable.options[value].template_id if value else None
            for variable, value in zip(graph.variables, self.state)
        )
        if expected != self.template_ids:
            raise ValueError("Hybrid plan template library does not match the compiled graph.")
        evaluation = graph.evaluate(
            self.state,
            max_qubits=self.max_fragment_qubits,
            partition_mode=self.partition_mode,
            max_components=self.max_components,
            validate=False,
            forced_groups=self.forced_groups,
        )
        if not evaluation.feasible:
            raise ValueError("Hybrid plan violates physical static allocation constraints.")
        return evaluation

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "schema_version": (
                        1
                        if self.graph_schema == "four-port-v1"
                        else 3
                        if self.grouping_fingerprint
                        else 2
                    ),
                    **asdict(self),
                },
                indent=2,
            )
            + "\n"
        )

    @classmethod
    def load(cls, path: str | Path) -> HybridPlan:
        data = json.loads(Path(path).read_text())
        version = data.pop("schema_version", None)
        if (version, data.get("graph_schema")) not in (
            (1, "four-port-v1"),
            (2, "temporal-incidence-v2"),
            (3, "temporal-incidence-grouped-v3"),
        ):
            raise ValueError("Unsupported hybrid plan schema.")
        data["state"] = tuple(data["state"])
        data["template_ids"] = tuple(data["template_ids"])
        if data.get("forced_groups") is not None:
            data["forced_groups"] = tuple(data["forced_groups"])
        return cls(**data)


@dataclass(frozen=True)
class HybridExperiment:
    circuits: tuple[QuantumCircuit, ...]
    score_masks: tuple[int, ...]
    coefficient: float


@dataclass(frozen=True)
class HybridResult:
    expectation: float
    standard_error: float | None
    reconstruction_trials: int
    physical_shots: int
    executed_circuits: int
    gamma: float
    estimator_gamma: float
    elapsed_seconds: float
    mode: str


class CompiledHybrid:
    def __init__(self, graph: PartitionGraph, plan: HybridPlan):
        self.graph, self.plan = graph, plan
        self.evaluation = plan.validate(graph)
        library = getattr(graph, "grouping", None)
        covered = {
            v for group in self.evaluation.groups for v in library.by_id[group.group_id].variables
        }
        self.covered_variables = covered
        self.active_variables = tuple(
            index for index, value in enumerate(plan.state) if value and index not in covered
        )
        templates = [graph.variables[i].options[plan.state[i]] for i in self.active_variables]
        self.group_factors = {}
        self.group_anchors = {}
        self.replaced_operations = set()
        self.local_operation_ports = {}
        source_positions = {op: i for i, op in enumerate(graph.problem.operation_ids)}
        for selected in self.evaluation.groups:
            candidate = library.by_id[selected.group_id]
            basis = library.basis(selected)
            self.group_factors[selected.group_id] = len(templates), selected, basis
            self.group_anchors[candidate.anchor] = selected.group_id
            self.replaced_operations.update((*candidate.operations, *candidate.absorbed_operations))
            templates.append(basis)
            if candidate.family == "excision":
                # An outgoing canonical boundary lies after any intervening
                # one-qubit chain. That chain must act on the prepared output,
                # not on the input register that has just been measured.
                for operand, (pin, pout) in enumerate(candidate.excision_ports):
                    if pin == pout or pout == candidate.pins[operand]:
                        continue
                    wire = next(
                        v
                        for v in candidate.excision_wires
                        if graph.variables[v].ports == (pin, pout)
                    )
                    first = source_positions[candidate.anchor] + 1
                    last = source_positions[graph.variables[wire].operation_id]
                    q = candidate.qubits[operand]
                    for pos in range(first, last):
                        op = graph.problem.operation_ids[pos]
                        item = graph.problem.circuit.data[pos]
                        if (
                            item.operation.num_qubits == 1
                            and graph.problem.circuit.qubits[q] in item.qubits
                        ):
                            self.local_operation_ports[op] = ((pout, pout),)
        self.templates = tuple(templates)
        if any(
            getattr(t, "assumed_gamma", None) is not None
            and not np.isclose(t.assumed_gamma, t.gamma, atol=1e-10, rtol=1e-12)
            for t in self.templates
        ):
            raise ValueError(
                "Assumed excision cost has no matching physical recipe; cost study only."
            )
        self.factor_positions = {
            variable: position for position, variable in enumerate(self.active_variables)
        }
        self.segment_locations = []
        counts = [0] * len(self.evaluation.block_widths)
        for component in self.evaluation.segment_components:
            block = self.evaluation.component_blocks[component]
            self.segment_locations.append((block, counts[block]))
            counts[block] += 1
        if tuple(counts) != self.evaluation.block_widths:
            raise RuntimeError("Compiler and evaluator disagree on allocated segment widths.")
        self.observable_terms = tuple(graph.problem.observable.to_list())
        self.gamma = prod(template.gamma for template in self.templates)
        if library is not None and not np.isclose(
            np.log(self.gamma) * 2, self.evaluation.log_overhead, atol=1e-10, rtol=0
        ):
            raise ValueError("Grouped objective and executable sampling norm disagree.")
        self.estimator_gamma = prod(
            sum(abs(c) for c in template.coefficients) for template in self.templates
        ) * sum(abs(float(weight.real)) for _, weight in self.observable_terms)

    def _location(self, port: int) -> tuple[int, int]:
        return self.segment_locations[self.evaluation.node_segments[port]]

    def experiment(
        self, branches: tuple[int, ...], *, observable_index: int = 0
    ) -> HybridExperiment:
        """Compile one global branch; there is no hidden exponential enumeration."""
        if len(branches) != len(self.templates) or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < len(template.coefficients)
            for value, template in zip(branches, self.templates)
        ):
            raise ValueError("Invalid decomposition branch assignment.")
        if not 0 <= observable_index < len(self.observable_terms):
            raise ValueError("Invalid observable term.")
        instructions: list[list] = [[] for _ in self.evaluation.block_widths]
        observable, weight = self.observable_terms[observable_index]
        coefficient = float(weight.real) * prod(
            template.coefficients[value] for template, value in zip(self.templates, branches)
        )

        def append(port, operation, scored=False):
            block, qubit = self._location(port)
            instructions[block].append((operation, (qubit,), scored))

        def local_map(port, operations):
            for operation in operations:
                append(
                    port,
                    Measure() if operation.name == "qpd_measure" else operation,
                    operation.name == "qpd_measure",
                )

        def measure_axis(port, axis, scored):
            if axis == "X":
                append(port, HGate())
            elif axis == "Y":
                append(port, SdgGate())
                append(port, HGate())
            append(port, Measure(), scored)

        def prepare(port, axis, eigenbit):
            append(port, Reset())
            if eigenbit:
                append(port, XGate())
            if axis in ("X", "Y"):
                append(port, HGate())
            if axis == "Y":
                append(port, SGate())

        def grouped_map(group_id):
            position, selected, basis = self.group_factors[group_id]
            candidate = self.graph.grouping.by_id[group_id]
            branch = branches[position]
            if candidate.family == "excision":
                term = basis.excision.terms[branch]
                for local, (pin, pout) in enumerate(candidate.excision_ports):
                    axis = term.input_pauli[-local - 1]
                    if axis != "I":
                        measure_axis(pin, axis, True)
                    prepare(pout, term.output_pauli[-local - 1], term.eigenbits[local])
                return
            if candidate.family == "fusion":
                for local, pin in enumerate(candidate.pins):
                    local_map(pin, basis.basis.maps[branch][local])
                return
            for side in (0, 1):
                pins = tuple(
                    pin
                    for i, pin in enumerate(candidate.pins)
                    if ((selected.pattern >> i) & 1) == side
                )
                for step in basis.maps[branch][side]:
                    locations = [self._location(pins[q]) for q in step.qubits]
                    if len({block for block, _ in locations}) != 1:
                        raise ValueError("A joint local map does not fit one allocated fragment.")
                    block = locations[0][0]
                    instructions[block].append(
                        (step.operation, tuple(q for _, q in locations), step.scored)
                    )

        for op_id, item in zip(self.graph.problem.operation_ids, self.graph.problem.circuit.data):
            if item.operation.name == "barrier":
                continue
            for wire in self.graph.incoming_wires[op_id]:
                if (
                    wire is not None
                    and self.plan.state[wire]
                    and wire not in self.covered_variables
                ):
                    factor = self.factor_positions[wire]
                    maps = self.templates[factor].basis.maps[branches[factor]]
                    old, new = self.graph.variables[wire].ports
                    local_map(old, maps[0])
                    local_map(new, maps[1])
            if op_id in self.group_anchors:
                grouped_map(self.group_anchors[op_id])
            if op_id in self.replaced_operations:
                continue
            variable = self.graph.gate_variables.get(op_id)
            ports = self.local_operation_ports.get(op_id, self.graph.gate_ports[op_id])
            if variable is None or not self.plan.state[variable]:
                locations = [self._location(pin) for pin, _ in ports]
                if len(set(block for block, _ in locations)) != 1:
                    raise RuntimeError("A retained gate crosses executable blocks.")
                block = locations[0][0]
                instructions[block].append((item.operation, tuple(q for _, q in locations), False))
            else:
                factor = self.factor_positions[variable]
                template = self.templates[factor]
                branch = branches[factor]
                if template.kind == "gate":
                    for local, (pin, _) in enumerate(ports):
                        local_map(pin, template.basis.maps[branch][local])
                else:
                    term = template.excision.terms[branch]
                    for local, (pin, pout) in enumerate(ports):
                        axis = term.input_pauli[-local - 1]
                        if axis != "I":
                            measure_axis(pin, axis, True)
                        prepare(pout, term.output_pauli[-local - 1], term.eigenbits[local])
        for qubit, output in enumerate(self.graph.output_nodes):
            axis = observable[-qubit - 1]
            measure_axis(output, axis, axis != "I")
        circuits, masks = [], []
        for block, sequence in enumerate(instructions):
            measured = sum(op.name == "measure" for op, _, _ in sequence)
            if measured == 0:
                # Aer requires a classical result even for constant-score boundary fragments.
                sequence.append((Measure(), (0,), False))
                measured = 1
            circuit = QuantumCircuit(
                self.evaluation.block_widths[block], measured, name=f"hybrid_f{block}"
            )
            next_bit, mask = 0, 0
            for operation, qubits, scored in sequence:
                if operation.name == "measure":
                    circuit.append(operation, qubits, [next_bit])
                    if scored:
                        mask |= 1 << next_bit
                    next_bit += 1
                else:
                    circuit.append(operation, qubits)
            if circuit.num_qubits > self.plan.max_fragment_qubits:
                raise RuntimeError("A physical experiment violates the plan's width bound.")
            circuits.append(circuit)
            masks.append(mask)
        return HybridExperiment(tuple(circuits), tuple(masks), coefficient)

    def run_exact(self, *, max_experiments: int = 10000, max_qubits: int = 12) -> HybridResult:
        start = perf_counter()
        self.plan.validate(self.graph)
        branch_options = [
            tuple(i for i, c in enumerate(t.coefficients) if c) for t in self.templates
        ]
        count = (
            prod(map(len, branch_options))
            * len(self.observable_terms)
            * len(self.evaluation.block_widths)
        )
        if count > max_experiments:
            raise ValueError(f"Exact hybrid enumeration needs {count} physical experiments.")
        expectation = 0.0
        executed = 0
        for branches in product(*branch_options):
            for observable_index in range(len(self.observable_terms)):
                experiment = self.experiment(branches, observable_index=observable_index)
                if not experiment.coefficient:
                    continue
                results = execute_exact(list(experiment.circuits), max_qubits=max_qubits)
                means = [
                    sum(
                        probability * (-1 if (outcome & mask).bit_count() % 2 else 1)
                        for outcome, probability in distribution.items()
                    )
                    for distribution, mask in zip(results.distributions, experiment.score_masks)
                ]
                expectation += experiment.coefficient * prod(means)
                executed += len(experiment.circuits)
        return HybridResult(
            float(expectation),
            None,
            0,
            0,
            executed,
            self.gamma,
            self.estimator_gamma,
            perf_counter() - start,
            "exact-enumerated",
        )

    def run_sampled(self, *, trials: int, seed: int = 0, noise_model=None) -> HybridResult:
        """Importance-sample global branches, then pair independent fragment shots.

        Random ordering of each fragment's score histogram recovers independent
        trial pairing. The returned SE includes QPD and measurement randomness.
        """
        start = perf_counter()
        self.plan.validate(self.graph)
        if isinstance(trials, bool) or not isinstance(trials, int) or trials < 2:
            raise ValueError("trials must be an integer at least two.")
        if self.estimator_gamma == 0:
            return HybridResult(
                0, 0, 0, 0, 0, self.gamma, 0, perf_counter() - start, "importance-sampled"
            )
        rng = np.random.default_rng(seed)
        choices = []
        for template in self.templates:
            weights = np.abs(template.coefficients)
            choices.append(rng.choice(len(weights), size=trials, p=weights / weights.sum()))
        weights = np.array([abs(float(w.real)) for _, w in self.observable_terms])
        choices.append(rng.choice(len(weights), size=trials, p=weights / weights.sum()))
        assignments = Counter(zip(*choices))
        total, executed, physical_shots = 0.0, 0, 0
        for assignment, count in assignments.items():
            branches = tuple(int(v) for v in assignment[:-1])
            experiment = self.experiment(branches, observable_index=int(assignment[-1]))
            scores = np.ones(count, dtype=np.int8)
            for circuit, mask in zip(experiment.circuits, experiment.score_masks):
                results = execute_shots(
                    [circuit],
                    shots=count,
                    seed=int(rng.integers(0, 2**31 - 1)),
                    noise_model=noise_model,
                )
                negatives = sum(
                    int(round(probability * count))
                    for outcome, probability in results.distributions[0].items()
                    if (outcome & mask).bit_count() % 2
                )
                local = np.ones(count, dtype=np.int8)
                local[:negatives] = -1
                rng.shuffle(local)
                scores *= local
                executed += 1
                physical_shots += count
            total += self.estimator_gamma * np.sign(experiment.coefficient) * int(scores.sum())
        expectation = float(total / trials)
        se = float(np.sqrt(max(0, self.estimator_gamma**2 - expectation**2) / (trials - 1)))
        return HybridResult(
            expectation,
            se,
            trials,
            physical_shots,
            executed,
            self.gamma,
            self.estimator_gamma,
            perf_counter() - start,
            "importance-sampled",
        )


def compile_hybrid(graph: PartitionGraph, plan: HybridPlan) -> CompiledHybrid:
    return CompiledHybrid(graph, plan)
