"""Cost-bounded exact reference in the same finite hybrid action library."""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from time import perf_counter

from .annealing import AnnealConfig, allowed_options, seed_states
from .partition_graph import GraphEvaluation, PartitionGraph


@dataclass(frozen=True)
class ExactSearchResult:
    best: GraphEvaluation | None
    complete: bool
    nodes_visited: int
    states_evaluated: int
    elapsed_seconds: float


def exact_search(
    graph: PartitionGraph,
    config: AnnealConfig,
    *,
    max_nodes: int = 100000,
    incumbent: GraphEvaluation | None = None,
) -> ExactSearchResult:
    start = perf_counter()
    config.validate()
    if graph.graph_schema not in {"temporal-incidence-v2", "temporal-incidence-grouped-v3"}:
        raise ValueError("Core exact search requires the gate/wire graph.")
    graph.problem.validate_snapshot()
    if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes < 1:
        raise ValueError("max_nodes must be a positive integer.")
    if (
        config.partition_mode == "two_blocks"
        and graph.problem.circuit.num_qubits > 2 * config.max_qubits
    ):
        return ExactSearchResult(None, True, 0, 0, perf_counter() - start)
    if (
        config.partition_mode != "two_blocks"
        and config.max_components is not None
        and graph.problem.circuit.num_qubits > config.max_components * config.max_qubits
    ):
        return ExactSearchResult(None, True, 0, 0, perf_counter() - start)
    allowed = allowed_options(graph, config)
    grouping_costs_complete = True

    def evaluate(state):
        nonlocal grouping_costs_complete
        result = graph.evaluate(
            state,
            max_qubits=config.max_qubits,
            partition_mode=config.partition_mode,
            max_components=config.max_components,
            validate=False,
        )
        if result.grouping_selection_complete is False:
            grouping_costs_complete = False
        return result

    def score(evaluation):
        return evaluation.log_overhead + (
            log(len(evaluation.block_widths)) if config.objective == "fragment_shots" else 0
        )

    if incumbent is not None:
        # Incumbents from a different configuration/library cannot bound this search.
        graph.validate_state(incumbent.state)
        if any(value not in choices for value, choices in zip(incumbent.state, allowed)):
            raise ValueError("Incumbent uses a disabled action.")
        incumbent = evaluate(incumbent.state)
        if not incumbent.feasible:
            raise ValueError("Exact-search incumbent is infeasible under these constraints.")
    else:
        for state in seed_states(graph, config, count=1):
            candidate = evaluate(state)
            if candidate.feasible:
                incumbent = candidate
    variables = [i for i, choices in enumerate(allowed) if len(choices) > 1]
    variables.sort(
        key=lambda i: -max(graph.variables[i].options[v].log_overhead for v in allowed[i] if v)
    )
    state = [0] * len(graph.variables)
    visited = evaluated = 0
    complete = True

    def visit(position, cost, extra_segments):
        nonlocal visited, evaluated, incumbent, complete
        if visited >= max_nodes:
            complete = False
            return
        visited += 1
        # Every remaining action has nonnegative cost and segment increment.
        # A conservative floating margin avoids pruning a numerically smaller optimum.
        if incumbent is not None and cost > score(incumbent) + 1e-12:
            return
        if (
            config.partition_mode == "two_blocks"
            and graph.problem.circuit.num_qubits + extra_segments > 2 * config.max_qubits
        ):
            return
        if (
            config.partition_mode != "two_blocks"
            and config.max_components is not None
            and graph.problem.circuit.num_qubits + extra_segments
            > config.max_components * config.max_qubits
        ):
            return
        if position == len(variables):
            evaluated += 1
            candidate = evaluate(tuple(state))
            if candidate.feasible and (incumbent is None or score(candidate) < score(incumbent)):
                incumbent = candidate
            return
        index = variables[position]
        for value in allowed[index]:
            if not complete:
                break
            state[index] = value
            option = graph.variables[index].options[value]
            increment = int(option is not None and option.kind == "wire")
            visit(
                position + 1,
                cost
                + (
                    option.log_overhead
                    if option and (graph.grouping is None or option.kind == "wire")
                    else 0
                ),
                extra_segments + increment,
            )
        state[index] = 0

    visit(0, 0.0, 0)
    certified = complete and grouping_costs_complete
    if graph.grouping is not None and config.objective != "overhead":
        # The inner selector minimizes norm first, not norm times fragment count.
        certified = False
    return ExactSearchResult(incumbent, certified, visited, evaluated, perf_counter() - start)
