"""Explicit migration of gate/wire-only historical plans to canonical boundaries."""

from __future__ import annotations

from .hybrid import HybridPlan
from .legacy_graph import LegacyPartitionGraph
from .partition_graph import PartitionGraph


def migrate_legacy_plan(graph: PartitionGraph, plan: HybridPlan) -> HybridPlan:
    if plan.graph_schema != "four-port-v1":
        plan.validate(graph)
        return plan
    old = LegacyPartitionGraph(graph.problem)
    previous = plan.validate(old)
    state = [0] * len(graph.variables)
    ids = graph.problem.operation_ids
    positions = {op: i for i, op in enumerate(ids)}
    for variable, value in zip(old.variables, plan.state):
        if not value:
            continue
        template = variable.options[value]
        if template.kind == "excision":
            raise ValueError("Excision plans belong to the historical/refinement path.")
        if template.kind == "gate":
            target = graph.gate_variables[variable.operation_id]
        else:
            next_op = next(
                (
                    op
                    for op in ids[positions[variable.operation_id] :]
                    if op in graph.gate_variables
                    and any(
                        graph.nodes[port]["qubit"] == variable.qubit
                        for port in graph.variables[graph.gate_variables[op]].ports
                    )
                ),
                None,
            )
            if next_op is None:
                raise ValueError("Trailing boundary requires explicit redundant-cut elimination.")
            if (next_op, variable.qubit) not in graph.wire_variables:
                raise ValueError("Input-side boundary requires explicit redundant-cut elimination.")
            target = graph.wire_variables[next_op, variable.qubit]
        if state[target]:
            raise ValueError("Several historical cuts collapse to one canonical boundary.")
        state[target] = 1
    migrated = HybridPlan.create(
        graph,
        tuple(state),
        max_qubits=plan.max_fragment_qubits,
        partition_mode=plan.partition_mode,
        max_components=plan.max_components,
    )
    current = migrated.validate(graph)
    if abs(current.log_overhead - previous.log_overhead) > 1e-10:
        raise ValueError("Migration changed the physical coefficient norm.")
    if sorted(current.component_widths) != sorted(previous.component_widths):
        raise ValueError("Migration changed the physical component widths.")
    return migrated
