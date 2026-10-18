"""Capacity-respecting agglomeration seeds for multiway temporal partitions."""

from __future__ import annotations

import networkx as nx
import numpy as np


def bridge_seed_state(graph, allowed, *, max_qubits, include_gates=False):
    """Optimal capacity partition restricted to available graph bridges.

    After deleting every such bridge, quotient components form a forest. Joining
    a wire bridge saves one register; joining a gate bridge saves none. A tree
    knapsack DP chooses which bridge edges to restore. This is exact only within
    this restricted bridge family, not for general gate/wire cuts.
    """
    source = graph.networkx_graph()
    bridges = {frozenset(edge) for edge in nx.bridges(source)}
    indices = [
        i
        for i, variable in enumerate(graph.variables)
        if (variable.kind == "wire" or include_gates)
        and 1 in allowed[i]
        and frozenset(variable.ports) in bridges
    ]
    state = [0] * len(graph.variables)
    for index in indices:
        state[index] = 1
    fine = graph.evaluate(tuple(state), max_qubits=max_qubits, partition_mode="multiway")
    if not fine.feasible:
        return None
    forest = nx.Graph()
    forest.add_nodes_from(range(len(fine.component_widths)))
    for index in indices:
        left, right = graph.variables[index].ports
        a = fine.segment_components[fine.node_segments[left]]
        b = fine.segment_components[fine.node_segments[right]]
        forest.add_edge(a, b, variable=index)
    if not nx.is_forest(forest):
        raise RuntimeError("The bridge quotient must be a forest.")
    chosen = []
    for nodes in nx.connected_components(forest):
        root = min(nodes)
        parent, order = {root: None}, [root]
        for node in order:
            for child in sorted(forest[node]):
                if child != parent[node]:
                    parent[child] = node
                    order.append(child)
        tables = {}
        for node in reversed(order):
            table = {fine.component_widths[node]: (0.0, ())}
            for child in sorted(forest[node]):
                if parent.get(child) != node:
                    continue
                variable = forest[node][child]["variable"]
                edge_cost = graph.variables[variable].options[1].log_overhead
                joined = {}
                for used, (cost, cuts) in table.items():
                    for child_used, (child_cost, child_cuts) in tables[child].items():
                        options = [
                            (
                                used,
                                cost + child_cost + edge_cost,
                                tuple(sorted((*cuts, *child_cuts, variable))),
                            )
                        ]
                        joined_width = (
                            used + child_used - int(graph.variables[variable].kind == "wire")
                        )
                        if joined_width <= max_qubits:
                            options.append(
                                (
                                    joined_width,
                                    cost + child_cost,
                                    tuple(sorted((*cuts, *child_cuts))),
                                )
                            )
                        for weight, value, cutset in options:
                            item = (value, cutset)
                            if weight not in joined or item < joined[weight]:
                                joined[weight] = item
                table = joined
            tables[node] = table
        chosen.extend(min(tables[root].values())[1])
    result = [0] * len(graph.variables)
    for index in chosen:
        result[index] = 1
    return tuple(result)


def merge_seed_states(graph, allowed, *, max_qubits, count=12, seed=0):
    """Start from fine partitions and greedily restore supported graph edges.

    Two families prevent a spatial-only initialization bias: retain all gates
    and initially cut wires, or initially cut every available edge. Width is
    recomputed after each restoration, including the register saved by a wire
    join. No quantum simulation is used. Returned states obey width bounds;
    a fragment-count constraint is checked by the caller.
    """
    rng = np.random.default_rng(seed)
    starts = []
    for kinds in ({"wire"}, {"gate", "wire"}):
        starts.append(
            tuple(
                max(allowed[index]) if variable.kind in kinds else 0
                for index, variable in enumerate(graph.variables)
            )
        )
    candidates = []
    for attempt in range(max(2, count)):
        state = list(starts[attempt % len(starts)])
        evaluation = graph.evaluate(tuple(state), max_qubits=max_qubits, partition_mode="multiway")
        if not evaluation.feasible:
            continue
        indices = [i for i, value in enumerate(state) if value]
        if attempt < 2:
            indices.sort(key=lambda i: -graph.variables[i].options[state[i]].log_overhead)
        else:
            rng.shuffle(indices)
        changed = True
        while changed:
            changed = False
            for index in indices:
                if not state[index]:
                    continue
                old = state[index]
                state[index] = 0
                trial = graph.evaluate(
                    tuple(state), max_qubits=max_qubits, partition_mode="multiway", validate=False
                )
                if trial.feasible:
                    changed = True
                else:
                    state[index] = old
        candidates.append(tuple(state))
    return tuple(dict.fromkeys(candidates))
