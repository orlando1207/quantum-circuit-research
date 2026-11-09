"""Capacity cover cuts for the WG/WGE MILP.

A fragment's width is at least the number of distinct qubits it touches, so any
connected edge set T whose vertices touch more than M distinct qubits cannot be
fully retained:

    sum_{e in T} x_e >= 1.

The inequality involves only x, so it is valid for WG and WGE and for any label
count K. Cuts are separated greedily (Prim-style growth from every vertex on the
LP values x*) and leaves that add no new qubit are pruned before a cut is kept.
"""

from __future__ import annotations

import heapq
from math import ceil
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, milp

from .partition_graph import PartitionGraph


def _adjacency(graph: PartitionGraph):
    adj = [[] for _ in graph.nodes]
    for left, right, _role, e in graph.edges:
        adj[left].append((right, e))
        adj[right].append((left, e))
    return adj


def separate_cover_cuts(
    graph: PartitionGraph, max_qubits: int, xval, *, max_cuts: int = 400, adj=None
) -> list[tuple[int, ...]]:
    """Cover cuts violated by the edge values `xval` (indexed by graph variable)."""
    adj = adj if adj is not None else _adjacency(graph)
    qubits = [node["qubit"] for node in graph.nodes]
    cuts: dict[tuple[int, ...], None] = {}
    for start in range(len(adj)):
        in_tree = {start: None}
        seen = {qubits[start]: 1}
        weight = 0.0
        heap = [(xval[e], e, u, start) for u, e in adj[start]]
        heapq.heapify(heap)
        found = False
        while heap:
            w, e, u, parent = heapq.heappop(heap)
            if u in in_tree:
                continue
            if weight + w >= 1.0 - 1e-6:
                break
            in_tree[u] = (parent, e)
            weight += w
            seen[qubits[u]] = seen.get(qubits[u], 0) + 1
            if len(seen) > max_qubits:
                found = True
                break
            for nxt, e2 in adj[u]:
                if nxt not in in_tree:
                    heapq.heappush(heap, (xval[e2], e2, nxt, u))
        if not found:
            continue
        children = {v: 0 for v in in_tree}
        for link in in_tree.values():
            if link is not None:
                children[link[0]] += 1
        stack = [v for v, c in children.items() if c == 0 and in_tree[v] is not None]
        while stack:
            v = stack.pop()
            if seen[qubits[v]] <= 1:
                continue
            seen[qubits[v]] -= 1
            parent, _e = in_tree.pop(v)
            children[parent] -= 1
            if children[parent] == 0 and in_tree[parent] is not None:
                stack.append(parent)
        cut = tuple(sorted(link[1] for link in in_tree.values() if link is not None))
        cuts[cut] = None
        if len(cuts) >= max_cuts:
            break
    return list(cuts)


def cover_cut_pool(
    graph: PartitionGraph,
    *,
    max_qubits: int,
    excision: bool = True,
    rounds: int = 40,
    time_budget: float = 30.0,
    K: int | None = None,
) -> list[tuple[int, ...]]:
    """Iteratively separate cover cuts against the MILP's LP relaxation."""
    from .milp import build_milp

    started = perf_counter()
    n = graph.problem.circuit.num_qubits
    K = K if K is not None else ceil(n / max_qubits) + 2
    model = build_milp(graph, max_qubits=max_qubits, K=K, excision=excision)
    adj = _adjacency(graph)
    pool: list[tuple[int, ...]] = []
    known: set[tuple[int, ...]] = set()
    A = model.constraints.A.tocsr()
    lb, ub = model.constraints.lb, model.constraints.ub
    from scipy import sparse
    from scipy.optimize import LinearConstraint

    for _ in range(rounds):
        if perf_counter() - started > time_budget:
            break
        constraints = [LinearConstraint(A, lb, ub)]
        if pool:
            rows = np.repeat(np.arange(len(pool)), [len(c) for c in pool])
            cols = np.concatenate([model.x_offset + np.array(c) for c in pool])
            C = sparse.csr_matrix((np.ones(len(cols)), (rows, cols)),
                                  shape=(len(pool), model.n_vars))
            constraints.append(LinearConstraint(C, np.ones(len(pool)), np.inf))
        remaining = time_budget - (perf_counter() - started)
        res = milp(model.c, integrality=np.zeros(model.n_vars),
                   bounds=Bounds(model.var_lb, model.var_ub), constraints=constraints,
                   options={"time_limit": max(remaining, 1.0)})
        if res.x is None:
            break
        xval = res.x[model.x_offset : model.x_offset + model.n_edges]
        new = [c for c in separate_cover_cuts(graph, max_qubits, xval, adj=adj) if c not in known]
        if not new:
            break
        known.update(new)
        pool.extend(new)
    return pool
