"""Source-aware group LUTs and capacity-aware, conflict-free group selection.

The source circuit, including one-qubit operations, is the authority. A group
can replace its gates only if moving them to its anchor preserves the unitary,
its interior wires are uncut, and its local parties fit physical fragments.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import combinations
from math import log, prod

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, Pauli

from .graph_common import _root, _union
from .group_qpd import (
    embedded_pauli,
    general_joint_basis,
    joint_pauli_basis,
    joint_pauli_recipes,
    multiply_paulis,
    native_lcu,
)


@dataclass(frozen=True)
class GroupingConfig:
    max_gates: int = 4
    window_gates: int = 6
    max_support: int = 6
    exact_selection_limit: int = 10
    group_moves: bool = True
    generic_budget: int = 32
    anchor_strategy: str = "both"
    wire_windows: bool = True
    families: tuple[str, ...] = (
        "fusion",
        "cascade",
        "parallel",
        "rotations",
        "clifford",
        "generic",
    )

    def validate(self):
        if self.anchor_strategy not in {"first", "last", "both"}:
            raise ValueError("anchor_strategy must be first, last, or both.")
        if not isinstance(self.wire_windows, bool):
            raise ValueError("wire_windows must be boolean.")
        for name in ("max_gates", "window_gates", "max_support"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer at least two.")
        if self.max_gates > 6 or self.max_support > 8:
            raise ValueError("Group LUTs are bounded to six gates and eight source qubits.")
        if (
            not isinstance(self.exact_selection_limit, int)
            or not 0 <= self.exact_selection_limit <= 16
        ):
            raise ValueError("exact_selection_limit must be between zero and 16.")
        if (
            isinstance(self.generic_budget, bool)
            or not isinstance(self.generic_budget, int)
            or self.generic_budget < 0
        ):
            raise ValueError("generic_budget must be a nonnegative integer.")
        if set(self.families) - {
            "fusion",
            "cascade",
            "parallel",
            "rotations",
            "clifford",
            "generic",
        }:
            raise ValueError("Unknown grouping family.")


@dataclass(frozen=True)
class GroupCandidate:
    group_id: str
    family: str
    operations: tuple[str, ...]
    absorbed_operations: tuple[str, ...]
    variables: tuple[int, ...]
    guards: tuple[int, ...]
    qubits: tuple[int, ...]
    pins: tuple[int, ...]
    patterns: tuple[int, ...]
    gamma: float
    independent_log_overhead: float
    anchor: str
    template_id: str | None = None
    pattern_gammas: tuple[tuple[int, float], ...] = ()

    def gamma_for(self, pattern):
        return dict(self.pattern_gammas).get(pattern, self.gamma)


@dataclass(frozen=True)
class SelectedGroup:
    group_id: str
    pattern: int
    gamma: float


@dataclass(frozen=True)
class GroupSelection:
    groups: tuple[SelectedGroup, ...]
    savings: float
    component_widths: tuple[int, ...]
    old_to_new: tuple[int, ...]
    component_blocks: tuple[int, ...]
    block_widths: tuple[int, ...]
    width_penalty: int
    complete: bool


def _constrained_two(widths, separations):
    """Exact two-bin packing with opposite-party constraints on group bundles."""
    neighbors = [set() for _ in widths]
    for a, b in separations:
        if a == b:
            return None
        neighbors[a].add(b)
        neighbors[b].add(a)
    colored, pieces = {}, []
    for root in range(len(widths)):
        if root in colored:
            continue
        colored[root] = 0
        stack, sides = [root], [[], []]
        while stack:
            v = stack.pop()
            sides[colored[v]].append(v)
            for u in sorted(neighbors[v]):
                if u in colored:
                    if colored[u] == colored[v]:
                        return None
                else:
                    colored[u] = 1 - colored[v]
                    stack.append(u)
        pieces.append(tuple(tuple(s) for s in sides))
    reachable = {0: ()}
    for a, b in pieces:
        nxt = {}
        for total, selected in reachable.items():
            for side in (a, b):
                used = total + sum(widths[v] for v in side)
                value = tuple(sorted((*selected, *side)))
                if used not in nxt or value < nxt[used]:
                    nxt[used] = value
        reachable = nxt
    return reachable


class GroupLibrary:
    schema = "source-aware-joint-groups-v2"

    def __init__(self, graph, config):
        config.validate()
        self.graph, self.config = graph, config
        self.items = tuple(graph.problem.circuit.data)
        self.ids = graph.problem.operation_ids
        self.positions = {op: i for i, op in enumerate(self.ids)}
        self.operands = tuple(
            tuple(graph.problem.circuit.find_bit(q).index for q in i.qubits) for i in self.items
        )
        self._commutation_cache = {}
        self._basis_cache = {}
        self._target_cache = {}
        self._native_cache = {}
        self.candidates = self._discover()
        self.by_id = {c.group_id: c for c in self.candidates}
        payload = {
            "schema": self.schema,
            "source": graph.problem.circuit_fingerprint,
            "config": asdict(config),
            "candidates": [asdict(c) for c in self.candidates],
        }
        self.fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        regions, orbits = set(), set()
        if config.group_moves:
            for c in self.candidates:
                for pattern in c.patterns:
                    party_regions = []
                    for side in (0, 1):
                        qubits = {q for i, q in enumerate(c.qubits) if (pattern >> i & 1) == side}
                        vertices = {
                            pin
                            for v in c.variables
                            for pin in graph.variables[v].ports
                            if graph.nodes[pin]["qubit"] in qubits
                        }
                        # Align the whole packet side, including both ends of
                        # every continuity guard; one anchor alone can break it.
                        vertices.update(
                            pin
                            for v in c.guards
                            if graph.variables[v].qubit in qubits
                            for pin in graph.variables[v].ports
                        )
                        region = tuple(sorted(vertices))
                        if region:
                            regions.add(region)
                            party_regions.append(region)
                    if len(party_regions) == 2:
                        orbits.add(tuple(party_regions))
        self.regions = tuple(sorted(regions))
        self.orbits = tuple(sorted(orbits))

    def _commutes(self, i, j):
        if not set(self.operands[i]).intersection(self.operands[j]):
            return True
        key = (i, j)
        if key not in self._commutation_cache:
            qs = sorted(set(self.operands[i]) | set(self.operands[j]))
            matrices = []
            for index in (i, j):
                qc = QuantumCircuit(len(qs))
                qc.append(self.items[index].operation, [qs.index(q) for q in self.operands[index]])
                matrices.append(Operator(qc).data)
            a, b = matrices
            self._commutation_cache[key] = bool(np.max(np.abs(a @ b - b @ a)) < 1e-11)
        return self._commutation_cache[key]

    def _native(self, i):
        """Names alone cannot certify a grouping identity for custom gates."""
        if i not in self._native_cache:
            self._native_cache[i] = self._native_uncached(i)
        return self._native_cache[i]

    def _native_uncached(self, i):
        operation = self.items[i].operation
        if operation.name not in {"cx", "cz", "cp", "rxx", "ryy", "rzz", "swap"}:
            return False
        try:
            posts, terms = native_lcu(operation, self.operands[i], self.operands[i])
            local = QuantumCircuit(2)
            for q, matrix in posts:
                local.unitary(matrix, [self.operands[i].index(q)])
            matrix = Operator(local).data @ sum(c * Pauli(p).to_matrix() for c, p in terms)
            target = Operator(operation).data
            overlap = np.vdot(target, matrix)
            phase = overlap / abs(overlap) if abs(overlap) else 1
            return bool(np.max(np.abs(matrix - phase * target)) < 1e-10)
        except (ValueError, TypeError):
            return False

    def _family(self, indices):
        supports = [set(self.operands[i]) for i in indices]
        names = [self.items[i].operation.name for i in indices]
        if all(s == supports[0] for s in supports):
            return "fusion"
        if not all(self._native(i) for i in indices):
            return None
        if len(set(names)) == 1 and names[0] in {"cx", "cz"}:
            distinct_pairs = len({tuple(sorted(s)) for s in supports}) == len(supports)
            if names[0] == "cz" and distinct_pairs and set.intersection(*supports):
                return "cascade"
            if (
                names[0] == "cx"
                and distinct_pairs
                and any(len({self.operands[i][side] for i in indices}) == 1 for side in (0, 1))
            ):
                return "cascade"
        if sum(map(len, supports)) == len(set.union(*supports)):
            return "parallel"
        if set(names) <= {"cp", "cz", "rzz"} or set(names) <= {"rxx"} or set(names) <= {"ryy"}:
            return "rotations"
        if set(names) <= {"cx", "cz", "swap"}:
            return "clifford"
        return "generic"

    def _local_clifford(self, index):
        matrix = Operator(self.items[index].operation).data
        paulis = [Pauli(p).to_matrix() for p in "IXYZ"]
        return all(
            any(
                np.max(np.abs(matrix.conj().T @ p @ matrix - sign * q)) < 1e-11
                for q in paulis
                for sign in (1, -1)
            )
            for p in paulis
        )

    def _target(self, candidate):
        if candidate.group_id not in self._target_cache:
            selected = set(candidate.operations) | set(candidate.absorbed_operations)
            qc = QuantumCircuit(len(candidate.qubits))
            for op, item, operands in zip(self.ids, self.items, self.operands, strict=True):
                if op in selected:
                    qc.append(item.operation, [candidate.qubits.index(q) for q in operands])
            self._target_cache[candidate.group_id] = Operator(qc).data
        return self._target_cache[candidate.group_id]

    def _index_windows(self):
        config = self.config
        gates = [self.positions[op] for op in self.graph.gate_variables]
        windows = set()
        for offset, first in enumerate(gates):
            tail = gates[offset + 1 : offset + config.window_gates]
            for count in range(1, min(config.max_gates - 1, len(tail)) + 1):
                windows.update((first, *rest) for rest in combinations(tail, count))
        if config.wire_windows:
            # Follow source-qubit chains rather than only global gate distance.
            # Unrelated gates on other wires must not hide a potential packet.
            for q in range(self.graph.problem.circuit.num_qubits):
                chain = [i for i in gates if q in self.operands[i]]
                for length in range(2, config.max_gates + 1):
                    windows.update(
                        tuple(chain[j : j + length]) for j in range(len(chain) - length + 1)
                    )
        return sorted(windows)

    def _discover(self):
        graph, config = self.graph, self.config
        candidates, generic_examined, seen = [], 0, set()
        for indices in self._index_windows():
            original_family = self._family(indices)
            if original_family not in config.families:
                continue
            support = tuple(sorted({q for i in indices for q in self.operands[i]}))
            if len(support) > config.max_support:
                continue
            first, last = indices[0], indices[-1]
            anchors = (
                (first, last)
                if config.anchor_strategy == "both"
                else (first if config.anchor_strategy == "first" else last,)
            )
            for anchor in anchors:

                def crosses(i, j):
                    return i < j <= anchor or anchor <= j < i

                family, absorbed, valid = original_family, [], True
                selected = set(indices)
                for j in range(first, last + 1):
                    if j in selected or self.items[j].operation.name == "barrier":
                        continue
                    if not set(self.operands[j]).intersection(support):
                        continue
                    commutes = all(self._commutes(i, j) for i in indices if crosses(i, j))
                    if len(self.operands[j]) == 1 and family in {"fusion", "generic"}:
                        absorbed.append(j)
                    elif len(self.operands[j]) == 1 and not commutes:
                        if (
                            all(
                                self.items[i].operation.name in {"cx", "cz", "swap"}
                                for i in indices
                            )
                            and self._local_clifford(j)
                            and "clifford" in config.families
                        ):
                            family = "clifford"
                        elif "generic" in config.families:
                            family = "generic"
                        else:
                            valid = False
                            break
                        absorbed.append(j)
                    elif not commutes:
                        valid = False
                        break
                if not valid:
                    continue
                if family == "generic":
                    if (
                        generic_examined >= config.generic_budget
                        or len(support) > 4
                        or prod(4 if self.items[i].operation.name == "swap" else 2 for i in indices)
                        > 8
                    ):
                        continue
                    generic_examined += 1
                    absorbed = [
                        j
                        for j in range(first, last + 1)
                        if len(self.operands[j]) == 1
                        and set(self.operands[j]).intersection(support)
                        and self.items[j].operation.name != "barrier"
                    ]
                # Every moved operation, including absorbed locals, must commute
                # with each external operation it crosses in this direction.
                packet = selected | set(absorbed)
                if any(
                    not self._commutes(i, j)
                    for i in packet
                    for j in range(first, last + 1)
                    if j not in packet
                    and crosses(i, j)
                    and self.items[j].operation.name != "barrier"
                ):
                    continue
                operations = tuple(self.ids[i] for i in indices)
                semantic_key = (family, operations, tuple(absorbed))
                if semantic_key in seen:
                    continue
                variables = tuple(graph.gate_variables[op] for op in operations)
                if any(len(graph.variables[v].options) < 2 for v in variables):
                    continue
                guards = tuple(
                    v
                    for v, variable in enumerate(graph.variables)
                    if variable.kind == "wire"
                    and variable.qubit in support
                    and first < self.positions[variable.operation_id] <= last
                )
                pins = tuple(
                    next(
                        pin
                        for v in variables
                        for pin in graph.variables[v].ports
                        if graph.nodes[pin]["qubit"] == q
                    )
                    for q in support
                )
                patterns = tuple(
                    bits << 1
                    for bits in range(1 << (len(support) - 1))
                    if all(
                        ((bits << 1) >> support.index(self.operands[i][0]) & 1)
                        != ((bits << 1) >> support.index(self.operands[i][1]) & 1)
                        for i in indices
                    )
                )
                if not patterns:
                    continue
                base = sum(graph.variables[v].options[1].log_overhead for v in variables)
                key = hashlib.sha256(
                    json.dumps([family, operations, absorbed, anchor]).encode()
                ).hexdigest()
                candidate = GroupCandidate(
                    key,
                    family,
                    operations,
                    tuple(self.ids[i] for i in absorbed),
                    variables,
                    guards,
                    support,
                    pins,
                    patterns,
                    3.0,
                    base,
                    self.ids[anchor],
                )
                if family == "fusion":
                    template, _ = graph.registry.for_gate(UnitaryGate(self._target(candidate)))
                    if template is None:
                        continue
                    candidate = GroupCandidate(
                        **{
                            **asdict(candidate),
                            "gamma": template.gamma,
                            "template_id": template.template_id,
                        }
                    )
                else:
                    pattern_gammas = []
                    if family != "generic":
                        _, terms = self._pauli_lcu(candidate)
                    for pattern in patterns:
                        if family == "generic":
                            coefficients, left, right, _ = self._general_lcu(candidate, pattern)
                            value = general_joint_basis(coefficients, left, right, cost_only=True)
                        else:
                            sides = tuple(
                                tuple(
                                    q for i, q in enumerate(support) if ((pattern >> i) & 1) == side
                                )
                                for side in (0, 1)
                            )
                            value = sum(
                                abs(c) for c, _, _ in joint_pauli_recipes(terms, support, sides)
                            )
                        if 2 * log(max(1.0, value)) < base - 1e-10:
                            pattern_gammas.append((pattern, float(value)))
                    if not pattern_gammas:
                        continue
                    candidate = GroupCandidate(
                        **{
                            **asdict(candidate),
                            "patterns": tuple(p for p, _ in pattern_gammas),
                            "gamma": min(v for _, v in pattern_gammas),
                            "pattern_gammas": tuple(pattern_gammas),
                        }
                    )
                if 2 * log(max(1.0, candidate.gamma)) < base - 1e-10:
                    candidates.append(candidate)
                    seen.add(semantic_key)
        return tuple(sorted(candidates, key=lambda c: c.group_id))

    def _allocate(self, widths, parent, separations, limit, mode, max_components):
        roots = [_root(parent, i) for i in range(len(widths))]
        ids = {root: i for i, root in enumerate(dict.fromkeys(roots))}
        mapping = tuple(ids[root] for root in roots)
        merged = [0] * len(ids)
        for i, width in enumerate(widths):
            merged[mapping[i]] += width
        separated = [(mapping[a], mapping[b]) for a, b in separations]
        if any(a == b for a, b in separated) or max(merged) > limit:
            return None
        if mode == "two_blocks":
            reachable = _constrained_two(merged, separated)
            if reachable is None:
                return None
            total = sum(merged)
            subtotal = min(
                reachable,
                key=lambda t: (
                    max(0, t - limit) ** 2 + max(0, total - t - limit) ** 2,
                    abs(total - 2 * t),
                    t,
                ),
            )
            if max(subtotal, total - subtotal) > limit:
                return None
            a = set(reachable[subtotal])
            blocks = tuple(0 if i in a else 1 for i in range(len(merged)))
            raw = (subtotal, total - subtotal)
            block_widths = tuple(w for w in raw if w)
            if 0 in raw:
                blocks = (0,) * len(merged)
        else:
            blocks, block_widths = tuple(range(len(merged))), tuple(merged)
        penalty = max(0, len(merged) - max_components) ** 2 if max_components is not None else 0
        return tuple(merged), mapping, blocks, block_widths, penalty

    def select(self, state, widths, node_components, limit, mode, max_components):
        """Jointly choose disjoint groups and legal local register bundles."""
        eligible = []
        for c in self.candidates:
            if not all(state[v] for v in c.variables) or any(state[v] for v in c.guards):
                continue
            components = tuple(node_components[pin] for pin in c.pins)
            occupied = sum(
                1 << self.positions[op] for op in (*c.operations, *c.absorbed_operations)
            )
            for pattern in c.patterns:
                sides = tuple(
                    tuple(
                        sorted(
                            {
                                components[i]
                                for i in range(len(c.qubits))
                                if ((pattern >> i) & 1) == side
                            }
                        )
                    )
                    for side in (0, 1)
                )
                if set(sides[0]).intersection(sides[1]):
                    continue
                savings = c.independent_log_overhead - 2 * log(c.gamma_for(pattern))
                eligible.append((savings, c.group_id, pattern, occupied, sides))
        if not eligible:
            return None
        eligible.sort(key=lambda row: (-row[0], row[1], row[2]))
        initial_parent = list(range(len(widths)))
        initial = self._allocate(widths, initial_parent, (), limit, mode, max_components)
        if initial is None:
            return None  # Local grouping cannot repair an oversized base component.
        best = ((), 0.0, initial)

        def rank(solution):
            groups, savings, allocation = solution
            return (
                allocation[-1],
                -savings,
                len(allocation[3]),
                tuple((g.group_id, g.pattern) for g in groups),
            )

        def append(parent, seps, row):
            _, _, _, _, sides = row
            candidate_parent = parent.copy()
            for side in sides:
                for v in side[1:]:
                    _union(candidate_parent, side[0], v)
            next_seps = (*seps, (sides[0][0], sides[1][0]))
            allocation = self._allocate(
                widths, candidate_parent, next_seps, limit, mode, max_components
            )
            return candidate_parent, next_seps, allocation

        # Greedy incumbent(s) also bound the small exact selection search.
        orders = [eligible, sorted(eligible, key=lambda r: (-r[0] / r[3].bit_count(), r[1], r[2]))]
        for order in orders:
            parent, seps, used, groups, savings = initial_parent.copy(), (), 0, (), 0.0
            for row in order:
                if used & row[3]:
                    continue
                new_parent, new_seps, allocation = append(parent, seps, row)
                if allocation is None:
                    continue
                parent, seps = new_parent, new_seps
                used |= row[3]
                groups += (SelectedGroup(row[1], row[2], self.by_id[row[1]].gamma_for(row[2])),)
                savings += row[0]
                solution = groups, savings, allocation
                if rank(solution) < rank(best):
                    best = solution
        complete = len(eligible) <= self.config.exact_selection_limit
        if complete:

            def visit(i, parent, seps, used, groups, savings, allocation):
                nonlocal best
                solution = groups, savings, allocation
                if rank(solution) < rank(best):
                    best = solution
                if i == len(eligible):
                    return
                row = eligible[i]
                if not used & row[3]:
                    nxt, nxt_seps, allocated = append(parent, seps, row)
                    if allocated is not None:
                        visit(
                            i + 1,
                            nxt,
                            nxt_seps,
                            used | row[3],
                            (
                                *groups,
                                SelectedGroup(row[1], row[2], self.by_id[row[1]].gamma_for(row[2])),
                            ),
                            savings + row[0],
                            allocated,
                        )
                visit(i + 1, parent, seps, used, groups, savings, allocation)

            visit(0, initial_parent.copy(), (), 0, (), 0.0, initial)
        groups, savings, allocation = best
        if not groups:
            return None
        merged, mapping, blocks, block_widths, penalty = allocation
        return GroupSelection(
            tuple(sorted(groups, key=lambda g: g.group_id)),
            savings,
            merged,
            mapping,
            blocks,
            block_widths,
            penalty,
            complete,
        )

    def basis(self, selected):
        key = (selected.group_id, selected.pattern)
        if key in self._basis_cache:
            return self._basis_cache[key]
        c = self.by_id[selected.group_id]
        if c.family == "fusion":
            return self.graph.registry.entries[c.template_id]
        if c.family == "generic":
            coefficients, left, right, sides = self._general_lcu(c, selected.pattern)
            order = sides[0] + sides[1]
            indices = [
                sum(((bits >> i) & 1) << c.qubits.index(q) for i, q in enumerate(order))
                for bits in range(2 ** len(order))
            ]
            target = self._target(c)[np.ix_(indices, indices)]
            basis = general_joint_basis(coefficients, left, right, target)
            if not np.isclose(basis.gamma, selected.gamma, rtol=1e-10, atol=1e-10):
                raise ValueError("General packet executable cost differs from its LUT.")
            self._basis_cache[key] = basis
            return basis
        sides = tuple(
            tuple(q for i, q in enumerate(c.qubits) if ((selected.pattern >> i) & 1) == side)
            for side in (0, 1)
        )
        posts, terms = self._pauli_lcu(c)
        basis = joint_pauli_basis(terms, posts, c.qubits, sides, self._target(c))
        if not np.isclose(basis.gamma, selected.gamma, atol=1e-10, rtol=1e-10):
            raise ValueError(
                f"Group cost does not match executable QPD: {basis.gamma} != {c.gamma}"
            )
        self._basis_cache[key] = basis
        return basis

    def _pauli_lcu(self, c):
        posts, terms = [], [(1.0 + 0j, "I" * len(c.qubits))]
        if c.family == "cascade":
            pairs = [self.operands[self.positions[op]] for op in c.operations]
            name = self.items[self.positions[c.operations[0]]].operation.name
            if name == "cz":
                common = min(set.intersection(*(set(p) for p in pairs)))
                axes = {q: "Z" for pair in pairs for q in pair}
            else:
                side = next(s for s in (0, 1) if len({p[s] for p in pairs}) == 1)
                common = pairs[0][side]
                axes = {
                    q: ("Z" if position == 0 else "X")
                    for pair in pairs
                    for position, q in enumerate(pair)
                }
            # Product of same-control/target CXs is a controlled parity Pauli.
            sides = tuple(
                tuple(q for i, q in enumerate(c.qubits) if ((c.patterns[0] >> i) & 1) == side)
                for side in (0, 1)
            )
            for side in sides:
                label = embedded_pauli({q: axes[q] for q in side}, side)
                p = Pauli(label).to_matrix()
                posts.append((side, (np.eye(len(p)) - 1j * p) / np.sqrt(2)))
            terms = [
                (1 / np.sqrt(2), "I" * len(c.qubits)),
                (1j / np.sqrt(2), embedded_pauli(axes, c.qubits)),
            ]
            del common  # The side assignment is already encoded by the gate pattern.
        else:
            accumulated = {q: np.eye(2, dtype=complex) for q in c.qubits}
            packet_ops = sorted(
                (*c.operations, *c.absorbed_operations), key=self.positions.__getitem__
            )
            for op in packet_ops:
                index = self.positions[op]
                if len(self.operands[index]) == 1:
                    q = self.operands[index][0]
                    accumulated[q] = Operator(self.items[index].operation).data @ accumulated[q]
                    continue
                local, expanded = native_lcu(
                    self.items[index].operation, self.operands[index], c.qubits
                )
                following = []
                for coefficient, label in expanded:
                    # Pull earlier local Cliffords to the left of the new Pauli
                    # generator. For commuting diagonal/parallel packets this
                    # also works with their non-Clifford local phase factors.
                    transformed = {}
                    for q in c.qubits:
                        axis = label[-c.qubits.index(q) - 1]
                        matrix = accumulated[q].conj().T @ Pauli(axis).to_matrix() @ accumulated[q]
                        match = next(
                            (
                                (sign, p)
                                for p in "IXYZ"
                                for sign in (1, -1)
                                if np.max(np.abs(matrix - sign * Pauli(p).to_matrix())) < 1e-10
                            ),
                            None,
                        )
                        if match is None:
                            raise ValueError("Packet requires a non-Pauli local instrument.")
                        coefficient *= match[0]
                        transformed[q] = match[1]
                    label = embedded_pauli(transformed, c.qubits)
                    for previous, old_label in terms:
                        phase, new_label = multiply_paulis(label, old_label)
                        following.append((coefficient * previous * phase, new_label))
                terms = following
                for q, matrix in local:
                    accumulated[q] = matrix @ accumulated[q]
            posts = [((q,), matrix) for q, matrix in accumulated.items()]
        return posts, terms

    def _general_lcu(self, c, pattern):
        sides = tuple(
            tuple(q for i, q in enumerate(c.qubits) if (pattern >> i & 1) == side)
            for side in (0, 1)
        )
        coefficients = [1.0 + 0j]
        local = [[np.eye(2 ** len(side), dtype=complex)] for side in sides]
        packet = sorted((*c.operations, *c.absorbed_operations), key=self.positions.__getitem__)
        for op in packet:
            index = self.positions[op]
            item, operands = self.items[index], self.operands[index]
            if len(operands) == 1:
                side = next(s for s in (0, 1) if operands[0] in sides[s])
                qc = QuantumCircuit(len(sides[side]))
                qc.append(item.operation, [sides[side].index(operands[0])])
                u = Operator(qc).data
                local[side] = [u @ matrix for matrix in local[side]]
                continue
            posts, terms = native_lcu(item.operation, operands, c.qubits)
            post_matrices = []
            for side in sides:
                qc = QuantumCircuit(len(side))
                for q, matrix in posts:
                    if q in side:
                        qc.unitary(matrix, [side.index(q)])
                post_matrices.append(Operator(qc).data)
            next_coefficients, next_local = [], [[], []]
            for coefficient, label in terms:
                factors = []
                for s, side in enumerate(sides):
                    p = "".join(label[-c.qubits.index(q) - 1] for q in reversed(side))
                    factors.append(post_matrices[s] @ Pauli(p).to_matrix())
                for j, previous in enumerate(coefficients):
                    next_coefficients.append(coefficient * previous)
                    for s in (0, 1):
                        next_local[s].append(factors[s] @ local[s][j])
            coefficients, local = next_coefficients, next_local
        return coefficients, local[0], local[1], sides

    def save(self, path):
        from pathlib import Path

        Path(path).write_text(
            json.dumps(
                {
                    "schema": self.schema,
                    "fingerprint": self.fingerprint,
                    "source_fingerprint": self.graph.problem.circuit_fingerprint,
                    "config": asdict(self.config),
                    "candidates": [asdict(c) for c in self.candidates],
                    "region_count": len(self.regions),
                    "cost_model": "Executable LO joint QPD; no live communication or ancillas",
                },
                indent=2,
            )
            + "\n"
        )
