"""Temporal label proposals for binary and multiway graph partitions."""

from __future__ import annotations

from math import log

from .graph_common import _root, _union


class LabelMoves:
    def __init__(self, graph, allowed, block_size):
        parent = list(range(len(graph.nodes)))
        for left, right, _, variable in graph.edges:
            if len(allowed[variable]) == 1:
                _union(parent, left, right)
        units = {}
        for vertex in range(len(graph.nodes)):
            units.setdefault(_root(parent, vertex), []).append(vertex)
        self.units = tuple(tuple(unit) for unit in units.values())
        unit_of = {v: unit for unit in self.units for v in unit}

        def expand(region):
            return tuple(sorted({v for node in region for v in unit_of[node]}))

        regions, polish = set(), set(self.units)
        for wire in graph.wire_nodes:
            for length in range(2, min(block_size, len(wire)) + 1):
                for start in range(len(wire) - length + 1):
                    regions.add(expand(wire[start : start + length]))
            for boundary in range(1, len(wire)):
                polish.add(expand(wire[:boundary]))
                polish.add(expand(wire[boundary:]))
        self.regions = tuple(sorted(regions))
        self.polish_regions = tuple(
            sorted(polish | regions, key=lambda region: (len(region), region))
        )
        self.group_orbits = tuple(
            sorted(
                {
                    (expand(left), expand(right))
                    for left, right in getattr(getattr(graph, "grouping", None), "orbits", ())
                    if not set(expand(left)).intersection(expand(right))
                }
            )
        )
        self.spatial_orbits = ()
        if self.group_orbits:
            wires = graph.wire_nodes
            qubit_pairs = {
                tuple(sorted(graph.nodes[v]["qubit"] for v in variable.ports))
                for variable in graph.variables
                if variable.kind == "gate"
            }
            self.spatial_orbits = tuple(
                (expand(wires[a]), expand(wires[b]))
                for a, b in sorted(qubit_pairs)
                if not set(expand(wires[a])).intersection(expand(wires[b]))
            )

    @staticmethod
    def flip(labels, region):
        result = list(labels)
        for node in region:
            result[node] ^= 1
        return tuple(result)

    def propose(self, labels, rng, block_probability):
        block = bool(self.regions) and rng.random() < block_probability
        regions = self.regions if block else self.units
        region = regions[int(rng.integers(len(regions)))]
        return self.flip(labels, region), block


class MultiwayLabelMoves(LabelMoves):
    """Swap two colors inside a fixed region, with an exact Hastings correction.

    Targets mix existing colors (joins/exchanges) with unused colors (splits).
    Labels are kept in a fixed palette during SA; no state-dependent relabeling
    occurs inside the Markov chain. The component evaluator defines fragments.
    """

    def __init__(self, graph, allowed, block_size, split_probability=0.25):
        super().__init__(graph, allowed, block_size)
        self.palette_size = max(2, len(graph.nodes))
        self.split_probability = split_probability
        unit_of = {v: unit for unit in self.units for v in unit}

        def expand(vertices):
            return tuple(sorted({v for node in vertices for v in unit_of[node]}))

        regions = set(self.regions) | set(self.polish_regions)
        regions.update(expand(wire) for wire in graph.wire_nodes)
        regions.update(
            expand(variable.ports) for variable in graph.variables if variable.kind == "gate"
        )
        regions.update(expand(region) for region in getattr(graph, "group_regions", ()))
        self.regions = tuple(sorted(regions))
        self.polish_regions = tuple(sorted(regions | set(self.units), key=lambda r: (len(r), r)))

    @staticmethod
    def swap(labels, region, a, b):
        result = list(labels)
        for node in region:
            if result[node] == a:
                result[node] = b
            elif result[node] == b:
                result[node] = a
        return tuple(result)

    def target_probability(self, used, source, target):
        if source == target:
            return 0.0
        occupied = len(used) - 1
        empty = self.palette_size - len(used)
        split = self.split_probability if occupied and empty else float(bool(empty))
        if target in used:
            return (1 - split) / occupied if occupied else 0.0
        return split / empty if empty else 0.0

    def pair_probability(self, labels, region, a, b):
        used = set(labels)
        count_a = sum(labels[v] == a for v in region)
        count_b = sum(labels[v] == b for v in region)
        return (
            count_a * self.target_probability(used, a, b)
            + count_b * self.target_probability(used, b, a)
        ) / len(region)

    def propose(self, labels, rng, block_probability):
        block = bool(self.regions) and rng.random() < block_probability
        regions = self.regions if block else self.units
        region = regions[int(rng.integers(len(regions)))]
        source = labels[region[int(rng.integers(len(region)))]]
        used = set(labels)
        occupied = sorted(used - {source})
        empty = [color for color in range(self.palette_size) if color not in used]
        split = self.split_probability if occupied and empty else float(bool(empty))
        targets = empty if rng.random() < split else occupied
        target = targets[int(rng.integers(len(targets)))]
        candidate = self.swap(labels, region, source, target)
        forward = self.pair_probability(labels, region, source, target)
        reverse = self.pair_probability(candidate, region, source, target)
        return candidate, block, log(reverse) - log(forward)

    def alternatives(self, labels, region):
        """All useful color swaps for deterministic refinement, including a split."""
        used = set(labels)
        targets = sorted(used)
        if len(used) < self.palette_size:
            targets.append(next(c for c in range(self.palette_size) if c not in used))
        pairs = {
            tuple(sorted((labels[v], target)))
            for v in region
            for target in targets
            if labels[v] != target
        }
        for a, b in sorted(pairs):
            yield self.swap(labels, region, a, b)
