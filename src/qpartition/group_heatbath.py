"""Reversible heat-bath moves on fixed, coupled group-region swap orbits."""

from __future__ import annotations

from itertools import product
from math import log

import numpy as np

from .label_moves import MultiwayLabelMoves


def pair_probability(labels, a, b, palette_size, split_probability=0.25):
    """Choose an occupied color, then mix joins/exchanges with fresh-color splits."""
    used = set(labels)
    occupied, empty = len(used) - 1, palette_size - len(used)
    split = split_probability if occupied and empty else float(bool(empty))

    def target(color):
        return (
            ((1 - split) / occupied if occupied else 0.0)
            if color in used
            else (split / empty if empty else 0.0)
        )

    return (int(a in used) * target(b) + int(b in used) * target(a)) / len(used)


def orbit(labels, regions, a, b):
    """Independent involutions on disjoint fixed regions form a finite group."""
    states = set()
    for flags in product((False, True), repeat=len(regions)):
        state = labels
        for selected, region in zip(flags, regions, strict=True):
            if selected:
                state = MultiwayLabelMoves.swap(state, region, a, b)
        states.add(state)
    return tuple(sorted(states))


def choose_pair(labels, palette_size, rng, split_probability=0.25):
    used = sorted(set(labels))
    a = used[int(rng.integers(len(used)))]
    existing = [c for c in used if c != a]
    empty = [c for c in range(palette_size) if c not in used]
    split = split_probability if existing and empty else float(bool(empty))
    targets = empty if rng.random() < split else existing
    b = targets[int(rng.integers(len(targets)))]
    return min(a, b), max(a, b)


def probabilities(
    states, energies, temperature, pair, palette_size, binary, split_probability=0.25
):
    """Conditional weights of pi(state) q(auxiliary pair | state).

    The orbit is invariant under every member's swaps, so its normalizer is
    identical in both directions. Multiplying by q preserves detailed balance
    even when a color appears/disappears. No state-dependent relabeling occurs.
    """
    logs = np.array(
        [
            -energy / temperature
            + (0.0 if binary else log(pair_probability(s, *pair, palette_size, split_probability)))
            for s, energy in zip(states, energies, strict=True)
        ]
    )
    weights = np.exp(logs - max(logs))
    return weights / sum(weights)
