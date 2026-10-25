"""Shared action/result types and exact static component packing."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import TYPE_CHECKING

from .templates import ActionTemplate

if TYPE_CHECKING:
    from .grouping import SelectedGroup


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
    independent_log_overhead: float | None = None
    groups: tuple[SelectedGroup, ...] = ()
    grouping_selection_complete: bool | None = None
    excision_wire_cuts: int = 0

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
