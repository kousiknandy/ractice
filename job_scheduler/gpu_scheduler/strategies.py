"""Pluggable packing strategies.

Within any tier the *bandwidth score is already fixed* (tier 1 -> 100, tier 2 ->
50, tier 3 -> 25). What a strategy controls is which specific node/rack to use
and how to spread GPUs across nodes -- i.e. fragmentation -- which determines the
scores achievable by *future* jobs in the stream. This is the only module where
that behavior differs.

Each strategy implements three hooks:
  * pick(candidates, free_of, n)   -- choose ONE container that fits (tier 1 node,
                                       tier 2 rack).
  * distribute(nodes, n)           -- assign n GPUs across the given nodes
                                       (sum of free >= n guaranteed by caller).
  * rack_order(racks)              -- order in which to consume racks when a job
                                       must span racks (tier 3).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Sequence, TypeVar

from gpu_scheduler.cluster import Node, Rack

T = TypeVar("T")


def _tight_fill(nodes_ordered: Sequence[Node], n: int) -> dict[int, int]:
    """Fill nodes one at a time (in the given order) until n GPUs are placed."""
    placement: dict[int, int] = {}
    remaining = n
    for node in nodes_ordered:
        if remaining == 0:
            break
        take = min(node.free, remaining)
        if take > 0:
            placement[node.node_id] = take
            remaining -= take
    if remaining > 0:
        raise ValueError("insufficient capacity for tight fill")
    return placement


def _even_spread(nodes: Sequence[Node], n: int) -> dict[int, int]:
    """Round-robin one GPU at a time across nodes (maximizes partial nodes)."""
    placement: dict[int, int] = {node.node_id: 0 for node in nodes}
    free_left = {node.node_id: node.free for node in nodes}
    remaining = n
    while remaining > 0:
        progressed = False
        for node in nodes:
            if remaining == 0:
                break
            if free_left[node.node_id] > 0:
                placement[node.node_id] += 1
                free_left[node.node_id] -= 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise ValueError("insufficient capacity for even spread")
    return {nid: c for nid, c in placement.items() if c > 0}


class Strategy(ABC):
    name: str

    @abstractmethod
    def pick(self, candidates: Sequence[T], free_of: Callable[[T], int], n: int) -> T:
        ...

    @abstractmethod
    def distribute(self, nodes: Sequence[Node], n: int) -> dict[int, int]:
        ...

    @abstractmethod
    def rack_order(self, racks: Sequence[Rack]) -> list[Rack]:
        ...


class BestFit(Strategy):
    """Tight packing: smallest container that fits; fill fewest nodes/racks.

    Preserves whole/large free regions for future high-locality placements.
    """

    name = "best_fit"

    def pick(self, candidates, free_of, n):
        fitting = [c for c in candidates if free_of(c) >= n]
        return min(fitting, key=free_of)

    def distribute(self, nodes, n):
        ordered = sorted(nodes, key=lambda nd: nd.free, reverse=True)
        return _tight_fill(ordered, n)

    def rack_order(self, racks):
        return sorted(racks, key=lambda r: r.free_total, reverse=True)


class FirstFit(Strategy):
    """First container (in topology/index order) that fits; fill in index order."""

    name = "first_fit"

    def pick(self, candidates, free_of, n):
        for c in candidates:
            if free_of(c) >= n:
                return c
        raise ValueError("no fitting candidate")

    def distribute(self, nodes, n):
        ordered = sorted(nodes, key=lambda nd: nd.node_id)
        return _tight_fill(ordered, n)

    def rack_order(self, racks):
        return sorted(racks, key=lambda r: r.rack_id)


class WorstFit(Strategy):
    """Spread: largest container; even round-robin fill. Deliberate contrast baseline."""

    name = "worst_fit"

    def pick(self, candidates, free_of, n):
        fitting = [c for c in candidates if free_of(c) >= n]
        return max(fitting, key=free_of)

    def distribute(self, nodes, n):
        return _even_spread(list(nodes), n)

    def rack_order(self, racks):
        return sorted(racks, key=lambda r: r.free_total)


DEFAULT_STRATEGIES: list[Strategy] = [BestFit(), FirstFit(), WorstFit()]
