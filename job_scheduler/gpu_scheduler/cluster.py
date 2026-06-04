"""Cluster topology: racks -> nodes -> GPUs, plus the bandwidth scoring rule.

The bandwidth score of a placement is the *minimum* pairwise bandwidth among the
GPUs the job is given. Because the minimum dominates, the score collapses to a
step function of how spread out a placement is:

    all GPUs on one node            -> 100
    spans nodes, stays in one rack  ->  50
    spans racks                     ->  25
"""

from __future__ import annotations

from dataclasses import dataclass, field

SAME_NODE = 100
SAME_RACK = 50
CROSS_RACK = 25


@dataclass
class Node:
    node_id: int
    rack_id: int
    capacity: int
    free: int

    @property
    def used(self) -> int:
        return self.capacity - self.free

    @property
    def is_empty(self) -> bool:
        return self.free == self.capacity


@dataclass
class Rack:
    rack_id: int
    nodes: list[Node] = field(default_factory=list)

    @property
    def free_total(self) -> int:
        return sum(n.free for n in self.nodes)

    @property
    def capacity_total(self) -> int:
        return sum(n.capacity for n in self.nodes)


class Cluster:
    """A hierarchical GPU cluster with incrementally maintained free-GPU aggregates."""

    def __init__(self, racks: list[Rack]):
        self.racks = racks
        self._nodes_by_id: dict[int, Node] = {}
        for rack in racks:
            for node in rack.nodes:
                self._nodes_by_id[node.node_id] = node
        # Cached aggregate, kept in sync by allocate/free primitives.
        self._free_total = sum(n.free for n in self._nodes_by_id.values())

    @classmethod
    def build_uniform(
        cls, num_racks: int, nodes_per_rack: int, gpus_per_node: int
    ) -> "Cluster":
        racks: list[Rack] = []
        node_id = 0
        for rack_id in range(num_racks):
            rack = Rack(rack_id=rack_id)
            for _ in range(nodes_per_rack):
                rack.nodes.append(
                    Node(
                        node_id=node_id,
                        rack_id=rack_id,
                        capacity=gpus_per_node,
                        free=gpus_per_node,
                    )
                )
                node_id += 1
            racks.append(rack)
        return cls(racks)

    @property
    def free_total(self) -> int:
        return self._free_total

    @property
    def capacity_total(self) -> int:
        return sum(n.capacity for n in self._nodes_by_id.values())

    @property
    def gpus_per_node(self) -> int:
        # Uniform topology: every node shares the same capacity.
        return next(iter(self._nodes_by_id.values())).capacity

    @property
    def nodes(self) -> list[Node]:
        return list(self._nodes_by_id.values())

    def node(self, node_id: int) -> Node:
        return self._nodes_by_id[node_id]

    def allocate_on_node(self, node: Node, n: int) -> None:
        if n > node.free:
            raise ValueError(
                f"cannot allocate {n} GPUs on node {node.node_id} with {node.free} free"
            )
        node.free -= n
        self._free_total -= n

    def free_on_node(self, node: Node, n: int) -> None:
        if node.free + n > node.capacity:
            raise ValueError(
                f"cannot free {n} GPUs on node {node.node_id}: would exceed capacity"
            )
        node.free += n
        self._free_total += n


def bandwidth_score(nodes_used: list[Node]) -> int:
    """Score a placement given the distinct nodes it occupies (each with count > 0)."""
    if len(nodes_used) <= 1:
        return SAME_NODE
    racks_used = {n.rack_id for n in nodes_used}
    if len(racks_used) == 1:
        return SAME_RACK
    return CROSS_RACK
