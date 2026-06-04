"""The scheduler: greedy 3-tier placement that maximizes each job's bandwidth score.

Tier 1 (score 100): fit entirely on one node.
Tier 2 (score  50): fit entirely within one rack, spanning nodes.
Tier 3 (score  25): span racks.
Otherwise reject. Capacity is a hard constraint at every step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gpu_scheduler.cluster import Cluster, bandwidth_score
from gpu_scheduler.metrics import MetricsCollector
from gpu_scheduler.strategies import Strategy


@dataclass
class Allocation:
    job_id: str
    num_gpus: int
    placement: dict[int, int] = field(default_factory=dict)  # node_id -> gpu count
    bandwidth_score: int | None = None
    accepted: bool = False


class Scheduler:
    def __init__(
        self,
        cluster: Cluster,
        strategy: Strategy,
        metrics: MetricsCollector | None = None,
    ):
        self.cluster = cluster
        self.strategy = strategy
        self.metrics = metrics or MetricsCollector()
        self.allocations: dict[str, Allocation] = {}

    def schedule(self, job_id: str, num_gpus: int) -> Allocation:
        if job_id in self.allocations:
            raise ValueError(f"job {job_id!r} is already scheduled")
        if num_gpus <= 0:
            raise ValueError("num_gpus must be positive")

        placement = self._place(num_gpus)
        if placement is None:
            alloc = Allocation(job_id=job_id, num_gpus=num_gpus, accepted=False)
            self.metrics.record_schedule(num_gpus, None, accepted=False)
            return alloc

        self._commit(placement)
        nodes_used = [self.cluster.node(nid) for nid in placement]
        score = bandwidth_score(nodes_used)
        alloc = Allocation(
            job_id=job_id,
            num_gpus=num_gpus,
            placement=placement,
            bandwidth_score=score,
            accepted=True,
        )
        self.allocations[job_id] = alloc
        self.metrics.record_schedule(num_gpus, score, accepted=True)
        return alloc

    def release(self, job_id: str) -> bool:
        alloc = self.allocations.pop(job_id, None)
        if alloc is None:
            return False
        for node_id, count in alloc.placement.items():
            self.cluster.free_on_node(self.cluster.node(node_id), count)
        return True

    # -- placement tiers ---------------------------------------------------

    def _place(self, n: int) -> dict[int, int] | None:
        return (
            self._place_on_node(n)
            or self._place_in_rack(n)
            or self._place_across_racks(n)
        )

    def _place_on_node(self, n: int) -> dict[int, int] | None:
        if n > self.cluster.gpus_per_node:
            return None
        candidates = [nd for nd in self.cluster.nodes if nd.free >= n]
        if not candidates:
            return None
        node = self.strategy.pick(candidates, lambda nd: nd.free, n)
        return {node.node_id: n}

    def _place_in_rack(self, n: int) -> dict[int, int] | None:
        candidates = [r for r in self.cluster.racks if r.free_total >= n]
        if not candidates:
            return None
        rack = self.strategy.pick(candidates, lambda r: r.free_total, n)
        usable = [nd for nd in rack.nodes if nd.free > 0]
        return self.strategy.distribute(usable, n)

    def _place_across_racks(self, n: int) -> dict[int, int] | None:
        if self.cluster.free_total < n:
            return None
        racks = [r for r in self.cluster.racks if r.free_total > 0]
        placement: dict[int, int] = {}
        remaining = n
        for rack in self.strategy.rack_order(racks):
            if remaining == 0:
                break
            take = min(rack.free_total, remaining)
            usable = [nd for nd in rack.nodes if nd.free > 0]
            for node_id, count in self.strategy.distribute(usable, take).items():
                placement[node_id] = placement.get(node_id, 0) + count
            remaining -= take
        return placement

    def _commit(self, placement: dict[int, int]) -> None:
        for node_id, count in placement.items():
            self.cluster.allocate_on_node(self.cluster.node(node_id), count)
