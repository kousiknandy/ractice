"""Offline comparison harness.

Replays the *same* generated streams through each strategy on a fresh cluster and
prints a comparison table plus a recommended default (highest GPU-weighted average
bandwidth, ties broken by acceptance rate).

Run:  python -m gpu_scheduler.simulate
"""

from __future__ import annotations

from dataclasses import dataclass

from gpu_scheduler.cluster import Cluster
from gpu_scheduler.metrics import MetricsCollector
from gpu_scheduler.scheduler import Scheduler
from gpu_scheduler.strategies import DEFAULT_STRATEGIES, Strategy
from gpu_scheduler.workload import Op, WorkloadParams, generate_stream


@dataclass
class Topology:
    num_racks: int
    nodes_per_rack: int
    gpus_per_node: int

    def build(self) -> Cluster:
        return Cluster.build_uniform(
            self.num_racks, self.nodes_per_rack, self.gpus_per_node
        )


def run_stream(topology: Topology, strategy: Strategy, ops: list[Op]) -> MetricsCollector:
    sched = Scheduler(topology.build(), strategy)
    for op in ops:
        if op[0] == "schedule":
            sched.schedule(op[1], op[2])
        else:
            sched.release(op[1])
    return sched.metrics


@dataclass
class StrategyResult:
    name: str
    gpu_weighted_avg_bw: float
    simple_avg_bw: float
    acceptance_rate: float
    histogram: dict[int, dict[str, int]]


def _average_results(name: str, runs: list[MetricsCollector]) -> StrategyResult:
    k = len(runs)
    hist: dict[int, dict[str, int]] = {}
    for m in runs:
        for score, d in m.score_histogram().items():
            agg = hist.setdefault(score, {"jobs": 0, "gpus": 0})
            agg["jobs"] += d["jobs"]
            agg["gpus"] += d["gpus"]
    return StrategyResult(
        name=name,
        gpu_weighted_avg_bw=sum(m.gpu_weighted_avg_bw for m in runs) / k,
        simple_avg_bw=sum(m.simple_avg_bw for m in runs) / k,
        acceptance_rate=sum(m.acceptance_rate for m in runs) / k,
        histogram=hist,
    )


def compare_strategies(
    strategies: list[Strategy],
    topology: Topology,
    workload: WorkloadParams,
    seeds: list[int],
) -> list[StrategyResult]:
    streams = [generate_stream(s, workload) for s in seeds]
    results: list[StrategyResult] = []
    for strategy in strategies:
        runs = [run_stream(topology, strategy, ops) for ops in streams]
        results.append(_average_results(strategy.name, runs))
    return results


def print_report(
    results: list[StrategyResult], topology: Topology, workload: WorkloadParams, seeds: list[int]
) -> None:
    total_gpus = topology.num_racks * topology.nodes_per_rack * topology.gpus_per_node
    print(
        f"Cluster: {topology.num_racks} racks x {topology.nodes_per_rack} nodes "
        f"x {topology.gpus_per_node} GPUs = {total_gpus} GPUs"
    )
    print(
        f"Workload: {workload.num_jobs} jobs/stream, "
        f"{len(seeds)} seed(s) averaged\n"
    )

    header = f"{'strategy':<12}{'gpu_wt_bw':>11}{'avg_bw':>9}{'accept%':>9}"
    header += f"{'%gpu@100':>10}{'%gpu@50':>9}{'%gpu@25':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        total = sum(d["gpus"] for d in r.histogram.values()) or 1
        pct = {s: 100.0 * d["gpus"] / total for s, d in r.histogram.items()}
        print(
            f"{r.name:<12}{r.gpu_weighted_avg_bw:>11.2f}{r.simple_avg_bw:>9.2f}"
            f"{100 * r.acceptance_rate:>9.2f}"
            f"{pct.get(100, 0):>10.1f}{pct.get(50, 0):>9.1f}{pct.get(25, 0):>9.1f}"
        )

    best = max(
        results, key=lambda r: (r.gpu_weighted_avg_bw, r.acceptance_rate)
    )
    print(
        f"\nRecommended default: {best.name} "
        f"(gpu-weighted avg bandwidth {best.gpu_weighted_avg_bw:.2f}, "
        f"acceptance {100 * best.acceptance_rate:.2f}%)"
    )


def main() -> None:
    topology = Topology(num_racks=16, nodes_per_rack=16, gpus_per_node=8)  # 2048 GPUs
    workload = WorkloadParams(num_jobs=3000, release_prob=0.5)
    seeds = [1, 2, 3, 4, 5]
    results = compare_strategies(DEFAULT_STRATEGIES, topology, workload, seeds)
    print_report(results, topology, workload, seeds)


if __name__ == "__main__":
    main()
