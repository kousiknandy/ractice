"""Observability: per-job and aggregate bandwidth statistics for a stream of ops.

The headline metric is the GPU-count-weighted average bandwidth over accepted
jobs: a 100-GPU job placed at score 50 should weigh far more than a 1-GPU job at
score 100, because it represents far more GPU-to-GPU communication.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gpu_scheduler.cluster import CROSS_RACK, SAME_NODE, SAME_RACK

SCORES = (SAME_NODE, SAME_RACK, CROSS_RACK)


def size_class(num_gpus: int) -> str:
    if num_gpus <= 4:
        return "small"
    if num_gpus <= 64:
        return "medium"
    return "large"


@dataclass
class MetricsCollector:
    accepted: int = 0
    rejected: int = 0
    rejected_gpus: int = 0
    # Parallel records for accepted jobs.
    _gpus: list[int] = field(default_factory=list)
    _scores: list[int] = field(default_factory=list)

    def record_schedule(self, num_gpus: int, score: int | None, accepted: bool) -> None:
        if accepted:
            self.accepted += 1
            self._gpus.append(num_gpus)
            self._scores.append(score)  # type: ignore[arg-type]
        else:
            self.rejected += 1
            self.rejected_gpus += num_gpus

    @property
    def total(self) -> int:
        return self.accepted + self.rejected

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.total if self.total else 0.0

    @property
    def gpu_weighted_avg_bw(self) -> float:
        total_gpus = sum(self._gpus)
        if total_gpus == 0:
            return 0.0
        return sum(g * s for g, s in zip(self._gpus, self._scores)) / total_gpus

    @property
    def simple_avg_bw(self) -> float:
        return sum(self._scores) / len(self._scores) if self._scores else 0.0

    def score_histogram(self) -> dict[int, dict[str, int]]:
        """Per-score breakdown: {score: {'jobs': k, 'gpus': g}}."""
        hist = {s: {"jobs": 0, "gpus": 0} for s in SCORES}
        for g, s in zip(self._gpus, self._scores):
            hist[s]["jobs"] += 1
            hist[s]["gpus"] += g
        return hist

    def by_size_class(self) -> dict[str, dict[str, float]]:
        """Per-size-class accepted-job stats: count, total gpus, gpu-weighted bw."""
        classes: dict[str, dict[str, float]] = {
            c: {"jobs": 0, "gpus": 0, "gpu_weighted_avg_bw": 0.0}
            for c in ("small", "medium", "large")
        }
        for g, s in zip(self._gpus, self._scores):
            c = classes[size_class(g)]
            c["jobs"] += 1
            c["gpus"] += g
            c["gpu_weighted_avg_bw"] += g * s
        for c in classes.values():
            c["gpu_weighted_avg_bw"] = (
                c["gpu_weighted_avg_bw"] / c["gpus"] if c["gpus"] else 0.0
            )
        return classes
