"""Synthetic workload generator: an explicit stream of schedule/release operations.

Job sizes are drawn from three classes (frequencies and ranges follow the brief):
    small  : 1-4 GPUs    (most frequent, short-lived)
    medium : 8-64 GPUs   (regular)
    large  : 128-1024 GPUs (rare, long-lived)

Releases are interleaved at unpredictable points by drawing from the pool of
currently-active jobs. Everything is deterministic given a seed so the *same*
stream can be replayed through different strategies for a fair comparison.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# (weight, low, high) per class. Larger jobs are rarer and longer-lived, which we
# model by making them less likely to be released on any given step.
SMALL = (0.70, 1, 4)
MEDIUM = (0.25, 8, 64)
LARGE = (0.05, 128, 1024)

# Per-class probability that an active job of that class is eligible for release
# on a given release step -- larger jobs live longer.
RELEASE_BIAS = {"small": 1.0, "medium": 0.5, "large": 0.15}

Op = tuple  # ("schedule", job_id, num_gpus) | ("release", job_id)


def _size_class(num_gpus: int) -> str:
    if num_gpus <= 4:
        return "small"
    if num_gpus <= 64:
        return "medium"
    return "large"


@dataclass
class WorkloadParams:
    num_jobs: int = 2000
    release_prob: float = 0.45  # chance to attempt a release after each schedule
    small: tuple = SMALL
    medium: tuple = MEDIUM
    large: tuple = LARGE


def _draw_size(rng: random.Random, params: WorkloadParams) -> int:
    r = rng.random()
    if r < params.small[0]:
        lo, hi = params.small[1], params.small[2]
    elif r < params.small[0] + params.medium[0]:
        lo, hi = params.medium[1], params.medium[2]
    else:
        lo, hi = params.large[1], params.large[2]
    return rng.randint(lo, hi)


def generate_stream(seed: int, params: WorkloadParams | None = None) -> list[Op]:
    params = params or WorkloadParams()
    rng = random.Random(seed)
    ops: list[Op] = []
    active: dict[str, str] = {}  # job_id -> size_class

    for i in range(params.num_jobs):
        num_gpus = _draw_size(rng, params)
        job_id = f"j{i}"
        ops.append(("schedule", job_id, num_gpus))
        active[job_id] = _size_class(num_gpus)

        if active and rng.random() < params.release_prob:
            # Pick an active job to release, biased so large jobs persist longer.
            candidates = [
                jid for jid, cls in active.items() if rng.random() < RELEASE_BIAS[cls]
            ]
            if candidates:
                victim = rng.choice(candidates)
                ops.append(("release", victim))
                del active[victim]

    return ops
