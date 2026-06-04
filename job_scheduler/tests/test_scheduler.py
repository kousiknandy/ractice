import pytest

from gpu_scheduler import (
    BestFit,
    Cluster,
    FirstFit,
    MetricsCollector,
    Scheduler,
    WorstFit,
    bandwidth_score,
)
from gpu_scheduler.cluster import CROSS_RACK, SAME_NODE, SAME_RACK


def make_scheduler(strategy=None, racks=2, nodes=2, gpus=4, metrics=None):
    cluster = Cluster.build_uniform(racks, nodes, gpus)
    return Scheduler(cluster, strategy or BestFit(), metrics)


# -- scoring ---------------------------------------------------------------

def test_score_single_gpu_is_max():
    s = make_scheduler()
    alloc = s.schedule("j", 1)
    assert alloc.bandwidth_score == SAME_NODE


def test_score_single_node():
    s = make_scheduler(gpus=4)
    alloc = s.schedule("j", 4)
    assert len(alloc.placement) == 1
    assert alloc.bandwidth_score == SAME_NODE


def test_score_same_rack_multi_node():
    # 2 nodes x 4 GPUs per rack; 6 GPUs cannot fit one node but fits one rack.
    s = make_scheduler(racks=2, nodes=2, gpus=4)
    alloc = s.schedule("j", 6)
    assert len(alloc.placement) >= 2
    assert {s.cluster.node(n).rack_id for n in alloc.placement} == {0} or len(
        {s.cluster.node(n).rack_id for n in alloc.placement}
    ) == 1
    assert alloc.bandwidth_score == SAME_RACK


def test_score_cross_rack():
    # One rack holds 8; 10 GPUs must span racks.
    s = make_scheduler(racks=2, nodes=2, gpus=4)
    alloc = s.schedule("j", 10)
    rack_ids = {s.cluster.node(n).rack_id for n in alloc.placement}
    assert len(rack_ids) > 1
    assert alloc.bandwidth_score == CROSS_RACK


def test_bandwidth_score_helper():
    c = Cluster.build_uniform(2, 2, 4)
    n0, n1 = c.racks[0].nodes  # same rack
    other = c.racks[1].nodes[0]
    assert bandwidth_score([n0]) == SAME_NODE
    assert bandwidth_score([n0, n1]) == SAME_RACK
    assert bandwidth_score([n0, other]) == CROSS_RACK


# -- capacity is a hard constraint -----------------------------------------

def test_node_never_exceeds_capacity():
    s = make_scheduler(racks=1, nodes=1, gpus=4)
    s.schedule("a", 3)
    s.schedule("b", 1)
    node = s.cluster.node(0)
    assert node.free == 0
    assert node.used <= node.capacity


def test_reject_when_cluster_full():
    s = make_scheduler(racks=1, nodes=1, gpus=4)
    assert s.schedule("a", 4).accepted
    rej = s.schedule("b", 1)
    assert not rej.accepted
    assert rej.bandwidth_score is None


def test_reject_oversized_job():
    s = make_scheduler(racks=2, nodes=2, gpus=4)  # 16 GPUs total
    assert not s.schedule("big", 17).accepted


# -- tier preference -------------------------------------------------------

def test_prefers_node_then_rack_then_cluster():
    s = make_scheduler(racks=2, nodes=2, gpus=4)
    assert s.schedule("node", 4).bandwidth_score == SAME_NODE
    assert s.schedule("rack", 4).bandwidth_score == SAME_NODE  # second whole node
    # Now each rack has one full + space; force a rack-level placement.
    s2 = make_scheduler(racks=2, nodes=2, gpus=4)
    assert s2.schedule("r", 5).bandwidth_score == SAME_RACK


# -- release ---------------------------------------------------------------

def test_release_returns_capacity():
    s = make_scheduler(racks=1, nodes=1, gpus=4)
    s.schedule("a", 4)
    assert not s.schedule("b", 1).accepted
    assert s.release("a") is True
    assert s.schedule("c", 4).accepted
    assert s.cluster.free_total == 0


def test_release_unknown_is_noop():
    s = make_scheduler()
    assert s.release("nope") is False


def test_duplicate_job_id_raises():
    s = make_scheduler()
    s.schedule("a", 1)
    with pytest.raises(ValueError):
        s.schedule("a", 1)


# -- weighted metric correctness -------------------------------------------

def test_gpu_weighted_average_hand_computed():
    m = MetricsCollector()
    m.record_schedule(1, 100, accepted=True)   # tiny, high score
    m.record_schedule(99, 25, accepted=True)   # big, low score
    # weighted = (1*100 + 99*25) / 100 = (100 + 2475)/100 = 25.75
    assert m.gpu_weighted_avg_bw == pytest.approx(25.75)
    # simple = (100 + 25)/2 = 62.5
    assert m.simple_avg_bw == pytest.approx(62.5)


def test_metrics_acceptance_rate_and_rejected_gpus():
    m = MetricsCollector()
    m.record_schedule(4, 100, accepted=True)
    m.record_schedule(8, None, accepted=False)
    assert m.acceptance_rate == pytest.approx(0.5)
    assert m.rejected_gpus == 8


# -- fragmentation: tight (BestFit) preserves whole nodes; spread does not ---

def test_bestfit_preserves_whole_node_vs_worstfit():
    # 1 rack, 4 nodes x 4 GPUs. Place four 2-GPU jobs.
    def whole_nodes_left(strategy):
        s = make_scheduler(strategy=strategy, racks=1, nodes=4, gpus=4)
        for i in range(4):
            s.schedule(f"j{i}", 2)
        return sum(1 for n in s.cluster.nodes if n.is_empty)

    assert whole_nodes_left(BestFit()) > whole_nodes_left(WorstFit())


def test_firstfit_runs_end_to_end():
    s = make_scheduler(strategy=FirstFit(), racks=2, nodes=2, gpus=4)
    assert s.schedule("a", 6).bandwidth_score == SAME_RACK
