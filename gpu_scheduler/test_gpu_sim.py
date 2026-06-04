import unittest

from gpu_sim import (
    Branch,
    Instruction,
    InstructionType,
    Simulator,
    Warp,
    split_half,
)

ALU = InstructionType.ALU
MEM = InstructionType.MEMORY
ALU1 = Instruction(ALU, 1)


def warp(wid: int, *ops: tuple[InstructionType, int]) -> Warp:
    program = [Instruction(kind, lat) for kind, lat in ops]
    return Warp(id=wid, thread_mask=0xFFFFFFFF, program=program)


def popcounts(trace: list[tuple[int, int, int]]) -> list[int]:
    return [bin(m).count("1") for _, _, m in trace]


class TestSimulator(unittest.TestCase):
    def test_single_warp_zero_latency(self):
        result = Simulator([warp(0, *((ALU, 0),) * 5)]).run()
        self.assertEqual(result.total_cycles, 5)
        self.assertEqual(result.stall_cycles, {0: 0})

    def test_single_warp_memory_stall(self):
        result = Simulator([warp(0, (ALU, 1), (MEM, 10), (ALU, 1))]).run()
        self.assertEqual(result.total_cycles, 12)
        self.assertEqual(result.stall_cycles[0], 9)

    def test_second_warp_fills_stall(self):
        w0 = warp(0, (MEM, 10), (ALU, 1))
        w1 = warp(1, (ALU, 1), (ALU, 1), (ALU, 1))
        result = Simulator([w0, w1]).run()
        self.assertEqual(result.total_cycles, 11)
        self.assertEqual(result.stall_cycles, {0: 9, 1: 0})

    def test_all_warps_stall_skips_idle_cycles(self):
        w0 = warp(0, (MEM, 5), (ALU, 1))
        w1 = warp(1, (MEM, 5), (ALU, 1))
        result = Simulator([w0, w1]).run()
        self.assertEqual(result.total_cycles, 7)
        self.assertEqual(result.stall_cycles, {0: 4, 1: 4})

    def test_all_warps_complete(self):
        w0 = warp(0, (ALU, 2), (MEM, 5), (ALU, 1))
        w1 = warp(1, (ALU, 1), (ALU, 1))
        sim = Simulator([w0, w1])
        sim.run()
        self.assertTrue(all(w.done for w in sim.warps))


class TestBranch(unittest.TestCase):
    def test_split_half_distributes_bits_evenly(self):
        taken, not_taken = split_half(0xFFFFFFFF)
        self.assertEqual(bin(taken).count("1"), 16)
        self.assertEqual(bin(not_taken).count("1"), 16)
        self.assertEqual(taken | not_taken, 0xFFFFFFFF)
        self.assertEqual(taken & not_taken, 0)

    def test_split_half_nested_halving(self):
        a = 0xFFFFFFFF
        b, _ = split_half(a)
        c, _ = split_half(b)
        d, _ = split_half(c)
        self.assertEqual(
            [bin(x).count("1") for x in (a, b, c, d)], [32, 16, 8, 4]
        )

    def test_split_half_single_thread(self):
        taken, not_taken = split_half(0b1)
        self.assertEqual(taken, 0)
        self.assertEqual(not_taken, 1)

    def test_simple_branch_runs_both_paths(self):
        br = Branch(latency=1, true_path=[ALU1, ALU1], false_path=[ALU1])
        w = Warp(id=0, thread_mask=0xFFFFFFFF, program=[ALU1, br, ALU1])
        sim = Simulator([w])
        result = sim.run()
        # 6 dispatches total (1 + BR + 2 true + 1 false + 1), single-warp serial.
        self.assertEqual(result.total_cycles, 6)

    def test_mask_restored_after_branch(self):
        br = Branch(latency=1, true_path=[ALU1], false_path=[ALU1])
        w = Warp(id=0, thread_mask=0xFFFFFFFF, program=[br, ALU1])
        sim = Simulator([w])
        sim.run()
        # BRANCH @ 32, true.ALU @ 16, false.ALU @ 16, post-reconv ALU @ 32.
        self.assertEqual(popcounts(sim.trace), [32, 16, 16, 32])

    def test_nested_branch_halves_mask_again(self):
        inner = Branch(latency=1, true_path=[ALU1], false_path=[ALU1])
        outer = Branch(
            latency=1, true_path=[ALU1, inner, ALU1], false_path=[ALU1]
        )
        w = Warp(id=0, thread_mask=0xFFFFFFFF, program=[outer])
        sim = Simulator([w])
        result = sim.run()
        # outer BR @ 32, outer.true ALU @ 16, inner BR @ 16,
        # inner.true ALU @ 8, inner.false ALU @ 8,
        # outer.true ALU (post inner) @ 16, outer.false ALU @ 16.
        self.assertEqual(
            popcounts(sim.trace), [32, 16, 16, 8, 8, 16, 16]
        )
        self.assertEqual(result.total_cycles, 7)

    def test_three_deep_nesting_halves_to_four(self):
        b3 = Branch(latency=1, true_path=[ALU1], false_path=[ALU1])
        b2 = Branch(latency=1, true_path=[b3], false_path=[ALU1])
        b1 = Branch(latency=1, true_path=[b2], false_path=[ALU1])
        w = Warp(id=0, thread_mask=0xFFFFFFFF, program=[b1])
        sim = Simulator([w])
        sim.run()
        # Only the innermost split halves to 4. Outer `false` paths run with the
        # half that never entered the inner branch (so 8, then 16).
        # b1@32, b2@16, b3@8, b3.true@4, b3.false@4, b2.false@8, b1.false@16.
        self.assertEqual(popcounts(sim.trace), [32, 16, 8, 4, 4, 8, 16])

    def test_branch_with_empty_taken_side_runs_only_alternate(self):
        # Single active thread -> split is (0, 1); true_path is skipped.
        br = Branch(latency=1, true_path=[ALU1, ALU1], false_path=[ALU1])
        w = Warp(id=0, thread_mask=0x1, program=[br])
        sim = Simulator([w])
        result = sim.run()
        # Only BRANCH + false_path[0] dispatch — true_path never runs.
        self.assertEqual(result.total_cycles, 2)
        self.assertEqual(len(sim.trace), 2)
        self.assertEqual(popcounts(sim.trace), [1, 1])


if __name__ == "__main__":
    unittest.main()
