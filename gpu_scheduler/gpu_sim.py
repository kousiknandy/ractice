from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from heapq import heappop, heappush
from typing import Union


class InstructionType(Enum):
    ALU = "ALU"
    MEMORY = "MEMORY"


@dataclass(frozen=True)
class Instruction:
    kind: InstructionType
    latency: int


@dataclass
class Branch:
    latency: int
    true_path: list[Op]
    false_path: list[Op]


Op = Union[Instruction, Branch]


@dataclass
class Frame:
    program: list[Op]
    ip: int
    restored_mask: int
    alternate: tuple[list[Op], int] | None


@dataclass
class Warp:
    id: int
    thread_mask: int
    program: list[Op]
    ip: int = 0
    active_mask: int = field(init=False, default=0)
    stack: list[Frame] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.active_mask = self.thread_mask

    @property
    def done(self) -> bool:
        return self.ip >= len(self.program) and not self.stack


@dataclass
class SimResult:
    total_cycles: int
    stall_cycles: dict[int, int]


def split_half(mask: int) -> tuple[int, int]:
    """Split the set bits of `mask` evenly; floor(n/2) bits go to taken."""
    half = bin(mask).count("1") // 2
    taken, m = 0, mask
    for _ in range(half):
        lsb = m & -m
        taken |= lsb
        m ^= lsb
    return taken, mask ^ taken


class Simulator:
    def __init__(self, warps: list[Warp]) -> None:
        self.warps = warps
        self.cycle: int = 0
        self.ready: int = (1 << len(warps)) - 1
        self.pending: list[tuple[int, int]] = []
        self.stall_cycles: dict[int, int] = {w.id: 0 for w in warps}
        self.trace: list[tuple[int, int, int]] = []

    def _pick(self) -> int:
        lsb = self.ready & -self.ready
        return lsb.bit_length() - 1

    def _wake(self, up_to: int) -> None:
        while self.pending and self.pending[0][0] <= up_to:
            _, idx = heappop(self.pending)
            self.ready |= 1 << idx

    def _resolve(self, warp: Warp) -> None:
        while warp.stack and (warp.ip >= len(warp.program) or warp.active_mask == 0):
            frame = warp.stack[-1]
            alt = frame.alternate
            if alt is not None:
                frame.alternate = None
                alt_program, alt_mask = alt
                if alt_mask != 0:
                    warp.program = alt_program
                    warp.ip = 0
                    warp.active_mask = alt_mask
                    continue
            warp.stack.pop()
            warp.program = frame.program
            warp.ip = frame.ip
            warp.active_mask = frame.restored_mask

    def _dispatch(self, warp: Warp) -> int:
        self.trace.append((self.cycle, warp.id, warp.active_mask))
        instr = warp.program[warp.ip]
        warp.ip += 1
        if isinstance(instr, Branch):
            taken, not_taken = split_half(warp.active_mask)
            warp.stack.append(Frame(
                program=warp.program,
                ip=warp.ip,
                restored_mask=warp.active_mask,
                alternate=(instr.false_path, not_taken),
            ))
            warp.program = instr.true_path
            warp.ip = 0
            warp.active_mask = taken
        self._resolve(warp)
        return instr.latency

    def run(self) -> SimResult:
        active = len(self.warps)
        while active > 0:
            if self.ready == 0:
                self.cycle = self.pending[0][0]
                self._wake(self.cycle)

            idx = self._pick()
            warp = self.warps[idx]
            latency = self._dispatch(warp)

            if warp.done:
                self.ready &= ~(1 << idx)
                active -= 1
            elif latency > 0:
                self.ready &= ~(1 << idx)
                heappush(self.pending, (self.cycle + latency, idx))
                self.stall_cycles[warp.id] += latency - 1

            self.cycle += 1
            self._wake(self.cycle)

        return SimResult(total_cycles=self.cycle, stall_cycles=self.stall_cycles)
