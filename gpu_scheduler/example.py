from gpu_sim import Instruction, InstructionType, Simulator, Warp

ALU = Instruction(InstructionType.ALU, latency=1)
MEM = Instruction(InstructionType.MEMORY, latency=10)
FULL_MASK = 0xFFFFFFFF


def main() -> None:
    warps = [
        Warp(id=0, thread_mask=FULL_MASK, program=[ALU, MEM, ALU, ALU]),
        Warp(id=1, thread_mask=FULL_MASK, program=[ALU, ALU, MEM, ALU]),
        Warp(id=2, thread_mask=FULL_MASK, program=[ALU, ALU, ALU, ALU]),
    ]
    result = Simulator(warps).run()
    print(f"total_cycles = {result.total_cycles}")
    print(f"stall_cycles = {result.stall_cycles}")


if __name__ == "__main__":
    main()
