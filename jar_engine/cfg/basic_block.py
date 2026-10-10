"""
jar_engine.cfg.basic_block: Basic block partitioning and instruction representations.
Conforms to JVMS §4.7.3 and §6.5 with 5-rule leader identification.
"""

from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple, Iterator, Dict
import struct

from jar_engine.model.constants import Opcode, FIXED_OPCODE_LENGTHS
from jar_engine.model.member_info import ExceptionTableEntry
from jar_engine.walker.call_graph import InstructionScanner


class CFGEdgeType(str, Enum):
    FALLTHROUGH = "fallthrough"               # Sequential non-branching or false branch of conditional jump
    CONDITIONAL_BRANCH = "conditional_branch" # Taken target of if* conditional jump
    UNCONDITIONAL_BRANCH = "unconditional"    # Target of goto or goto_w
    SWITCH_CASE = "switch_case"               # Specific value match in tableswitch / lookupswitch
    SWITCH_DEFAULT = "switch_default"         # Default target of switch
    EXCEPTION = "exception"                   # Edge from try-covered block to handler block


@dataclass
class Instruction:
    pc: int
    opcode: int
    operands: bytes = b""

    @property
    def length(self) -> int:
        return 1 + len(self.operands)

    @property
    def next_pc(self) -> int:
        return self.pc + self.length

    @property
    def is_conditional_branch(self) -> bool:
        return self.opcode in {
            Opcode.IFEQ, Opcode.IFNE, Opcode.IFLT, Opcode.IFGE, Opcode.IFGT, Opcode.IFLE,
            Opcode.IF_ICMPEQ, Opcode.IF_ICMPNE, Opcode.IF_ICMPLT, Opcode.IF_ICMPGE,
            Opcode.IF_ICMPGT, Opcode.IF_ICMPLE, Opcode.IF_ACMPEQ, Opcode.IF_ACMPNE,
            Opcode.IFNULL, Opcode.IFNONNULL
        }

    @property
    def is_unconditional_branch(self) -> bool:
        return self.opcode in {Opcode.GOTO, Opcode.GOTO_W}

    @property
    def is_switch(self) -> bool:
        return self.opcode in {Opcode.TABLESWITCH, Opcode.LOOKUPSWITCH}

    @property
    def is_return(self) -> bool:
        return self.opcode in {
            Opcode.IRETURN, Opcode.LRETURN, Opcode.FRETURN,
            Opcode.DRETURN, Opcode.ARETURN, Opcode.RETURN
        }

    @property
    def is_throw(self) -> bool:
        return self.opcode == Opcode.ATHROW

    @property
    def is_terminal(self) -> bool:
        """Instructions that never fall through to the next sequential instruction."""
        return self.is_unconditional_branch or self.is_return or self.is_throw or (self.opcode == Opcode.RET)

    @property
    def has_fallthrough(self) -> bool:
        return not self.is_terminal and not self.is_switch

    def get_branch_targets(self) -> List[int]:
        """Extracts absolute target PCs for all jump and switch instructions."""
        targets: List[int] = []
        if self.is_conditional_branch or self.opcode in (Opcode.GOTO, Opcode.JSR):
            if len(self.operands) >= 2:
                offset = struct.unpack(">h", self.operands[:2])[0]
                targets.append(self.pc + offset)
        elif self.opcode in (Opcode.GOTO_W, Opcode.JSR_W):
            if len(self.operands) >= 4:
                offset = struct.unpack(">i", self.operands[:4])[0]
                targets.append(self.pc + offset)
        elif self.opcode == Opcode.TABLESWITCH:
            pad = (4 - (self.pc + 1) % 4) % 4
            if len(self.operands) >= pad + 12:
                default_off = struct.unpack(">i", self.operands[pad : pad + 4])[0]
                targets.append(self.pc + default_off)
                low, high = struct.unpack(">ii", self.operands[pad + 4 : pad + 12])
                count = max(0, high - low + 1)
                for i in range(count):
                    off_start = pad + 12 + 4 * i
                    if off_start + 4 <= len(self.operands):
                        case_off = struct.unpack(">i", self.operands[off_start : off_start + 4])[0]
                        targets.append(self.pc + case_off)
        elif self.opcode == Opcode.LOOKUPSWITCH:
            pad = (4 - (self.pc + 1) % 4) % 4
            if len(self.operands) >= pad + 8:
                default_off = struct.unpack(">i", self.operands[pad : pad + 4])[0]
                targets.append(self.pc + default_off)
                npairs = struct.unpack(">i", self.operands[pad + 4 : pad + 8])[0]
                for i in range(max(0, npairs)):
                    off_start = pad + 8 + 8 * i + 4
                    if off_start + 4 <= len(self.operands):
                        case_off = struct.unpack(">i", self.operands[off_start : off_start + 4])[0]
                        targets.append(self.pc + case_off)
        return targets

    def to_bytes(self) -> bytes:
        return bytes([self.opcode]) + self.operands


@dataclass
class CFGEdge:
    source_id: int
    target_id: int
    edge_type: CFGEdgeType
    case_key: Optional[int] = None            # Integer key for SWITCH_CASE
    catch_type: Optional[int] = None          # Constant pool class index (0 for any/finally)
    catch_class_name: Optional[str] = None    # Resolved class name for exception


@dataclass
class BasicBlock:
    id: int
    start_pc: int
    end_pc: int
    instructions: List[Instruction] = field(default_factory=list)
    exception_handlers: List[ExceptionTableEntry] = field(default_factory=list)

    @property
    def terminator(self) -> Optional[Instruction]:
        return self.instructions[-1] if self.instructions else None

    @property
    def is_terminal(self) -> bool:
        return self.terminator.is_terminal if self.terminator else False

    @property
    def has_fallthrough(self) -> bool:
        if not self.terminator:
            return False
        return not self.terminator.is_terminal and not self.terminator.is_switch

    def contains_pc(self, pc: int) -> bool:
        return self.start_pc <= pc < self.end_pc


def decode_instructions(code: bytes) -> List[Instruction]:
    """Decodes bytecode into a flat list of Instruction objects."""
    instructions: List[Instruction] = []
    for pc, opcode, operands in InstructionScanner.scan(code):
        instructions.append(Instruction(pc=pc, opcode=opcode, operands=operands))
    return instructions


def find_leaders(
    instructions: List[Instruction],
    exception_table: List[ExceptionTableEntry],
    code_len: int
) -> Set[int]:
    """
    Identifies leader PCs using the 5-rule algorithm:
    1. Method entry: PC 0 is a leader.
    2. Branch targets: Targets of conditional jumps, GOTOs, switches.
    3. Post-branch / Post-terminator: Instruction immediately following a branch or terminator.
    4. Exception handlers: handler_pc in exception table.
    5. Try-catch boundaries: start_pc and end_pc of try blocks.
    """
    if not instructions:
        return set()

    leaders: Set[int] = {instructions[0].pc}
    pc_set = {inst.pc for inst in instructions}

    for inst in instructions:
        # Rule 2: Targets of branches
        for target in inst.get_branch_targets():
            if target in pc_set:
                leaders.add(target)

        # Rule 3: Instruction following a branch or terminator
        if inst.is_conditional_branch or inst.is_terminal or inst.is_switch:
            if inst.next_pc in pc_set:
                leaders.add(inst.next_pc)

    # Rule 4 & 5: Exception table boundaries
    for entry in exception_table:
        if entry.start_pc in pc_set:
            leaders.add(entry.start_pc)
        if entry.end_pc in pc_set:
            leaders.add(entry.end_pc)
        if entry.handler_pc in pc_set:
            leaders.add(entry.handler_pc)

    return leaders


def partition_basic_blocks(
    code: bytes,
    exception_table: Optional[List[ExceptionTableEntry]] = None
) -> List[BasicBlock]:
    """
    Decodes bytecode and partitions it into BasicBlock instances using the 5-rule leader algorithm.
    """
    if exception_table is None:
        exception_table = []

    instructions = decode_instructions(code)
    if not instructions:
        return []

    code_len = len(code)
    leaders = find_leaders(instructions, exception_table, code_len)

    blocks: List[BasicBlock] = []
    curr_insts: List[Instruction] = []
    block_id = 0

    for i, inst in enumerate(instructions):
        if inst.pc in leaders and curr_insts:
            # Seal previous block
            start_pc = curr_insts[0].pc
            end_pc = curr_insts[-1].next_pc
            b = BasicBlock(id=start_pc, start_pc=start_pc, end_pc=end_pc, instructions=curr_insts)
            blocks.append(b)
            curr_insts = []

        curr_insts.append(inst)

    if curr_insts:
        start_pc = curr_insts[0].pc
        end_pc = curr_insts[-1].next_pc
        b = BasicBlock(id=start_pc, start_pc=start_pc, end_pc=end_pc, instructions=curr_insts)
        blocks.append(b)

    # Attach relevant exception handlers to each block
    for block in blocks:
        for ex in exception_table:
            # Block is covered by try block if block start falls inside [start_pc, end_pc)
            if ex.start_pc <= block.start_pc < ex.end_pc:
                block.exception_handlers.append(ex)

    return blocks
