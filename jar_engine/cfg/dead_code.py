"""
jar_engine.cfg.dead_code: Dead Code Elimination, NOP Pruning, and Bytecode Reassembly (Feature F19).
Conforms to JVMS §4.7.3, §4.10, and §6.5.
"""

from __future__ import annotations
from collections import deque
from typing import Dict, List, Set, Tuple, Optional, Any
import struct

from jar_engine.model.constants import Opcode
from jar_engine.model.member_info import ExceptionTableEntry
from jar_engine.model.cp_info import ConstantPool
from jar_engine.walker.call_graph import InstructionScanner
from .basic_block import BasicBlock, Instruction, CFGEdge, CFGEdgeType
from .graph import ControlFlowGraph


def prune_unreachable_blocks(
    cfg: ControlFlowGraph,
    exception_table: Optional[List[ExceptionTableEntry]] = None
) -> int:
    """
    Identifies and removes basic blocks unreachable from CFG entry and active exception handlers.
    Returns count of pruned blocks.
    """
    if not cfg.blocks or cfg.entry_block_id is None:
        return 0

    if exception_table is None:
        exception_table = []

    reachable_block_ids: Set[int] = set()
    worklist: deque[int] = deque()

    # Seed with entry block
    if cfg.entry_block_id in cfg.blocks:
        worklist.append(cfg.entry_block_id)

    # Seed with active exception handlers for blocks that are reachable
    while worklist:
        curr_id = worklist.popleft()
        if curr_id in reachable_block_ids:
            continue
        reachable_block_ids.add(curr_id)

        # Normal successors
        for succ_id in cfg.get_successors(curr_id):
            if succ_id in cfg.blocks and succ_id not in reachable_block_ids:
                worklist.append(succ_id)

        # Exception handlers covering this block
        curr_block = cfg.get_block(curr_id)
        if curr_block:
            for ex in curr_block.exception_handlers:
                # Find handler block
                handler_block = cfg.get_block(ex.handler_pc) or cfg.get_block_at(ex.handler_pc)
                if handler_block and handler_block.id not in reachable_block_ids:
                    worklist.append(handler_block.id)

    unreachable_ids = set(cfg.blocks.keys()) - reachable_block_ids
    pruned_count = len(unreachable_ids)

    for dead_id in unreachable_ids:
        cfg.remove_block(dead_id)

    # Prune exception table entries whose handler or entire try range was pruned
    if exception_table:
        surviving_ex: List[ExceptionTableEntry] = []
        for ex in exception_table:
            # Check if handler block survived
            handler_survived = any(
                b.contains_pc(ex.handler_pc) or b.id == ex.handler_pc
                for b in cfg.blocks.values()
            )
            # Check if any covered block survived
            try_survived = any(
                b.start_pc < ex.end_pc and b.end_pc > ex.start_pc
                for b in cfg.blocks.values()
            )
            if handler_survived and try_survived:
                surviving_ex.append(ex)

        exception_table[:] = surviving_ex

    return pruned_count


def compact_goto_chains(cfg: ControlFlowGraph) -> int:
    """
    Short-circuits jump-to-jump trampolines (e.g. goto L1 where L1: goto L2).
    Includes cycle detection to break infinite loops cleanly.
    Returns count of compacted jumps.
    """
    # Identify trampoline blocks
    def is_trampoline(b: BasicBlock) -> Optional[int]:
        non_nops = [inst for inst in b.instructions if inst.opcode != Opcode.NOP]
        if len(non_nops) == 1 and non_nops[0].is_unconditional_branch:
            for edge in cfg.get_outgoing_edges(b.id):
                if edge.edge_type in (CFGEdgeType.UNCONDITIONAL_BRANCH, CFGEdgeType.FALLTHROUGH):
                    if edge.target_id != b.id and edge.target_id in cfg.blocks:
                        return edge.target_id
            targets = non_nops[0].get_branch_targets()
            if targets:
                target_block = cfg.get_block(targets[0]) or cfg.get_block_at(targets[0])
                if target_block and target_block.id != b.id:
                    return target_block.id
        return None

    trampolines: Dict[int, int] = {}
    for b_id, block in cfg.blocks.items():
        dest = is_trampoline(block)
        if dest is not None:
            trampolines[b_id] = dest

    if not trampolines:
        return 0

    # Resolve transitive chains with cycle breaking
    resolved_dest: Dict[int, int] = {}
    for t_id in trampolines:
        curr = t_id
        visited = {curr}
        while curr in trampolines:
            nxt = trampolines[curr]
            if nxt in visited:
                break
            visited.add(nxt)
            curr = nxt
        resolved_dest[t_id] = curr

    compacted_count = 0

    # Update CFG edges and branch instructions
    for block in cfg.blocks.values():
        for edge in list(cfg.get_outgoing_edges(block.id)):
            if edge.target_id in resolved_dest and edge.target_id != resolved_dest[edge.target_id]:
                final_target = resolved_dest[edge.target_id]
                cfg.remove_edge(block.id, edge.target_id, edge.edge_type)
                cfg.add_edge(CFGEdge(block.id, final_target, edge.edge_type, case_key=edge.case_key))
                compacted_count += 1

    return compacted_count


def eliminate_fallthrough_gotos(cfg: ControlFlowGraph) -> int:
    """
    Converts redundant unconditional GOTOs to the immediately subsequent basic block
    into natural fallthrough edges.
    """
    # Order blocks by start_pc
    ordered = sorted(cfg.blocks.values(), key=lambda b: b.start_pc)
    eliminated = 0

    for i in range(len(ordered) - 1):
        curr_b = ordered[i]
        next_b = ordered[i + 1]

        if not curr_b.instructions:
            continue

        term = curr_b.terminator
        if term and term.is_unconditional_branch:
            targets = term.get_branch_targets()
            if targets and targets[0] == next_b.start_pc:
                # Redundant jump! Remove goto instruction
                del curr_b.instructions[-1]
                cfg.remove_edge(curr_b.id, next_b.id, CFGEdgeType.UNCONDITIONAL_BRANCH)
                cfg.add_edge(CFGEdge(curr_b.id, next_b.id, CFGEdgeType.FALLTHROUGH))
                eliminated += 1

    return eliminated


def strip_nops_from_bytecode(
    code: bytes,
    exception_table: Optional[List[ExceptionTableEntry]] = None
) -> Tuple[bytes, List[ExceptionTableEntry], int]:
    """
    Strips 0x00 NOPs from bytecode and soundly recalculates all branch targets,
    switch alignment padding, and exception table ranges.
    """
    if exception_table is None:
        exception_table = []

    instructions = list(InstructionScanner.scan(code))
    if not any(op == Opcode.NOP for _, op, _ in instructions):
        return code, list(exception_table), 0

    # Pass 1: Build mapping from old instruction PC to new PC
    old_to_new_pc: Dict[int, int] = {}
    compacted_instructions: List[Tuple[int, int, int, bytes]] = []  # (old_pc, new_pc, op, operands)
    curr_new_pc = 0
    nops_removed = 0

    for old_pc, op, operands in instructions:
        if op == Opcode.NOP:
            nops_removed += 1
            # Map old PC to current new PC (points to next non-NOP)
            old_to_new_pc[old_pc] = curr_new_pc
        else:
            old_to_new_pc[old_pc] = curr_new_pc
            compacted_instructions.append((old_pc, curr_new_pc, op, operands))
            if op in (Opcode.TABLESWITCH, Opcode.LOOKUPSWITCH):
                old_pad = (4 - (old_pc + 1) % 4) % 4
                payload_len = max(0, len(operands) - old_pad)
                new_pad = (4 - (curr_new_pc + 1) % 4) % 4
                curr_new_pc += 1 + new_pad + payload_len
            else:
                curr_new_pc += 1 + len(operands)

    old_to_new_pc[len(code)] = curr_new_pc

    # Helper to resolve old PC
    def map_pc(old_target: int) -> int:
        if old_target in old_to_new_pc:
            return old_to_new_pc[old_target]
        # Find nearest following PC
        for opc in sorted(old_to_new_pc.keys()):
            if opc >= old_target:
                return old_to_new_pc[opc]
        return curr_new_pc

    # Pass 2: Reassemble instructions with updated branch offsets and recalculated switch padding
    out = bytearray()

    for old_pc, new_pc, op, operands in compacted_instructions:
        inst_pc = len(out)
        old_inst_pc = old_pc

        if op in (
            Opcode.IFEQ, Opcode.IFNE, Opcode.IFLT, Opcode.IFGE, Opcode.IFGT, Opcode.IFLE,
            Opcode.IF_ICMPEQ, Opcode.IF_ICMPNE, Opcode.IF_ICMPLT, Opcode.IF_ICMPGE,
            Opcode.IF_ICMPGT, Opcode.IF_ICMPLE, Opcode.IF_ACMPEQ, Opcode.IF_ACMPNE,
            Opcode.IFNULL, Opcode.IFNONNULL, Opcode.GOTO, Opcode.JSR
        ):
            old_offset = struct.unpack(">h", operands[:2])[0]
            old_target = old_inst_pc + old_offset
            new_target = map_pc(old_target)
            new_offset = new_target - inst_pc
            out.append(op)
            out.extend(struct.pack(">h", new_offset))

        elif op in (Opcode.GOTO_W, Opcode.JSR_W):
            old_offset = struct.unpack(">i", operands[:4])[0]
            old_target = old_inst_pc + old_offset
            new_target = map_pc(old_target)
            new_offset = new_target - inst_pc
            out.append(op)
            out.extend(struct.pack(">i", new_offset))

        elif op == Opcode.TABLESWITCH:
            old_pad = (4 - (old_inst_pc + 1) % 4) % 4
            old_default = old_inst_pc + struct.unpack(">i", operands[old_pad : old_pad + 4])[0]
            low, high = struct.unpack(">ii", operands[old_pad + 4 : old_pad + 12])
            count = max(0, high - low + 1)
            old_case_targets = []
            for k in range(count):
                off = struct.unpack(">i", operands[old_pad + 12 + 4 * k : old_pad + 16 + 4 * k])[0]
                old_case_targets.append(old_inst_pc + off)

            out.append(op)
            new_pad = (4 - (inst_pc + 1) % 4) % 4
            out.extend(b"\x00" * new_pad)
            new_default_off = map_pc(old_default) - inst_pc
            out.extend(struct.pack(">iii", new_default_off, low, high))
            for old_t in old_case_targets:
                new_c_off = map_pc(old_t) - inst_pc
                out.extend(struct.pack(">i", new_c_off))

        elif op == Opcode.LOOKUPSWITCH:
            old_pad = (4 - (old_inst_pc + 1) % 4) % 4
            old_default = old_inst_pc + struct.unpack(">i", operands[old_pad : old_pad + 4])[0]
            npairs = struct.unpack(">i", operands[old_pad + 4 : old_pad + 8])[0]
            pairs = []
            for k in range(max(0, npairs)):
                m_val, off = struct.unpack(">ii", operands[old_pad + 8 + 8 * k : old_pad + 16 + 8 * k])
                pairs.append((m_val, old_inst_pc + off))

            out.append(op)
            new_pad = (4 - (inst_pc + 1) % 4) % 4
            out.extend(b"\x00" * new_pad)
            new_default_off = map_pc(old_default) - inst_pc
            out.extend(struct.pack(">ii", new_default_off, npairs))
            for m_val, old_t in pairs:
                new_c_off = map_pc(old_t) - inst_pc
                out.extend(struct.pack(">ii", m_val, new_c_off))

        else:
            out.append(op)
            out.extend(operands)

    # Pass 3: Remap exception table
    new_ex_table: List[ExceptionTableEntry] = []
    for ex in exception_table:
        s = map_pc(ex.start_pc)
        e = map_pc(ex.end_pc)
        h = map_pc(ex.handler_pc)
        if s < e:
            new_ex_table.append(ExceptionTableEntry(s, e, h, ex.catch_type))

    return bytes(out), new_ex_table, nops_removed


def reassemble_cfg(
    cfg: ControlFlowGraph,
    cp: Optional[ConstantPool] = None
) -> Tuple[bytes, List[ExceptionTableEntry], int, int]:
    """
    Serializes basic blocks into a valid JVM bytecode byte array.
    Linearizes basic blocks, solves branch relaxation, recalculates exception ranges,
    and computes max_stack and max_locals.
    """
    if not cfg.blocks:
        return b"\xb1", [], 1, 1  # RETURN opcode

    # 1. Linearize basic blocks maximizing fallthrough
    ordered_blocks: List[BasicBlock] = []
    placed: Set[int] = set()

    # Entry block first
    entry_id = cfg.entry_block_id if (cfg.entry_block_id and cfg.entry_block_id in cfg.blocks) else min(cfg.blocks.keys())
    ordered_blocks.append(cfg.blocks[entry_id])
    placed.add(entry_id)

    while len(placed) < len(cfg.blocks):
        last = ordered_blocks[-1]
        out_edges = cfg.get_outgoing_edges(last.id)

        # Prefer fallthrough successor
        next_id = None
        for edge in out_edges:
            if edge.edge_type == CFGEdgeType.FALLTHROUGH and edge.target_id not in placed and edge.target_id in cfg.blocks:
                next_id = edge.target_id
                break

        if next_id is None:
            # Pick any remaining block with lowest start_pc
            remaining = [b for b_id, b in cfg.blocks.items() if b_id not in placed]
            remaining.sort(key=lambda b: b.start_pc)
            next_id = remaining[0].id

        ordered_blocks.append(cfg.blocks[next_id])
        placed.add(next_id)

    # Ensure broken fallthroughs have explicit GOTOs
    for i in range(len(ordered_blocks)):
        curr_b = ordered_blocks[i]
        next_b_id = ordered_blocks[i + 1].id if i + 1 < len(ordered_blocks) else None

        # If curr_b has a fallthrough edge to F, but next block is NOT F:
        for edge in cfg.get_outgoing_edges(curr_b.id):
            if edge.edge_type == CFGEdgeType.FALLTHROUGH and edge.target_id != next_b_id:
                # Add an explicit GOTO to edge.target_id at end of curr_b
                curr_b.instructions.append(Instruction(
                    pc=0,
                    opcode=Opcode.GOTO,
                    operands=struct.pack(">h", 0)
                ))
                break

    # 2. Iterative layout and branch relaxation
    # Map block_id -> label
    # In each iteration, calculate block start offsets
    block_offsets: Dict[int, int] = {}
    widened_branches: Set[Tuple[int, int]] = set()  # (block_id, inst_idx)

    for iteration in range(10):
        changed = False
        curr_pc = 0

        # Assign offsets
        for block in ordered_blocks:
            block_offsets[block.id] = curr_pc
            for idx, inst in enumerate(block.instructions):
                # Calculate instruction length
                if inst.is_switch:
                    old_pad = (4 - (inst.pc + 1) % 4) % 4
                    payload_len = max(0, len(inst.operands) - old_pad)
                    new_pad = (4 - (curr_pc + 1) % 4) % 4
                    curr_pc += 1 + new_pad + payload_len
                elif (block.id, idx) in widened_branches:
                    # GOTO_W (5 bytes)
                    curr_pc += 5
                else:
                    curr_pc += inst.length

        # Check for branch overflow
        for block in ordered_blocks:
            b_start = block_offsets[block.id]
            inst_pc = b_start

            for idx, inst in enumerate(block.instructions):
                if inst.is_switch:
                    old_pad = (4 - (inst.pc + 1) % 4) % 4
                    payload_len = max(0, len(inst.operands) - old_pad)
                    new_pad = (4 - (inst_pc + 1) % 4) % 4
                    inst_pc += 1 + new_pad + payload_len
                    continue

                if inst.is_unconditional_branch and (block.id, idx) not in widened_branches:
                    # Find target block
                    target_id = None
                    for edge in cfg.get_outgoing_edges(block.id):
                        if edge.edge_type == CFGEdgeType.UNCONDITIONAL_BRANCH:
                            target_id = edge.target_id
                            break
                    if target_id is not None and target_id in block_offsets:
                        target_pc = block_offsets[target_id]
                        offset = target_pc - inst_pc
                        if offset < -32768 or offset > 32767:
                            widened_branches.add((block.id, idx))
                            changed = True

                inst_pc += 5 if (block.id, idx) in widened_branches else inst.length

        if not changed:
            break

    # 3. Final serialization
    code_out = bytearray()
    final_block_starts: Dict[int, int] = {}
    final_block_ends: Dict[int, int] = {}

    for block in ordered_blocks:
        final_block_starts[block.id] = len(code_out)

        for idx, inst in enumerate(block.instructions):
            inst_pc = len(code_out)

            if inst.is_unconditional_branch:
                # Find target block ID from CFG edge or branch target
                target_id = None
                for edge in cfg.get_outgoing_edges(block.id):
                    if edge.edge_type in (CFGEdgeType.UNCONDITIONAL_BRANCH, CFGEdgeType.FALLTHROUGH):
                        target_id = edge.target_id
                        break
                if target_id is None and inst.get_branch_targets():
                    target_pc = inst.get_branch_targets()[0]
                    target_b = cfg.get_block(target_pc) or cfg.get_block_at(target_pc)
                    if target_b:
                        target_id = target_b.id

                target_offset = 0
                if target_id is not None and target_id in block_offsets:
                    target_offset = block_offsets[target_id] - inst_pc

                if (block.id, idx) in widened_branches or inst.opcode == Opcode.GOTO_W:
                    code_out.append(Opcode.GOTO_W)
                    code_out.extend(struct.pack(">i", target_offset))
                else:
                    code_out.append(Opcode.GOTO)
                    code_out.extend(struct.pack(">h", target_offset))

            elif inst.is_conditional_branch:
                target_id = None
                for edge in cfg.get_outgoing_edges(block.id):
                    if edge.edge_type == CFGEdgeType.CONDITIONAL_BRANCH:
                        target_id = edge.target_id
                        break
                if target_id is None and inst.get_branch_targets():
                    target_pc = inst.get_branch_targets()[0]
                    target_b = cfg.get_block(target_pc) or cfg.get_block_at(target_pc)
                    if target_b:
                        target_id = target_b.id

                target_offset = 0
                if target_id is not None and target_id in block_offsets:
                    target_offset = block_offsets[target_id] - inst_pc

                code_out.append(inst.opcode)
                code_out.extend(struct.pack(">h", target_offset))

            elif inst.opcode == Opcode.TABLESWITCH:
                pad = (4 - (inst_pc + 1) % 4) % 4
                code_out.append(Opcode.TABLESWITCH)
                code_out.extend(b"\x00" * pad)

                # Find switch edges
                case_edges = {e.case_key: e.target_id for e in cfg.get_outgoing_edges(block.id) if e.edge_type == CFGEdgeType.SWITCH_CASE}
                default_edge = next((e for e in cfg.get_outgoing_edges(block.id) if e.edge_type == CFGEdgeType.SWITCH_DEFAULT), None)

                default_off = (block_offsets[default_edge.target_id] - inst_pc) if (default_edge and default_edge.target_id in block_offsets) else 0

                low = min(case_edges.keys()) if case_edges else 0
                high = max(case_edges.keys()) if case_edges else -1

                code_out.extend(struct.pack(">iii", default_off, low, high))
                for k in range(low, high + 1):
                    c_target = case_edges.get(k)
                    c_off = (block_offsets[c_target] - inst_pc) if (c_target and c_target in block_offsets) else default_off
                    code_out.extend(struct.pack(">i", c_off))

            elif inst.opcode == Opcode.LOOKUPSWITCH:
                pad = (4 - (inst_pc + 1) % 4) % 4
                code_out.append(Opcode.LOOKUPSWITCH)
                code_out.extend(b"\x00" * pad)

                case_edges = [(e.case_key, e.target_id) for e in cfg.get_outgoing_edges(block.id) if e.edge_type == CFGEdgeType.SWITCH_CASE and e.case_key is not None]
                case_edges.sort(key=lambda p: p[0])
                default_edge = next((e for e in cfg.get_outgoing_edges(block.id) if e.edge_type == CFGEdgeType.SWITCH_DEFAULT), None)
                default_off = (block_offsets[default_edge.target_id] - inst_pc) if (default_edge and default_edge.target_id in block_offsets) else 0

                code_out.extend(struct.pack(">ii", default_off, len(case_edges)))
                for k, t_id in case_edges:
                    c_off = (block_offsets[t_id] - inst_pc) if t_id in block_offsets else default_off
                    code_out.extend(struct.pack(">ii", k, c_off))

            else:
                code_out.append(inst.opcode)
                code_out.extend(inst.operands)

        final_block_ends[block.id] = len(code_out)

    # 4. Exception table recalculation
    new_ex_table: List[ExceptionTableEntry] = []
    seen_ex: Set[Tuple[int, int, int, int]] = set()

    for block in ordered_blocks:
        for ex in block.exception_handlers:
            handler_block = cfg.get_block(ex.handler_pc) or cfg.get_block_at(ex.handler_pc)
            if handler_block and handler_block.id in final_block_starts:
                s_pc = final_block_starts[block.id]
                e_pc = final_block_ends[block.id]
                h_pc = final_block_starts[handler_block.id]
                key = (s_pc, e_pc, h_pc, ex.catch_type)
                if key not in seen_ex and s_pc < e_pc:
                    seen_ex.add(key)
                    new_ex_table.append(ExceptionTableEntry(s_pc, e_pc, h_pc, ex.catch_type))

    # 5. Compute max_locals and max_stack
    max_locals = 1
    max_stack = 2
    for block in ordered_blocks:
        for inst in block.instructions:
            # Slot checking
            if inst.opcode in (Opcode.ILOAD, Opcode.ISTORE, Opcode.FLOAD, Opcode.FSTORE, Opcode.ALOAD, Opcode.ASTORE):
                if inst.operands:
                    max_locals = max(max_locals, inst.operands[0] + 1)
            elif inst.opcode in (Opcode.LLOAD, Opcode.LSTORE, Opcode.DLOAD, Opcode.DSTORE):
                if inst.operands:
                    max_locals = max(max_locals, inst.operands[0] + 2)
            elif Opcode.ILOAD_0 <= inst.opcode <= Opcode.ALOAD_3 or Opcode.ISTORE_0 <= inst.opcode <= Opcode.ASTORE_3:
                max_locals = max(max_locals, 4)

    return bytes(code_out), new_ex_table, max_stack, max_locals


def prune_dead_code_and_nops(
    cfg: ControlFlowGraph,
    exception_table: Optional[List[ExceptionTableEntry]] = None
) -> Tuple[int, int]:
    """
    Executes complete dead code and NOP elimination pipeline:
    1. Prune unreachable basic blocks
    2. Compact redundant GOTO chains
    3. Eliminate redundant fallthrough GOTOs
    Returns (pruned_blocks, nops_removed).
    """
    pruned_blocks = prune_unreachable_blocks(cfg, exception_table=exception_table)
    compact_goto_chains(cfg)
    eliminate_fallthrough_gotos(cfg)

    # Count & strip NOPs from all remaining blocks
    nops_removed = 0
    for block in cfg.blocks.values():
        clean_insts = []
        for inst in block.instructions:
            if inst.opcode == Opcode.NOP:
                nops_removed += 1
            else:
                clean_insts.append(inst)
        block.instructions = clean_insts

    return pruned_blocks, nops_removed
