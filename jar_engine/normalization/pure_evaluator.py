"""
jar_engine.normalization.pure_evaluator: Inter-procedural static evaluation of pure mathematical
functions and symbolic opaque predicate/switch simplification (specifically targeting Skidfuscator/ZKM).
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Set, Optional, Any
import struct

from jar_engine.model.constants import Opcode, AccessFlags
from jar_engine.model.cp_info import (
    ConstantPool, CONSTANT_Integer_info, CONSTANT_Long_info,
    CONSTANT_String_info, CONSTANT_Class_info
)
from jar_engine.model.member_info import MethodInfo, CodeAttribute, ExceptionTableEntry
from jar_engine.model.class_file import ClassFile
from jar_engine.model.jar_archive import JarArchive
from jar_engine.walker.call_graph import InstructionScanner
from jar_engine.cfg.basic_block import BasicBlock, Instruction, CFGEdge, CFGEdgeType
from jar_engine.cfg.graph import ControlFlowGraph
from jar_engine.cfg.dead_code import prune_unreachable_blocks, compact_goto_chains
from jar_engine.normalization.string_decryptor import BytecodeEmulator


def get_const_int_val(inst: Instruction, cp: Optional[ConstantPool] = None) -> Optional[int]:
    """Extracts signed 32-bit integer constant value from an instruction."""
    if Opcode.ICONST_M1 <= inst.opcode <= Opcode.ICONST_5:
        return inst.opcode - Opcode.ICONST_0
    elif inst.opcode == Opcode.BIPUSH and len(inst.operands) >= 1:
        return struct.unpack(">b", inst.operands[:1])[0]
    elif inst.opcode == Opcode.SIPUSH and len(inst.operands) >= 2:
        return struct.unpack(">h", inst.operands[:2])[0]
    elif inst.opcode == Opcode.LDC and cp and len(inst.operands) >= 1:
        entry = cp.get(inst.operands[0])
        if isinstance(entry, CONSTANT_Integer_info):
            return entry.value
    elif inst.opcode == Opcode.LDC_W and cp and len(inst.operands) >= 2:
        idx = struct.unpack(">H", inst.operands[:2])[0]
        entry = cp.get(idx)
        if isinstance(entry, CONSTANT_Integer_info):
            return entry.value
    return None


def get_const_str_val(inst: Instruction, cp: Optional[ConstantPool] = None) -> Optional[str]:
    """Extracts string constant value from an instruction."""
    if not cp:
        return None
    if inst.opcode == Opcode.LDC and len(inst.operands) >= 1:
        idx = inst.operands[0]
        entry = cp.get(idx)
        if isinstance(entry, CONSTANT_String_info):
            return cp.resolve_string(idx)
    elif inst.opcode == Opcode.LDC_W and len(inst.operands) >= 2:
        idx = struct.unpack(">H", inst.operands[:2])[0]
        entry = cp.get(idx)
        if isinstance(entry, CONSTANT_String_info):
            return cp.resolve_string(idx)
    return None


class PureMethodScanner:
    """Discovers side-effect-free static helper methods across a JarArchive."""

    @classmethod
    def scan_archive(cls, archive: JarArchive) -> Dict[Tuple[str, str, str], Tuple[ClassFile, MethodInfo]]:
        pure_methods: Dict[Tuple[str, str, str], Tuple[ClassFile, MethodInfo]] = {}

        for c_name, cf in archive.classes.items():
            cp = cf.constant_pool
            for m in cf.methods:
                if not (m.access_flags & AccessFlags.STATIC):
                    continue
                code_attr = m.code_attribute(cp)
                if not code_attr or not code_attr.code or len(code_attr.code) > 250:
                    continue

                # Ensure method has no heap/IO side effects
                has_forbidden = False
                for pc, op, operands in InstructionScanner.scan(code_attr.code):
                    if op in (Opcode.PUTSTATIC, Opcode.PUTFIELD, Opcode.ATHROW, Opcode.MONITORENTER, Opcode.NEW):
                        has_forbidden = True
                        break
                if has_forbidden:
                    continue

                desc = m.descriptor(cp)
                if desc in ("(I)I", "(II)I", "(J)J", "(JJ)J", "(I)Z", "(II)Z"):
                    m_name = m.name(cp)
                    pure_methods[(c_name, m_name, desc)] = (cf, m)

        return pure_methods


class PureMethodEvaluator:
    """
    Symbolically executes deterministic state machines and unrolls Skidfuscator/ZKM
    inter-procedural opaque predicates and switch dispatchers using forward dataflow analysis.
    """

    def __init__(self, pure_methods: Optional[Dict[Tuple[str, str, str], Tuple[ClassFile, MethodInfo]]] = None):
        self.pure_methods = pure_methods or {}
        self.emulator = BytecodeEmulator()

    def simplify_cfg(
        self,
        cfg: ControlFlowGraph,
        cf: ClassFile,
        cp: ConstantPool,
        exception_table: Optional[List[ExceptionTableEntry]] = None
    ) -> int:
        """
        Evaluates pure invocations and collapses guarded branches and switches.
        Returns the number of simplified predicates/switches.
        """
        simplified_count = 0
        if not cfg.blocks or cfg.entry_block_id is None:
            return 0

        # Iterative forward dataflow tracking constant local variables
        # Map: block_id -> out_locals
        block_out_locals: Dict[int, Dict[int, int]] = {}

        for iteration in range(25):
            changed = False
            for b_id in list(cfg.blocks.keys()):
                if b_id not in cfg.blocks:
                    continue
                block = cfg.blocks[b_id]

                # Sound meet operator (intersection over all non-exception predecessors)
                preds = [e.source_id for e in cfg.get_incoming_edges(b_id) if e.edge_type != CFGEdgeType.EXCEPTION and e.source_id in cfg.blocks]
                if b_id == cfg.entry_block_id:
                    cur_locals: Dict[int, int] = {}
                elif not preds:
                    cur_locals = {}
                elif all(p in block_out_locals for p in preds):
                    common_slots = set(block_out_locals[preds[0]].keys())
                    for p in preds[1:]:
                        common_slots &= set(block_out_locals[p].keys())
                    cur_locals = {
                        s: block_out_locals[preds[0]][s]
                        for s in common_slots
                        if all(block_out_locals[p][s] == block_out_locals[preds[0]][s] for p in preds)
                    }
                else:
                    cur_locals = {}

                insts = list(block.instructions)

                # Intra-block symbolic simplification pass
                i = 0
                while i < len(insts):
                    inst = insts[i]

                    # Propagate known local: iload slot -> const_push
                    if inst.opcode == Opcode.ILOAD or (Opcode.ILOAD_0 <= inst.opcode <= Opcode.ILOAD_3):
                        slot = inst.opcode - Opcode.ILOAD_0 if inst.opcode != Opcode.ILOAD else inst.operands[0]
                        if slot in cur_locals:
                            np = self._create_int_push(cur_locals[slot], cp)
                            if np:
                                insts[i] = np
                                changed = True
                                continue

                    # Track integer stores: const_push; istore slot
                    elif inst.opcode == Opcode.ISTORE or (Opcode.ISTORE_0 <= inst.opcode <= Opcode.ISTORE_3):
                        slot = inst.opcode - Opcode.ISTORE_0 if inst.opcode != Opcode.ISTORE else inst.operands[0]
                        if i >= 1:
                            v = get_const_int_val(insts[i - 1], cp)
                            if v is not None:
                                cur_locals[slot] = v
                            else:
                                cur_locals.pop(slot, None)
                        else:
                            cur_locals.pop(slot, None)

                    # Invalidate mutated variables on iinc
                    elif inst.opcode == Opcode.IINC and len(inst.operands) >= 2:
                        slot = inst.operands[0]
                        cur_locals.pop(slot, None)

                    # Invalidate on other variable stores
                    elif (Opcode.LSTORE_0 <= inst.opcode <= Opcode.ASTORE_3) or inst.opcode in (
                        Opcode.LSTORE, Opcode.FSTORE, Opcode.DSTORE, Opcode.ASTORE
                    ):
                        if inst.opcode in (Opcode.LSTORE, Opcode.FSTORE, Opcode.DSTORE, Opcode.ASTORE) and len(inst.operands) >= 1:
                            slot = inst.operands[0]
                        else:
                            slot = (inst.opcode - Opcode.LSTORE_0) % 4
                        cur_locals.pop(slot, None)
                        cur_locals.pop(slot + 1, None)

                    # Evaluate Integer.parseInt("123")
                    elif inst.opcode == Opcode.INVOKESTATIC and len(inst.operands) >= 2:
                        ref_idx = struct.unpack(">H", inst.operands[:2])[0]
                        try:
                            t_cls, t_name, t_desc = cp.resolve_member_ref(ref_idx)
                            if t_cls == "java/lang/Integer" and t_name == "parseInt" and i >= 1:
                                s_val = get_const_str_val(insts[i - 1], cp)
                                if s_val is not None:
                                    parsed = int(s_val)
                                    np = self._create_int_push(parsed, cp)
                                    if np:
                                        insts[i - 1 : i + 1] = [np]
                                        i -= 1
                                        simplified_count += 1
                                        changed = True
                                        continue
                            elif (t_cls, t_name, t_desc) in self.pure_methods:
                                target_cf, target_m = self.pure_methods[(t_cls, t_name, t_desc)]
                                if t_desc in ("(I)I", "(I)Z") and i >= 1:
                                    arg_val = get_const_int_val(insts[i - 1], cp)
                                    if arg_val is not None:
                                        res = self.emulator.execute(target_cf, target_m, [arg_val])
                                        if res is not None:
                                            np = self._create_int_push(res, cp)
                                            if np:
                                                insts[i - 1 : i + 1] = [np]
                                                i -= 1
                                                simplified_count += 1
                                                changed = True
                                                continue
                                elif t_desc in ("(II)I", "(II)Z") and i >= 2:
                                    v1 = get_const_int_val(insts[i - 2], cp)
                                    v2 = get_const_int_val(insts[i - 1], cp)
                                    if v1 is not None and v2 is not None:
                                        res = self.emulator.execute(target_cf, target_m, [v1, v2])
                                        if res is not None:
                                            np = self._create_int_push(res, cp)
                                            if np:
                                                insts[i - 2 : i + 1] = [np]
                                                i -= 2
                                                simplified_count += 1
                                                changed = True
                                                continue
                        except Exception:
                            pass

                    # Evaluate binary arithmetic
                    elif inst.opcode in (
                        Opcode.IXOR, Opcode.IADD, Opcode.ISUB, Opcode.IMUL,
                        Opcode.IAND, Opcode.IOR, Opcode.ISHL, Opcode.ISHR, Opcode.IUSHR
                    ) and i >= 2:
                        v1 = get_const_int_val(insts[i - 2], cp)
                        v2 = get_const_int_val(insts[i - 1], cp)
                        if v1 is not None and v2 is not None:
                            if inst.opcode == Opcode.IXOR:
                                r = v1 ^ v2
                            elif inst.opcode == Opcode.IADD:
                                r = v1 + v2
                            elif inst.opcode == Opcode.ISUB:
                                r = v1 - v2
                            elif inst.opcode == Opcode.IMUL:
                                r = v1 * v2
                            elif inst.opcode == Opcode.IAND:
                                r = v1 & v2
                            elif inst.opcode == Opcode.IOR:
                                r = v1 | v2
                            elif inst.opcode == Opcode.ISHL:
                                r = v1 << (v2 & 0x1F)
                            elif inst.opcode == Opcode.ISHR:
                                r = v1 >> (v2 & 0x1F)
                            else:
                                r = (v1 & 0xFFFFFFFF) >> (v2 & 0x1F)
                            np = self._create_int_push(r, cp)
                            if np:
                                insts[i - 2 : i + 1] = [np]
                                i -= 2
                                simplified_count += 1
                                changed = True
                                continue

                    i += 1

                block.instructions = insts
                block_out_locals[b_id] = cur_locals

                # Branch Folding 1: Binary Comparisons (if_icmpeq, etc.)
                if len(insts) >= 3 and insts[-1].opcode in (
                    Opcode.IF_ICMPEQ, Opcode.IF_ICMPNE, Opcode.IF_ICMPLT,
                    Opcode.IF_ICMPGE, Opcode.IF_ICMPGT, Opcode.IF_ICMPLE
                ):
                    v1 = get_const_int_val(insts[-3], cp)
                    v2 = get_const_int_val(insts[-2], cp)
                    if v1 is not None and v2 is not None:
                        term = insts[-1]
                        if term.opcode == Opcode.IF_ICMPEQ:
                            cond_met = (v1 == v2)
                        elif term.opcode == Opcode.IF_ICMPNE:
                            cond_met = (v1 != v2)
                        elif term.opcode == Opcode.IF_ICMPLT:
                            cond_met = (v1 < v2)
                        elif term.opcode == Opcode.IF_ICMPGE:
                            cond_met = (v1 >= v2)
                        elif term.opcode == Opcode.IF_ICMPGT:
                            cond_met = (v1 > v2)
                        else:
                            cond_met = (v1 <= v2)

                        out_edges = cfg.get_outgoing_edges(b_id)
                        branch_edge = next((e for e in out_edges if e.edge_type == CFGEdgeType.CONDITIONAL_BRANCH), None)
                        if branch_edge:
                            if cond_met:
                                block.instructions = insts[:-3] + [Instruction(0, Opcode.GOTO, struct.pack(">h", 0))]
                                for e in list(out_edges):
                                    if e.edge_type != CFGEdgeType.EXCEPTION:
                                        cfg.remove_edge(e.source_id, e.target_id, e.edge_type)
                                cfg.add_edge(CFGEdge(b_id, branch_edge.target_id, CFGEdgeType.UNCONDITIONAL_BRANCH))
                            else:
                                block.instructions = insts[:-3]
                                cfg.remove_edge(b_id, branch_edge.target_id, CFGEdgeType.CONDITIONAL_BRANCH)
                            simplified_count += 1
                            changed = True

                # Branch Folding 2: Unary Comparisons (ifeq, ifne, etc.)
                elif len(insts) >= 2 and insts[-1].opcode in (
                    Opcode.IFEQ, Opcode.IFNE, Opcode.IFLT,
                    Opcode.IFGE, Opcode.IFGT, Opcode.IFLE
                ):
                    v = get_const_int_val(insts[-2], cp)
                    if v is not None:
                        term = insts[-1]
                        if term.opcode == Opcode.IFEQ:
                            cond_met = (v == 0)
                        elif term.opcode == Opcode.IFNE:
                            cond_met = (v != 0)
                        elif term.opcode == Opcode.IFLT:
                            cond_met = (v < 0)
                        elif term.opcode == Opcode.IFGE:
                            cond_met = (v >= 0)
                        elif term.opcode == Opcode.IFGT:
                            cond_met = (v > 0)
                        else:
                            cond_met = (v <= 0)

                        out_edges = cfg.get_outgoing_edges(b_id)
                        branch_edge = next((e for e in out_edges if e.edge_type == CFGEdgeType.CONDITIONAL_BRANCH), None)
                        if branch_edge:
                            if cond_met:
                                block.instructions = insts[:-2] + [Instruction(0, Opcode.GOTO, struct.pack(">h", 0))]
                                for e in list(out_edges):
                                    if e.edge_type != CFGEdgeType.EXCEPTION:
                                        cfg.remove_edge(e.source_id, e.target_id, e.edge_type)
                                cfg.add_edge(CFGEdge(b_id, branch_edge.target_id, CFGEdgeType.UNCONDITIONAL_BRANCH))
                            else:
                                block.instructions = insts[:-2]
                                cfg.remove_edge(b_id, branch_edge.target_id, CFGEdgeType.CONDITIONAL_BRANCH)
                            simplified_count += 1
                            changed = True

                # Switch Folding: tableswitch / lookupswitch
                elif len(insts) >= 2 and insts[-1].is_switch:
                    cv = get_const_int_val(insts[-2], cp)
                    if cv is not None:
                        out_edges = cfg.get_outgoing_edges(b_id)
                        target_id: Optional[int] = None
                        default_target: Optional[int] = None
                        for edge in out_edges:
                            if edge.edge_type == CFGEdgeType.SWITCH_CASE and edge.case_key == cv:
                                target_id = edge.target_id
                                break
                            elif edge.edge_type == CFGEdgeType.SWITCH_DEFAULT:
                                default_target = edge.target_id
                        if target_id is None:
                            target_id = default_target
                        if target_id is not None:
                            block.instructions = insts[:-2] + [Instruction(0, Opcode.GOTO, struct.pack(">h", 0))]
                            for e in list(out_edges):
                                if e.edge_type != CFGEdgeType.EXCEPTION:
                                    cfg.remove_edge(e.source_id, e.target_id, e.edge_type)
                            cfg.add_edge(CFGEdge(b_id, target_id, CFGEdgeType.UNCONDITIONAL_BRANCH))
                            simplified_count += 1
                            changed = True

            if changed:
                prune_unreachable_blocks(cfg, exception_table=exception_table)
                compact_goto_chains(cfg)
            else:
                break

        return simplified_count

    def _create_int_push(self, val: int, cp: ConstantPool) -> Optional[Instruction]:
        v = int(val)
        v = (v & 0xFFFFFFFF)
        if v >= 0x80000000:
            v -= 0x100000000
        if -1 <= v <= 5:
            return Instruction(0, Opcode.ICONST_0 + v)
        elif -128 <= v <= 127:
            return Instruction(0, Opcode.BIPUSH, struct.pack(">b", v))
        elif -32768 <= v <= 32767:
            return Instruction(0, Opcode.SIPUSH, struct.pack(">h", v))
        else:
            idx = cp.add_integer(v)
            if idx <= 255:
                return Instruction(0, Opcode.LDC, bytes([idx]))
            else:
                return Instruction(0, Opcode.LDC_W, struct.pack(">H", idx))
