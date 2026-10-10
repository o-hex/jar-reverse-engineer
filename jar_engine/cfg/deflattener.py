"""
jar_engine.cfg.deflattener: Control Flow De-Flattening (F14) and Opaque Predicate Elimination (F15).
Reverses state-variable switch/goto dispatch loops and eliminates algebraic & constant invariants.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any
import struct

from jar_engine.model.constants import Opcode
from .basic_block import BasicBlock, Instruction, CFGEdge, CFGEdgeType
from .graph import ControlFlowGraph, NaturalLoop


@dataclass
class DeflatteningReport:
    loops_detected: int = 0
    dispatchers_eliminated: int = 0
    edges_reconstructed: int = 0
    complexity_before: int = 0
    complexity_after: int = 0


@dataclass
class OpaquePredicateReport:
    algebraic_predicates_pruned: int = 0
    constant_predicates_pruned: int = 0
    unreachable_branches_removed: int = 0


# --- Helper functions for instruction inspection ---

def is_const_int_push(inst: Instruction) -> bool:
    return (
        Opcode.ICONST_M1 <= inst.opcode <= Opcode.ICONST_5
        or inst.opcode in (Opcode.BIPUSH, Opcode.SIPUSH)
    )


def get_const_int_value(inst: Instruction) -> Optional[int]:
    if Opcode.ICONST_M1 <= inst.opcode <= Opcode.ICONST_5:
        return inst.opcode - Opcode.ICONST_0
    elif inst.opcode == Opcode.BIPUSH and len(inst.operands) >= 1:
        return struct.unpack(">b", inst.operands[:1])[0]
    elif inst.opcode == Opcode.SIPUSH and len(inst.operands) >= 2:
        return struct.unpack(">h", inst.operands[:2])[0]
    return None


def is_istore(inst: Instruction) -> bool:
    return (
        Opcode.ISTORE_0 <= inst.opcode <= Opcode.ISTORE_3
        or inst.opcode == Opcode.ISTORE
    )


def get_istore_slot(inst: Instruction) -> Optional[int]:
    if Opcode.ISTORE_0 <= inst.opcode <= Opcode.ISTORE_3:
        return inst.opcode - Opcode.ISTORE_0
    elif inst.opcode == Opcode.ISTORE and len(inst.operands) >= 1:
        return inst.operands[0]
    return None


def is_iload(inst: Instruction) -> bool:
    return (
        Opcode.ILOAD_0 <= inst.opcode <= Opcode.ILOAD_3
        or inst.opcode == Opcode.ILOAD
    )


def get_iload_slot(inst: Instruction) -> Optional[int]:
    if Opcode.ILOAD_0 <= inst.opcode <= Opcode.ILOAD_3:
        return inst.opcode - Opcode.ILOAD_0
    elif inst.opcode == Opcode.ILOAD and len(inst.operands) >= 1:
        return inst.operands[0]
    return None


def is_nonnegative_mask_constant(inst: Instruction, cp: Optional[Any] = None) -> bool:
    """Checks if instruction pushes 0x7FFFFFFF (non-negative signed integer mask)."""
    val = get_const_int_value(inst)
    if val is not None:
        return val == 0x7FFFFFFF
    if inst.opcode in (Opcode.LDC, Opcode.LDC_W):
        if cp is not None:
            idx = inst.operands[0] if inst.opcode == Opcode.LDC else struct.unpack(">H", inst.operands[:2])[0]
            entry = cp.get(idx)
            return entry is not None and getattr(entry, "value", None) == 0x7FFFFFFF
        return False
    return False


class ControlFlowDeflattener:
    """
    Automated unflattening engine for state-variable switch/goto dispatch loops.
    """

    def __init__(self, cfg: ControlFlowGraph):
        self.cfg = cfg

    def deflatten(self) -> DeflatteningReport:
        report = DeflatteningReport()
        report.complexity_before = self.cfg.cyclomatic_complexity

        # Find candidate dispatchers
        # 1. Natural loops with switch blocks
        loops = self.cfg.find_natural_loops()
        dispatchers_found: Set[int] = set()

        for loop in loops:
            for b_id in loop.body:
                block = self.cfg.get_block(b_id)
                if not block or not block.terminator:
                    continue
                if block.terminator.is_switch and b_id not in dispatchers_found:
                    # Check in-degree or loop membership
                    in_edges = self.cfg.get_incoming_edges(b_id)
                    if len(in_edges) >= 2 or len(loop.body) >= 3:
                        if self._process_dispatcher(block, loop, report):
                            dispatchers_found.add(b_id)

        # Fallback: scan any switch block with in-degree >= 3 even if not in natural loop
        for b_id, block in list(self.cfg.blocks.items()):
            if b_id in dispatchers_found or not block.terminator or not block.terminator.is_switch:
                continue
            in_edges = self.cfg.get_incoming_edges(b_id)
            if len(in_edges) >= 3:
                # Synthesize a dummy loop with all predecessors
                dummy_loop = NaturalLoop(header_id=b_id, latch_id=b_id, body=set(self.cfg.get_predecessors(b_id)) | {b_id})
                if self._process_dispatcher(block, dummy_loop, report):
                    dispatchers_found.add(b_id)

        # Eliminate isolated dispatcher blocks
        for d_id in dispatchers_found:
            in_preds = self.cfg.get_predecessors(d_id)
            if not in_preds:
                self.cfg.remove_block(d_id)
                report.dispatchers_eliminated += 1

        report.complexity_after = self.cfg.cyclomatic_complexity
        return report

    def _process_dispatcher(self, disp_block: BasicBlock, loop: NaturalLoop, report: DeflatteningReport) -> bool:
        """
        Unflattens a single switch dispatcher block.
        """
        terminator = disp_block.terminator
        if not terminator or not terminator.is_switch:
            return False

        # Identify state variable from instructions before switch
        state_var: Optional[int] = None
        for inst in reversed(disp_block.instructions[:-1]):
            slot = get_iload_slot(inst)
            if slot is not None:
                state_var = slot
                break

        # Extract case mapping: state_value -> target_block_id
        case_map: Dict[int, int] = {}
        default_target_id: Optional[int] = None

        for edge in self.cfg.get_outgoing_edges(disp_block.id):
            if edge.edge_type == CFGEdgeType.SWITCH_CASE and edge.case_key is not None:
                case_map[edge.case_key] = edge.target_id
            elif edge.edge_type == CFGEdgeType.SWITCH_DEFAULT:
                default_target_id = edge.target_id

        if not case_map:
            return False

        report.loops_detected += 1
        edges_redirected = 0

        # Find all blocks jumping into the dispatcher or loop header
        predecessors = list(self.cfg.get_predecessors(disp_block.id))
        if loop.header_id != disp_block.id:
            predecessors.extend(list(self.cfg.get_predecessors(loop.header_id)))
        predecessors = list(set(predecessors) - {disp_block.id})

        for pred_id in predecessors:
            pred_block = self.cfg.get_block(pred_id)
            if not pred_block or not pred_block.instructions:
                continue

            # Look for state store before jump:
            # Pattern: const_push; istore [state_var]; goto disp
            insts = pred_block.instructions
            found_state_update = False

            # Check last 3-4 instructions
            for k in range(len(insts) - 1, 0, -1):
                if is_istore(insts[k]):
                    store_slot = get_istore_slot(insts[k])
                    if state_var is None or store_slot == state_var:
                        # Check preceding instruction for constant push
                        prev_inst = insts[k - 1]
                        val = get_const_int_value(prev_inst)
                        if val is not None and val in case_map:
                            next_target_id = case_map[val]
                            # Rewrite direct jump
                            found_state_update = True

                            # Remove the push and store instructions
                            del insts[k]
                            del insts[k - 1]

                            # Update terminator
                            if insts and insts[-1].is_unconditional_branch:
                                # Replace with GOTO to next_target_id
                                # We pack GOTO with offset 0; reassembler calculates offsets
                                insts[-1] = Instruction(
                                    pc=insts[-1].pc,
                                    opcode=Opcode.GOTO,
                                    operands=struct.pack(">h", 0)
                                )
                            else:
                                pc = insts[-1].pc if insts else pred_block.start_pc
                                insts.append(Instruction(
                                    pc=pc,
                                    opcode=Opcode.GOTO,
                                    operands=struct.pack(">h", 0)
                                ))

                            # Rewire CFG edges
                            self.cfg.remove_edge(pred_id, disp_block.id)
                            if loop.header_id != disp_block.id:
                                self.cfg.remove_edge(pred_id, loop.header_id)
                            self.cfg.add_edge(CFGEdge(pred_id, next_target_id, CFGEdgeType.UNCONDITIONAL_BRANCH))
                            edges_redirected += 1
                            break

            if found_state_update:
                report.edges_reconstructed += 1

        # Check entry/initialization block entering loop from outside
        # Often sets initial state: bipush S_init; istore state_var; goto disp
        for b_id, block in self.cfg.blocks.items():
            if b_id in loop.body or not block.instructions:
                continue
            out_ids = self.cfg.get_successors(b_id)
            if disp_block.id in out_ids or loop.header_id in out_ids:
                insts = block.instructions
                for k in range(len(insts) - 1, 0, -1):
                    if is_istore(insts[k]):
                        store_slot = get_istore_slot(insts[k])
                        if state_var is None or store_slot == state_var:
                            val = get_const_int_value(insts[k - 1])
                            if val is not None and val in case_map:
                                init_target = case_map[val]
                                del insts[k]
                                del insts[k - 1]
                                if insts and insts[-1].is_unconditional_branch:
                                    insts[-1] = Instruction(
                                        pc=insts[-1].pc,
                                        opcode=Opcode.GOTO,
                                        operands=struct.pack(">h", 0)
                                    )
                                else:
                                    pc = insts[-1].pc if insts else block.start_pc
                                    insts.append(Instruction(
                                        pc=pc,
                                        opcode=Opcode.GOTO,
                                        operands=struct.pack(">h", 0)
                                    ))
                                self.cfg.remove_edge(b_id, disp_block.id)
                                if loop.header_id != disp_block.id:
                                    self.cfg.remove_edge(b_id, loop.header_id)
                                self.cfg.add_edge(CFGEdge(b_id, init_target, CFGEdgeType.UNCONDITIONAL_BRANCH))
                                report.edges_reconstructed += 1
                                break

        return edges_redirected > 0


class OpaquePredicateEliminator:
    """
    Detects and eliminates opaque predicates (algebraic invariants & constant comparisons),
    strictly preserving operand stack neutrality.
    """

    def __init__(self, cfg: ControlFlowGraph, cp: Optional[Any] = None):
        self.cfg = cfg
        self.cp = cp

    def eliminate(self) -> OpaquePredicateReport:
        report = OpaquePredicateReport()

        for b_id, block in list(self.cfg.blocks.items()):
            if not block.instructions:
                continue
            terminator = block.terminator
            if not terminator or not terminator.is_conditional_branch:
                continue

            # Case 1: Constant Comparison Predicates
            if self._try_eliminate_constant_predicate(block, report):
                continue

            # Case 2: Algebraic Invariants
            self._try_eliminate_algebraic_predicate(block, report)

        return report

    def _try_eliminate_constant_predicate(self, block: BasicBlock, report: OpaquePredicateReport) -> bool:
        """
        Eliminates branches comparing constant values directly.
        Examples:
        - iconst_0; ifeq L (Always True)
        - iconst_1; ifeq L (Always False)
        - iconst_0; ifne L (Always False)
        - iconst_1; ifne L (Always True)
        - bipush C1; bipush C2; if_icmpeq L
        - aconst_null; ifnull L (Always True)
        - aconst_null; ifnonnull L (Always False)
        """
        insts = block.instructions
        if len(insts) < 2:
            return False

        term = insts[-1]
        out_edges = self.cfg.get_outgoing_edges(block.id)
        branch_edge = next((e for e in out_edges if e.edge_type == CFGEdgeType.CONDITIONAL_BRANCH), None)
        fallthrough_edge = next((e for e in out_edges if e.edge_type == CFGEdgeType.FALLTHROUGH), None)

        if not branch_edge:
            return False

        # Pattern: iconst_C / bipush C; ifeq / ifne / iflt / ifge / ifgt / ifle
        prev = insts[-2]
        if term.opcode in (Opcode.IFEQ, Opcode.IFNE, Opcode.IFLT, Opcode.IFGE, Opcode.IFGT, Opcode.IFLE):
            val = get_const_int_value(prev)
            if val is not None:
                condition_met: bool
                if term.opcode == Opcode.IFEQ:
                    condition_met = (val == 0)
                elif term.opcode == Opcode.IFNE:
                    condition_met = (val != 0)
                elif term.opcode == Opcode.IFLT:
                    condition_met = (val < 0)
                elif term.opcode == Opcode.IFGE:
                    condition_met = (val >= 0)
                elif term.opcode == Opcode.IFGT:
                    condition_met = (val > 0)
                elif term.opcode == Opcode.IFLE:
                    condition_met = (val <= 0)
                else:
                    return False

                # Prune producer instruction (prev)
                del insts[-2]
                self._apply_branch_pruning(block, branch_edge, fallthrough_edge, condition_met)
                report.constant_predicates_pruned += 1
                return True

        # Pattern: aconst_null; ifnull / ifnonnull
        if term.opcode in (Opcode.IFNULL, Opcode.IFNONNULL) and prev.opcode == Opcode.ACONST_NULL:
            condition_met = (term.opcode == Opcode.IFNULL)
            del insts[-2]
            self._apply_branch_pruning(block, branch_edge, fallthrough_edge, condition_met)
            report.constant_predicates_pruned += 1
            return True

        # Pattern: const C1; const C2; if_icmp*
        if len(insts) >= 3 and term.opcode in (
            Opcode.IF_ICMPEQ, Opcode.IF_ICMPNE, Opcode.IF_ICMPLT,
            Opcode.IF_ICMPGE, Opcode.IF_ICMPGT, Opcode.IF_ICMPLE
        ):
            c1_inst = insts[-3]
            c2_inst = insts[-2]
            v1 = get_const_int_value(c1_inst)
            v2 = get_const_int_value(c2_inst)
            if v1 is not None and v2 is not None:
                if term.opcode == Opcode.IF_ICMPEQ:
                    condition_met = (v1 == v2)
                elif term.opcode == Opcode.IF_ICMPNE:
                    condition_met = (v1 != v2)
                elif term.opcode == Opcode.IF_ICMPLT:
                    condition_met = (v1 < v2)
                elif term.opcode == Opcode.IF_ICMPGE:
                    condition_met = (v1 >= v2)
                elif term.opcode == Opcode.IF_ICMPGT:
                    condition_met = (v1 > v2)
                elif term.opcode == Opcode.IF_ICMPLE:
                    condition_met = (v1 <= v2)
                else:
                    return False

                # Prune both push instructions
                del insts[-2]  # c2
                del insts[-2]  # c1 (shifted into -2)
                self._apply_branch_pruning(block, branch_edge, fallthrough_edge, condition_met)
                report.constant_predicates_pruned += 1
                return True

        return False

    def _try_eliminate_algebraic_predicate(self, block: BasicBlock, report: OpaquePredicateReport) -> bool:
        """
        Detects and prunes 7 families of algebraic invariants with strict stack neutrality:
        1. x ^ x == 0 (self-XOR)
        2. x - x == 0 (self-subtraction)
        3. x & 0 == 0 (zero-mask)
        4. (x | 1) != 0 (disjunction with 1)
        5. x * (x + 1) % 2 == 0 (consecutive parity)
        6. (x & 0x7FFFFFFF) >= 0 (non-negative mask)
        7. x * (x^2 - 1) % 6 == 0 (cube parity)
        """
        insts = block.instructions
        if len(insts) < 3:
            return False

        term = insts[-1]
        out_edges = self.cfg.get_outgoing_edges(block.id)
        branch_edge = next((e for e in out_edges if e.edge_type == CFGEdgeType.CONDITIONAL_BRANCH), None)
        fallthrough_edge = next((e for e in out_edges if e.edge_type == CFGEdgeType.FALLTHROUGH), None)

        if not branch_edge or term.opcode not in (Opcode.IFEQ, Opcode.IFNE, Opcode.IFGE, Opcode.IFLT):
            return False

        # Invariant 1: iload x; iload x; ixor; ifeq / ifne
        if len(insts) >= 4 and insts[-2].opcode == Opcode.IXOR:
            l1, l2 = insts[-4], insts[-3]
            s1, s2 = get_iload_slot(l1), get_iload_slot(l2)
            if s1 is not None and s1 == s2 and term.opcode in (Opcode.IFEQ, Opcode.IFNE):
                condition_met = (term.opcode == Opcode.IFEQ)
                del insts[-4 : -1]
                self._apply_branch_pruning(block, branch_edge, fallthrough_edge, condition_met)
                report.algebraic_predicates_pruned += 1
                return True

        # Invariant 2: iload x; iload x; isub; ifeq / ifne
        if len(insts) >= 4 and insts[-2].opcode == Opcode.ISUB:
            l1, l2 = insts[-4], insts[-3]
            s1, s2 = get_iload_slot(l1), get_iload_slot(l2)
            if s1 is not None and s1 == s2 and term.opcode in (Opcode.IFEQ, Opcode.IFNE):
                condition_met = (term.opcode == Opcode.IFEQ)
                del insts[-4 : -1]
                self._apply_branch_pruning(block, branch_edge, fallthrough_edge, condition_met)
                report.algebraic_predicates_pruned += 1
                return True

        # Invariant 3: iload x; iconst_0; iand; ifeq / ifne (commutative)
        if len(insts) >= 4 and insts[-2].opcode == Opcode.IAND:
            c0 = get_const_int_value(insts[-3])
            s0 = get_iload_slot(insts[-4])
            c1 = get_const_int_value(insts[-4])
            s1 = get_iload_slot(insts[-3])
            matched = (c0 == 0 and s0 is not None) or (c1 == 0 and s1 is not None)
            if matched and term.opcode in (Opcode.IFEQ, Opcode.IFNE):
                condition_met = (term.opcode == Opcode.IFEQ)
                del insts[-4 : -1]
                self._apply_branch_pruning(block, branch_edge, fallthrough_edge, condition_met)
                report.algebraic_predicates_pruned += 1
                return True

        # Invariant 4: iload x; iconst_1; ior; ifne / ifeq (commutative)
        if len(insts) >= 4 and insts[-2].opcode == Opcode.IOR:
            c0 = get_const_int_value(insts[-3])
            s0 = get_iload_slot(insts[-4])
            c1 = get_const_int_value(insts[-4])
            s1 = get_iload_slot(insts[-3])
            matched = (c0 == 1 and s0 is not None) or (c1 == 1 and s1 is not None)
            if matched and term.opcode in (Opcode.IFEQ, Opcode.IFNE):
                condition_met = (term.opcode == Opcode.IFNE)
                del insts[-4 : -1]
                self._apply_branch_pruning(block, branch_edge, fallthrough_edge, condition_met)
                report.algebraic_predicates_pruned += 1
                return True

        # Invariant 5: Consecutive integer parity: x * (x + 1) % 2 == 0
        # Canonical: iload x; iload x; iconst_1; iadd; imul; iconst_2; irem; ifeq / ifne
        if len(insts) >= 8 and insts[-2].opcode == Opcode.IREM and term.opcode in (Opcode.IFEQ, Opcode.IFNE):
            c_mod = get_const_int_value(insts[-3])
            if c_mod == 2 and insts[-4].opcode == Opcode.IMUL:
                matched = False
                # Form 1: iload x; (iload x; iconst_1; iadd); imul
                if insts[-5].opcode == Opcode.IADD:
                    if get_const_int_value(insts[-6]) == 1:
                        s1 = get_iload_slot(insts[-7])
                        s0 = get_iload_slot(insts[-8])
                        if s0 is not None and s0 == s1:
                            matched = True
                    elif get_const_int_value(insts[-7]) == 1:
                        s1 = get_iload_slot(insts[-6])
                        s0 = get_iload_slot(insts[-8])
                        if s0 is not None and s0 == s1:
                            matched = True
                # Form 2: (iload x; iconst_1; iadd); iload x; imul
                elif insts[-6].opcode == Opcode.IADD:
                    s_right = get_iload_slot(insts[-5])
                    if get_const_int_value(insts[-7]) == 1:
                        s_left = get_iload_slot(insts[-8])
                        if s_left is not None and s_left == s_right:
                            matched = True
                    elif get_const_int_value(insts[-8]) == 1:
                        s_left = get_iload_slot(insts[-7])
                        if s_left is not None and s_left == s_right:
                            matched = True
                if matched:
                    condition_met = (term.opcode == Opcode.IFEQ)
                    del insts[-8 : -1]
                    self._apply_branch_pruning(block, branch_edge, fallthrough_edge, condition_met)
                    report.algebraic_predicates_pruned += 1
                    return True

        # Invariant 6: (x & 0x7FFFFFFF) >= 0 (non-negative mask)
        # Pattern: iload x; ldc 0x7FFFFFFF; iand; ifge / iflt
        if len(insts) >= 4 and insts[-2].opcode == Opcode.IAND and term.opcode in (Opcode.IFGE, Opcode.IFLT):
            cp = getattr(self, "cp", None)
            matched = False
            if is_nonnegative_mask_constant(insts[-3], cp) and get_iload_slot(insts[-4]) is not None:
                matched = True
            elif is_nonnegative_mask_constant(insts[-4], cp) and get_iload_slot(insts[-3]) is not None:
                matched = True
            if matched:
                condition_met = (term.opcode == Opcode.IFGE)
                del insts[-4 : -1]
                self._apply_branch_pruning(block, branch_edge, fallthrough_edge, condition_met)
                report.algebraic_predicates_pruned += 1
                return True

        # Invariant 7: x * (x^2 - 1) % 6 == 0 (cube parity)
        # Form A: iload x; iload x; iload x; imul; iconst_1; isub; imul; bipush 6; irem; ifeq / ifne
        # Form B: iload x; iload x; imul; iconst_1; isub; iload x; imul; bipush 6; irem; ifeq / ifne
        if len(insts) >= 10 and insts[-2].opcode == Opcode.IREM and term.opcode in (Opcode.IFEQ, Opcode.IFNE):
            c_mod = get_const_int_value(insts[-3])
            if c_mod == 6 and insts[-4].opcode == Opcode.IMUL:
                matched = False
                # Form A: x * (x^2 - 1)
                if (insts[-5].opcode == Opcode.ISUB and
                    get_const_int_value(insts[-6]) == 1 and
                    insts[-7].opcode == Opcode.IMUL):
                    s2 = get_iload_slot(insts[-8])
                    s1 = get_iload_slot(insts[-9])
                    s0 = get_iload_slot(insts[-10])
                    if s0 is not None and s0 == s1 and s1 == s2:
                        matched = True
                # Form B: (x^2 - 1) * x
                elif (insts[-6].opcode == Opcode.ISUB and
                      get_const_int_value(insts[-7]) == 1 and
                      insts[-8].opcode == Opcode.IMUL):
                    s2 = get_iload_slot(insts[-5])
                    s1 = get_iload_slot(insts[-9])
                    s0 = get_iload_slot(insts[-10])
                    if s0 is not None and s0 == s1 and s1 == s2:
                        matched = True
                if matched:
                    condition_met = (term.opcode == Opcode.IFEQ)
                    del insts[-10 : -1]
                    self._apply_branch_pruning(block, branch_edge, fallthrough_edge, condition_met)
                    report.algebraic_predicates_pruned += 1
                    return True

        return False

    def _apply_branch_pruning(
        self,
        block: BasicBlock,
        branch_edge: CFGEdge,
        fallthrough_edge: Optional[CFGEdge],
        condition_met: bool
    ) -> None:
        """
        Replaces the conditional branch with GOTO (if condition always True)
        or eliminates the branch entirely into fallthrough (if condition always False).
        """
        term = block.instructions[-1]

        if condition_met:
            # Branch is ALWAYS TAKEN
            # Replace conditional branch instruction with unconditional GOTO
            block.instructions[-1] = Instruction(
                pc=term.pc,
                opcode=Opcode.GOTO,
                operands=struct.pack(">h", 0)
            )
            # Remove fallthrough edge
            if fallthrough_edge:
                self.cfg.remove_edge(block.id, fallthrough_edge.target_id, CFGEdgeType.FALLTHROUGH)
            # Change branch edge to UNCONDITIONAL_BRANCH
            branch_edge.edge_type = CFGEdgeType.UNCONDITIONAL_BRANCH
        else:
            # Branch is NEVER TAKEN
            # Remove conditional branch instruction, execution falls through
            del block.instructions[-1]
            # Remove conditional branch edge
            self.cfg.remove_edge(block.id, branch_edge.target_id, CFGEdgeType.CONDITIONAL_BRANCH)


def deflatten_control_flow(cfg: ControlFlowGraph) -> ControlFlowGraph:
    """Standalone functional interface matching PROJECT.md MethodNormalizer contract."""
    deflattener = ControlFlowDeflattener(cfg)
    deflattener.deflatten()
    return cfg


def eliminate_opaque_predicates(cfg: ControlFlowGraph, cp: Optional[Any] = None) -> int:
    """Functional interface returning count of pruned predicates."""
    eliminator = OpaquePredicateEliminator(cfg, cp=cp)
    report = eliminator.eliminate()
    return report.algebraic_predicates_pruned + report.constant_predicates_pruned
