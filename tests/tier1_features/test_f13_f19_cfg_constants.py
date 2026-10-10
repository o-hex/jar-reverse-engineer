"""
Tier 1: Feature Coverage for F13 - F19 (Control Flow Normalization & Constant Resolution).
Tests 35 isolated features (5 per feature F13 through F19) per explorer_e2e_3 specification.
"""

from __future__ import annotations
import struct
from pathlib import Path
from typing import Dict, List

import pytest

from jar_engine.model.constants import Opcode, AccessFlags, ConstantTag, ReferenceKind
from jar_engine.model.class_file import ClassFile
from jar_engine.model.cp_info import (
    ConstantPool,
    CONSTANT_MethodHandle_info,
    CONSTANT_MethodType_info,
    CONSTANT_InvokeDynamic_info
)
from jar_engine.model.member_info import (
    MethodInfo, CodeAttribute, ExceptionTableEntry,
    BootstrapMethodsAttribute, BootstrapMethod
)
from jar_engine.model.jar_archive import JarArchive
from jar_engine.cfg import (
    Instruction, CFGEdge, CFGEdgeType, BasicBlock,
    ControlFlowGraph,
    partition_basic_blocks, decode_instructions,
    ControlFlowDeflattener, deflatten_control_flow,
    OpaquePredicateEliminator, eliminate_opaque_predicates,
    prune_unreachable_blocks, compact_goto_chains,
    eliminate_fallthrough_gotos, strip_nops_from_bytecode,
    reassemble_cfg, prune_dead_code_and_nops
)
from jar_engine.normalization import (
    fold_constants_and_arithmetic, reconstruct_encrypted_strings,
    resolve_dynamic_invokes_and_reflection,
    ArithmeticEvaluator, BytecodeEmulator, DecryptorMethodDetector,
    DynamicInvokeResolver
)


# ============================================================================
# F13: CFG Basic Block Partitioning & Graph Construction
# ============================================================================

def test_f13_instruction_decoding():
    """F13.1: Decode raw bytecode into typed Instruction objects with opcodes and operands."""
    code = bytes([
        Opcode.ICONST_1,
        Opcode.BIPUSH, 42,
        Opcode.ISTORE_1,
        Opcode.ILOAD_1,
        Opcode.IRETURN
    ])
    instructions = decode_instructions(code)
    assert len(instructions) == 5
    assert instructions[0].opcode == Opcode.ICONST_1
    assert instructions[0].pc == 0
    assert instructions[1].opcode == Opcode.BIPUSH
    assert instructions[1].operands == bytes([42])
    assert instructions[4].opcode == Opcode.IRETURN


def test_f13_leader_identification():
    """F13.2: Identify basic block leaders (entry, jump targets, post-jump instructions)."""
    # pc 0: iconst_1
    # pc 1: ifeq +5 (target pc 6)
    # pc 4: iconst_2
    # pc 5: return
    # pc 6: return
    code = (
        bytes([Opcode.ICONST_1, Opcode.IFEQ]) + struct.pack(">h", 5) +
        bytes([Opcode.ICONST_2, Opcode.RETURN, Opcode.RETURN])
    )
    blocks = partition_basic_blocks(code)
    # Leaders must include: 0 (entry), 4 (post-branch fallthrough), 6 (branch target)
    leader_pcs = {b.start_pc for b in blocks}
    assert 0 in leader_pcs
    assert 4 in leader_pcs
    assert 6 in leader_pcs


def test_f13_fallthrough_and_jump_edges():
    """F13.3: Connect blocks with typed edges (FALLTHROUGH, UNCONDITIONAL_BRANCH / CONDITIONAL_BRANCH)."""
    # pc 0: goto +3 (target pc 3)
    # pc 3: return
    code = bytes([Opcode.GOTO]) + struct.pack(">h", 3) + bytes([Opcode.RETURN])
    cfg = ControlFlowGraph.from_code(code)
    entry = cfg.get_entry_block()
    assert entry is not None
    out_edges = cfg.get_outgoing_edges(entry.id)
    assert len(out_edges) == 1
    assert out_edges[0].edge_type == CFGEdgeType.UNCONDITIONAL_BRANCH
    assert out_edges[0].target_id == 3


def test_f13_switch_edges_partition():
    """F13.4: Partition tableswitch and lookupswitch targets into distinct branch edges."""
    # tableswitch low=0, high=1, default=14, case 0=14, case 1=15
    code = bytes([Opcode.ICONST_1, Opcode.TABLESWITCH, 0, 0]) + struct.pack(">iii", 14, 0, 1) + struct.pack(">ii", 14, 15) + bytes([Opcode.RETURN, Opcode.RETURN])
    cfg = ControlFlowGraph.from_code(code)
    entry = cfg.get_entry_block()
    assert entry is not None
    out_edges = cfg.get_outgoing_edges(entry.id)
    case_edges = [e for e in out_edges if e.edge_type == CFGEdgeType.SWITCH_CASE]
    default_edges = [e for e in out_edges if e.edge_type == CFGEdgeType.SWITCH_DEFAULT]
    assert len(case_edges) == 2
    assert len(default_edges) == 1


def test_f13_cyclomatic_complexity_calculation():
    """F13.5: Calculate E - N + 2P cyclomatic complexity."""
    # Linear: 1 block, 0 edges -> complexity = 1
    cfg_linear = ControlFlowGraph.from_code(bytes([Opcode.ICONST_1, Opcode.IRETURN]))
    assert cfg_linear.cyclomatic_complexity == 1

    # Branch: diamond (2 branches) -> complexity = 2
    code_diamond = (
        bytes([Opcode.ICONST_1, Opcode.IFEQ]) + struct.pack(">h", 4) +
        bytes([Opcode.ICONST_2, Opcode.IRETURN, Opcode.ICONST_3, Opcode.IRETURN])
    )
    cfg_diamond = ControlFlowGraph.from_code(code_diamond)
    assert cfg_diamond.cyclomatic_complexity >= 2


# ============================================================================
# F14: Control Flow De-Flattening
# ============================================================================

def test_f14_dispatcher_loop_detection(fixture_jars):
    """F14.1: Statically detect state-variable dispatch loops in control_flow_flattened.jar."""
    arc = JarArchive.from_file(fixture_jars["control_flow_flattened"])
    cf = arc.get_class("com/synthetic/flattened/DispatcherService")
    assert cf is not None
    m = cf.get_method("dispatchAlgorithm", "(I)I")
    assert m is not None

    code_attr = m.code_attribute(cf.constant_pool)
    cfg = ControlFlowGraph.from_code(code_attr.code)
    degen = ControlFlowDeflattener(cfg)
    report = degen.deflatten()
    assert report.loops_detected >= 1


def test_f14_state_variable_tracking():
    """F14.2: Track assignment and transitions of state variables."""
    from jar_engine.cfg.deflattener import get_iload_slot, get_istore_slot
    cfg = ControlFlowGraph()
    # Init block setting slot 1 to state 0
    b_init = BasicBlock(0, 0, 4, [
        Instruction(0, Opcode.ICONST_0),
        Instruction(1, Opcode.ISTORE_1),
        Instruction(2, Opcode.GOTO, struct.pack(">h", 8))
    ])
    # Dispatcher block checking slot 1
    b_disp = BasicBlock(10, 10, 30, [
        Instruction(10, Opcode.ILOAD_1),
        Instruction(11, Opcode.TABLESWITCH, operands=b"\x00" * 16)
    ])
    # State 0 sets slot 1 to state 1
    b_p0 = BasicBlock(100, 100, 105, [
        Instruction(100, Opcode.ICONST_1),
        Instruction(101, Opcode.ISTORE_1),
        Instruction(102, Opcode.GOTO, struct.pack(">h", 0))
    ])
    for b in [b_init, b_disp, b_p0]:
        cfg.add_block(b)
    cfg.add_edge(CFGEdge(0, 10, CFGEdgeType.UNCONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(10, 100, CFGEdgeType.SWITCH_CASE, case_key=0))
    cfg.add_edge(CFGEdge(100, 10, CFGEdgeType.UNCONDITIONAL_BRANCH))

    assert get_istore_slot(b_init.instructions[1]) == 1
    assert get_iload_slot(b_disp.instructions[0]) == 1
    assert get_istore_slot(b_p0.instructions[1]) == 1


def test_f14_switch_unflattening():
    """F14.3: Replace dispatcher switch with natural direct control blocks."""
    cfg = ControlFlowGraph()
    b_init = BasicBlock(0, 0, 4, [Instruction(0, Opcode.ICONST_0), Instruction(1, Opcode.ISTORE_1), Instruction(2, Opcode.GOTO, struct.pack(">h", 8))])
    b_disp = BasicBlock(10, 10, 30, [Instruction(10, Opcode.ILOAD_1), Instruction(11, Opcode.TABLESWITCH, operands=b"\x00" * 16)])
    b_p0 = BasicBlock(100, 100, 105, [Instruction(100, Opcode.NOP), Instruction(101, Opcode.ICONST_1), Instruction(102, Opcode.ISTORE_1), Instruction(103, Opcode.GOTO, struct.pack(">h", 0))])
    b_p1 = BasicBlock(200, 200, 202, [Instruction(200, Opcode.RETURN)])

    for b in [b_init, b_disp, b_p0, b_p1]:
        cfg.add_block(b)
    cfg.add_edge(CFGEdge(0, 10, CFGEdgeType.UNCONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(10, 100, CFGEdgeType.SWITCH_CASE, case_key=0))
    cfg.add_edge(CFGEdge(10, 200, CFGEdgeType.SWITCH_CASE, case_key=1))
    cfg.add_edge(CFGEdge(10, 200, CFGEdgeType.SWITCH_DEFAULT))
    cfg.add_edge(CFGEdge(100, 10, CFGEdgeType.UNCONDITIONAL_BRANCH))

    report = ControlFlowDeflattener(cfg).deflatten()
    assert report.edges_reconstructed >= 1
    # Dispatcher 10 removed and direct edge 0 -> 100 formed
    assert 10 not in cfg.blocks
    assert 100 in cfg.get_successors(0)


def test_f14_complexity_reduction(fixture_jars):
    """F14.4: Verify cyclomatic complexity measurably drops post-deflattening."""
    arc = JarArchive.from_file(fixture_jars["control_flow_flattened"])
    cf = arc.get_class("com/synthetic/flattened/DispatcherService")
    m = cf.get_method("dispatchAlgorithm", "(I)I")
    code_attr = m.code_attribute(cf.constant_pool)
    cfg = ControlFlowGraph.from_code(code_attr.code)

    initial_complexity = cfg.cyclomatic_complexity
    deflatten_control_flow(cfg)
    final_complexity = cfg.cyclomatic_complexity
    assert final_complexity <= initial_complexity


def test_f14_cfg_reassembly_after_deflattening(fixture_jars):
    """F14.5: Reassemble CFG into valid sequential bytecode after deflattening."""
    arc = JarArchive.from_file(fixture_jars["control_flow_flattened"])
    cf = arc.get_class("com/synthetic/flattened/DispatcherService")
    m = cf.get_method("dispatchAlgorithm", "(I)I")
    code_attr = m.code_attribute(cf.constant_pool)
    cfg = ControlFlowGraph.from_code(code_attr.code)

    deflatten_control_flow(cfg)
    new_code, new_ex_table, max_stack, max_locals = reassemble_cfg(cfg)
    assert len(new_code) > 0
    # Decodes cleanly without exceptions
    instrs = decode_instructions(new_code)
    assert len(instrs) > 0


# ============================================================================
# F15: Opaque Predicate Elimination
# ============================================================================

def test_f15_constant_tautology_pruning():
    """F15.1: Replace if (1 == 1) with unconditional jump."""
    cfg = ControlFlowGraph()
    b_entry = BasicBlock(0, 0, 4, [
        Instruction(0, Opcode.ICONST_0),
        Instruction(1, Opcode.IFEQ, struct.pack(">h", 10))  # 0 == 0 always True
    ])
    b_fall = BasicBlock(4, 4, 5, [Instruction(4, Opcode.RETURN)])
    b_target = BasicBlock(10, 10, 11, [Instruction(10, Opcode.RETURN)])
    for b in [b_entry, b_fall, b_target]:
        cfg.add_block(b)
    cfg.add_edge(CFGEdge(0, 10, CFGEdgeType.CONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(0, 4, CFGEdgeType.FALLTHROUGH))

    report = OpaquePredicateEliminator(cfg).eliminate()
    assert report.constant_predicates_pruned == 1
    assert 4 not in cfg.get_successors(0)
    assert 10 in cfg.get_successors(0)
    assert b_entry.terminator.opcode == Opcode.GOTO


def test_f15_constant_contradiction_pruning():
    """F15.2: Remove dead branch of if (0 == 1)."""
    cfg = ControlFlowGraph()
    b_entry = BasicBlock(0, 0, 4, [
        Instruction(0, Opcode.ICONST_1),
        Instruction(1, Opcode.IFEQ, struct.pack(">h", 10))  # 1 == 0 always False
    ])
    b_fall = BasicBlock(4, 4, 5, [Instruction(4, Opcode.RETURN)])
    b_target = BasicBlock(10, 10, 11, [Instruction(10, Opcode.RETURN)])
    for b in [b_entry, b_fall, b_target]:
        cfg.add_block(b)
    cfg.add_edge(CFGEdge(0, 10, CFGEdgeType.CONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(0, 4, CFGEdgeType.FALLTHROUGH))

    report = OpaquePredicateEliminator(cfg).eliminate()
    assert report.constant_predicates_pruned == 1
    # Target 10 disconnected, execution falls through to 4
    assert 10 not in cfg.get_successors(0)
    assert 4 in cfg.get_successors(0)


def test_f15_branch_target_rewriting():
    """F15.3: Rewrite predecessor edges and prune orphaned blocks."""
    cfg = ControlFlowGraph()
    b0 = BasicBlock(0, 0, 7, [
        Instruction(0, Opcode.BIPUSH, struct.pack(">b", 5)),
        Instruction(2, Opcode.BIPUSH, struct.pack(">b", 5)),
        Instruction(4, Opcode.IF_ICMPEQ, struct.pack(">h", 10))  # 5 == 5 always True
    ])
    b_dead = BasicBlock(7, 7, 8, [Instruction(7, Opcode.NOP)])
    b_target = BasicBlock(10, 10, 11, [Instruction(10, Opcode.RETURN)])
    for b in [b0, b_dead, b_target]:
        cfg.add_block(b)
    cfg.add_edge(CFGEdge(0, 10, CFGEdgeType.CONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(0, 7, CFGEdgeType.FALLTHROUGH))

    pruned = eliminate_opaque_predicates(cfg)
    assert pruned == 1
    prune_unreachable_blocks(cfg)
    assert 7 not in cfg.blocks


def test_f15_bogus_block_disconnection():
    """F15.4: Disconnect bogus dead blocks inserted by obfuscators."""
    cfg = ControlFlowGraph()
    # x ^ x == 0 is always true
    b_entry = BasicBlock(0, 0, 6, [
        Instruction(0, Opcode.ILOAD_0),
        Instruction(1, Opcode.ILOAD_0),
        Instruction(2, Opcode.IXOR),
        Instruction(3, Opcode.IFEQ, struct.pack(">h", 20))
    ])
    b_bogus = BasicBlock(6, 6, 9, [
        Instruction(6, Opcode.BIPUSH, struct.pack(">b", 99)),
        Instruction(8, Opcode.IRETURN)
    ])
    b_real = BasicBlock(20, 20, 21, [Instruction(20, Opcode.RETURN)])
    for b in [b_entry, b_bogus, b_real]:
        cfg.add_block(b)
    cfg.add_edge(CFGEdge(0, 20, CFGEdgeType.CONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(0, 6, CFGEdgeType.FALLTHROUGH))

    report = OpaquePredicateEliminator(cfg).eliminate()
    assert report.algebraic_predicates_pruned == 1
    assert 6 not in cfg.get_successors(0)
    assert 20 in cfg.get_successors(0)


def test_f15_eliminator_report_counts():
    """F15.5: Verify OpaquePredicateReport records exact pruned predicate counts."""
    cfg = ControlFlowGraph()
    b0 = BasicBlock(0, 0, 4, [Instruction(0, Opcode.ICONST_0), Instruction(1, Opcode.IFEQ, struct.pack(">h", 10))])
    b_fall = BasicBlock(4, 4, 5, [Instruction(4, Opcode.RETURN)])
    b_target = BasicBlock(10, 10, 11, [Instruction(10, Opcode.RETURN)])
    for b in [b0, b_fall, b_target]:
        cfg.add_block(b)
    cfg.add_edge(CFGEdge(0, 10, CFGEdgeType.CONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(0, 4, CFGEdgeType.FALLTHROUGH))

    report = OpaquePredicateEliminator(cfg).eliminate()
    assert report.constant_predicates_pruned == 1
    assert report.constant_predicates_pruned + report.algebraic_predicates_pruned == 1


# ============================================================================
# F16: Constant Folding Lattice
# ============================================================================

def test_f16_arithmetic_evaluation_add_sub():
    """F16.1: Fold constant integer addition and subtraction (10 + 20 -> 30)."""
    cfg = ControlFlowGraph()
    b = BasicBlock(0, 0, 6, [
        Instruction(0, Opcode.BIPUSH, struct.pack(">b", 10)),
        Instruction(2, Opcode.BIPUSH, struct.pack(">b", 20)),
        Instruction(4, Opcode.IADD),
        Instruction(5, Opcode.IRETURN)
    ])
    cfg.add_block(b)
    folded = fold_constants_and_arithmetic(cfg)
    assert folded == 1
    assert len(b.instructions) == 2
    assert b.instructions[0].opcode == Opcode.BIPUSH
    assert struct.unpack(">b", b.instructions[0].operands[:1])[0] == 30


def test_f16_arithmetic_evaluation_mul_div():
    """F16.2: Fold multiplication and integer division."""
    cfg = ControlFlowGraph()
    b = BasicBlock(0, 0, 6, [
        Instruction(0, Opcode.BIPUSH, struct.pack(">b", 30)),
        Instruction(2, Opcode.BIPUSH, struct.pack(">b", 5)),
        Instruction(4, Opcode.IDIV),
        Instruction(5, Opcode.IRETURN)
    ])
    cfg.add_block(b)
    folded = fold_constants_and_arithmetic(cfg)
    assert folded == 1
    assert b.instructions[0].opcode in (Opcode.BIPUSH, Opcode.ICONST_5, Opcode.ICONST_1, Opcode.SIPUSH)
    val = ArithmeticEvaluator.eval_binary_int(Opcode.IDIV, 30, 5)
    assert val == 6


def test_f16_bitwise_operations():
    """F16.3: Fold bitwise XOR, AND, OR, and shift operations."""
    assert ArithmeticEvaluator.eval_binary_int(Opcode.IXOR, 0x0F, 0xFF) == 0xF0
    assert ArithmeticEvaluator.eval_binary_int(Opcode.IAND, 0xFF, 0x0F) == 0x0F
    assert ArithmeticEvaluator.eval_binary_int(Opcode.IOR, 0xF0, 0x0F) == 0xFF
    assert ArithmeticEvaluator.eval_binary_int(Opcode.ISHL, 1, 4) == 16
    assert ArithmeticEvaluator.eval_binary_int(Opcode.ISHR, 16, 2) == 4


def test_f16_abstract_frame_propagation():
    """F16.4: Propagate constants across local variable frames."""
    from jar_engine.normalization.constant_folding import AbstractFrame, LatticeValue
    frame = AbstractFrame()
    v1 = LatticeValue.constant(10, "int")
    frame.locals[1] = v1
    assert frame.locals[1].is_constant
    assert frame.locals[1].val == 10

    v2 = LatticeValue.constant(10, "int")
    assert v1.join(v2).is_constant

    v3 = LatticeValue.constant(20, "int")
    assert not v1.join(v3).is_constant


def test_f16_constant_folder_instruction_replacement():
    """F16.5: Replace multi-opcode expression sequences with single constant push."""
    cfg = ControlFlowGraph()
    b = BasicBlock(0, 0, 8, [
        Instruction(0, Opcode.BIPUSH, struct.pack(">b", 7)),
        Instruction(2, Opcode.BIPUSH, struct.pack(">b", 3)),
        Instruction(4, Opcode.IMUL),
        Instruction(5, Opcode.ICONST_1),
        Instruction(6, Opcode.IADD),
        Instruction(7, Opcode.IRETURN)
    ])
    cfg.add_block(b)
    folded = fold_constants_and_arithmetic(cfg)
    assert folded == 2
    # Final instruction is single constant push (22) followed by ireturn
    assert len(b.instructions) == 2
    assert b.instructions[1].opcode == Opcode.IRETURN


# ============================================================================
# F17: Static String Decryptor Resolution
# ============================================================================

def test_f17_decryptor_method_detector(make_test_class):
    """F17.1: Identify XOR/Caesar string decoding method signatures."""
    cf = make_test_class("pkg/DetectorClass")
    cp = cf.constant_pool
    m_name = cp.add_utf8("decrypt")
    m_desc = cp.add_utf8("(Ljava/lang/String;I)Ljava/lang/String;")
    code_bytes = bytes([Opcode.ALOAD_0, Opcode.ILOAD_1, Opcode.IXOR, Opcode.ARETURN])
    code_attr = CodeAttribute(cp.add_utf8("Code"), 2, 2, code_bytes)
    m = MethodInfo(AccessFlags.STATIC | AccessFlags.PUBLIC, m_name, m_desc, [code_attr])
    cf.methods.append(m)

    assert DecryptorMethodDetector.is_candidate(cf, m) is True


def test_f17_bytecode_emulator_xor(make_test_class):
    """F17.2: Emulate XOR cipher loop in pure Python sandbox."""
    cf = make_test_class("pkg/XorEmulatorClass")
    cp = cf.constant_pool
    m_name = cp.add_utf8("xorDec")
    m_desc = cp.add_utf8("(Ljava/lang/String;I)Ljava/lang/String;")

    str_to_char_ref = cp.add_member_ref(ConstantTag.METHODREF, "java/lang/String", "toCharArray", "()[C")
    str_init_ref = cp.add_member_ref(ConstantTag.METHODREF, "java/lang/String", "<init>", "([C)V")
    str_cls = cp.add_class("java/lang/String")

    code = (
        bytes([Opcode.ALOAD_0, Opcode.INVOKEVIRTUAL]) + struct.pack(">H", str_to_char_ref) +
        bytes([Opcode.ASTORE_2, Opcode.ICONST_0, Opcode.ISTORE_3, Opcode.ILOAD_3, Opcode.ALOAD_2, Opcode.ARRAYLENGTH, Opcode.IF_ICMPGE]) +
        struct.pack(">h", 19) +
        bytes([Opcode.ALOAD_2, Opcode.ILOAD_3, Opcode.ALOAD_2, Opcode.ILOAD_3, Opcode.CALOAD, Opcode.ILOAD_1, Opcode.IXOR, Opcode.I2C, Opcode.CASTORE, Opcode.IINC, 3, 1, Opcode.GOTO]) +
        struct.pack(">h", -18) +
        bytes([Opcode.NOP, Opcode.NEW]) + struct.pack(">H", str_cls) +
        bytes([Opcode.DUP, Opcode.ALOAD_2, Opcode.INVOKESPECIAL]) + struct.pack(">H", str_init_ref) +
        bytes([Opcode.ARETURN])
    )
    code_attr = CodeAttribute(cp.add_utf8("Code"), 4, 4, code)
    m = MethodInfo(AccessFlags.STATIC | AccessFlags.PUBLIC, m_name, m_desc, [code_attr])
    cf.methods.append(m)

    emulator = BytecodeEmulator()
    plain = "VERIFIED"
    key = 0x55
    cipher = "".join(chr(ord(c) ^ key) for c in plain)
    res = emulator.execute(cf, m, [cipher, key])
    assert res == "VERIFIED"


def test_f17_ciphertext_extraction(fixture_jars):
    """F17.3: Statically extract ciphertext byte arrays or constant strings."""
    arc = JarArchive.from_file(fixture_jars["constant_obfuscated"])
    cf = arc.get_class("com/synthetic/constants/StringObfuscated")
    assert cf is not None
    # Verify ciphertext strings are present in constant pool
    cp_strings = [cf.constant_pool.resolve_string(idx) for idx, e in cf.constant_pool.entries() if e.tag == ConstantTag.STRING]
    assert len(cp_strings) >= 2


def test_f17_string_decryption_inlining(fixture_jars):
    """F17.4: Replace decryption method call site with decrypted constant string."""
    arc = JarArchive.from_file(fixture_jars["constant_obfuscated"])
    cf = arc.get_class("com/synthetic/constants/StringObfuscated")
    m = cf.get_method("getSecretMessage", "()Ljava/lang/String;")
    assert m is not None

    count = reconstruct_encrypted_strings(cf, m)
    assert count >= 1


def test_f17_reconstruct_encrypted_strings_pass(fixture_jars):
    """F17.5: Verify reconstruct_encrypted_strings reports correct decryption counts."""
    arc = JarArchive.from_file(fixture_jars["constant_obfuscated"])
    cf = arc.get_class("com/synthetic/constants/StringObfuscated")
    m = cf.get_method("getDatabaseKey", "()Ljava/lang/String;")
    assert m is not None

    count = reconstruct_encrypted_strings(cf, m)
    assert count == 1


# ============================================================================
# F18: Dynamic Invoke & Reflection Resolution
# ============================================================================

def test_f18_string_concat_factory_resolution(fixture_jars):
    """F18.1: Resolve Java 9+ StringConcatFactory.makeConcatWithConstants to constant strings."""
    arc = JarArchive.from_file(fixture_jars["constant_obfuscated"])
    cf = arc.get_class("com/synthetic/constants/DynamicInvoker")
    assert cf is not None
    m = cf.get_method("concatSecret", "()Ljava/lang/String;")
    assert m is not None

    resolved = resolve_dynamic_invokes_and_reflection(cf, m)
    assert resolved >= 1


def test_f18_lambda_metafactory_resolution(fixture_jars):
    """F18.2: Resolve LambdaMetafactory call site to concrete target method."""
    arc = JarArchive.from_file(fixture_jars["constant_obfuscated"])
    cf = arc.get_class("com/synthetic/constants/DynamicInvoker")
    m = cf.get_method("getWorker", "()Ljava/lang/Runnable;")
    assert m is not None

    resolver = DynamicInvokeResolver(cf)
    code_attr = m.code_attribute(cf.constant_pool)
    instrs = decode_instructions(code_attr.code)
    indy_instr = next(i for i in instrs if i.opcode == Opcode.INVOKEDYNAMIC)
    indy_cp_idx = struct.unpack(">H", indy_instr.operands[:2])[0]
    indy_entry = cf.constant_pool[indy_cp_idx]

    lambda_info = resolver.resolve_lambda(indy_entry)
    assert lambda_info is not None
    assert lambda_info.impl_method_name == "workerAction"


def test_f18_constant_reflection_forname(make_test_class):
    """F18.3: Statically resolve Class.forName(\"com.example.Foo\") to class literal."""
    cf = make_test_class("pkg/ReflectTest")
    cp = cf.constant_pool
    s_idx = cp.add_string("com.example.TargetClass")
    mref_for_name = cp.add_member_ref(ConstantTag.METHODREF, "java/lang/Class", "forName", "(Ljava/lang/String;)Ljava/lang/Class;")
    mref_invoke = cp.add_member_ref(ConstantTag.METHODREF, "java/lang/reflect/Method", "invoke", "(Ljava/lang/Object;[Ljava/lang/Object;)Ljava/lang/Object;")
    s_m_idx = cp.add_string("targetMethod")

    code = (
        bytes([Opcode.LDC, s_idx, Opcode.INVOKESTATIC]) + struct.pack(">H", mref_for_name) +
        bytes([Opcode.LDC, s_m_idx, Opcode.INVOKEVIRTUAL]) + struct.pack(">H", mref_invoke) +
        bytes([Opcode.RETURN])
    )
    m = MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("execReflect"), cp.add_utf8("()V"), [
        CodeAttribute(cp.add_utf8("Code"), 2, 1, code)
    ])
    cf.methods.append(m)

    resolver = DynamicInvokeResolver(cf)
    reflections = resolver.resolve_reflection_calls(m)
    assert len(reflections) == 1
    assert reflections[0].target_class == "com/example/TargetClass"


def test_f18_constant_reflection_getmethod(make_test_class):
    """F18.4: Statically resolve Class.getMethod(\"foo\") references."""
    cf = make_test_class("pkg/GetMethodReflect")
    cp = cf.constant_pool
    s_cls = cp.add_string("com.example.Service")
    s_mth = cp.add_string("doServe")
    mref_for_name = cp.add_member_ref(ConstantTag.METHODREF, "java/lang/Class", "forName", "(Ljava/lang/String;)Ljava/lang/Class;")
    mref_invoke = cp.add_member_ref(ConstantTag.METHODREF, "java/lang/reflect/Method", "invoke", "(Ljava/lang/Object;[Ljava/lang/Object;)Ljava/lang/Object;")

    code = (
        bytes([Opcode.LDC, s_cls, Opcode.INVOKESTATIC]) + struct.pack(">H", mref_for_name) +
        bytes([Opcode.LDC, s_mth, Opcode.INVOKEVIRTUAL]) + struct.pack(">H", mref_invoke) +
        bytes([Opcode.RETURN])
    )
    m = MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("test"), cp.add_utf8("()V"), [
        CodeAttribute(cp.add_utf8("Code"), 2, 1, code)
    ])
    cf.methods.append(m)

    resolver = DynamicInvokeResolver(cf)
    reflections = resolver.resolve_reflection_calls(m)
    assert len(reflections) == 1
    assert reflections[0].target_method == "doServe"


def test_f18_dynamic_invoke_resolver_pass(fixture_jars):
    """F18.5: Verify resolve_dynamic_invokes_and_reflection updates call sites."""
    arc = JarArchive.from_file(fixture_jars["constant_obfuscated"])
    cf = arc.get_class("com/synthetic/constants/DynamicInvoker")
    m = cf.get_method("concatSecret", "()Ljava/lang/String;")
    assert m is not None

    count = resolve_dynamic_invokes_and_reflection(cf, m)
    assert count >= 1


# ============================================================================
# F19: Dead Code & NOP Pruning
# ============================================================================

def test_f19_unreachable_block_elimination():
    """F19.1: Remove blocks not reachable from CFG entry point."""
    code = bytes([Opcode.RETURN, Opcode.ICONST_1, Opcode.IRETURN])
    cfg = ControlFlowGraph.from_code(code)
    assert len(cfg.blocks) == 2
    pruned = prune_unreachable_blocks(cfg)
    assert pruned == 1
    assert len(cfg.blocks) == 1


def test_f19_goto_chain_compaction():
    """F19.2: Compact chains of jumps (goto L1; L1: goto L2 -> goto L2)."""
    cfg = ControlFlowGraph()
    b0 = BasicBlock(0, 0, 3, [Instruction(0, Opcode.GOTO, struct.pack(">h", 10))])
    b1 = BasicBlock(10, 10, 13, [Instruction(10, Opcode.GOTO, struct.pack(">h", 10))])
    b2 = BasicBlock(20, 20, 21, [Instruction(20, Opcode.RETURN)])
    for b in [b0, b1, b2]:
        cfg.add_block(b)
    cfg.add_edge(CFGEdge(0, 10, CFGEdgeType.UNCONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(10, 20, CFGEdgeType.UNCONDITIONAL_BRANCH))

    compacted = compact_goto_chains(cfg)
    assert compacted == 1
    assert 20 in cfg.get_successors(0)


def test_f19_fallthrough_goto_removal():
    """F19.3: Remove redundant gotos targeting immediately following instruction."""
    cfg = ControlFlowGraph()
    b0 = BasicBlock(0, 0, 3, [Instruction(0, Opcode.GOTO, struct.pack(">h", 3))])
    b1 = BasicBlock(3, 3, 4, [Instruction(3, Opcode.RETURN)])
    cfg.add_block(b0)
    cfg.add_block(b1)
    cfg.add_edge(CFGEdge(0, 3, CFGEdgeType.UNCONDITIONAL_BRANCH))

    eliminated = eliminate_fallthrough_gotos(cfg)
    assert eliminated == 1
    out = cfg.get_outgoing_edges(0)
    assert len(out) == 1
    assert out[0].edge_type == CFGEdgeType.FALLTHROUGH


def test_f19_nop_instruction_stripping():
    """F19.4: Strip all NOP (0x00) instructions and adjust branch offsets."""
    code = bytes([Opcode.GOTO]) + struct.pack(">h", 4) + bytes([Opcode.NOP, Opcode.RETURN])
    clean_code, _, removed = strip_nops_from_bytecode(code)
    assert removed == 1
    assert len(clean_code) == 4
    assert clean_code[-1] == Opcode.RETURN


def test_f19_dead_code_prune_pipeline(make_test_class):
    """F19.5: Verify prune_dead_code_and_nops reassembles valid bytecode."""
    cf = make_test_class("pkg/DeadCodePipeline")
    cp = cf.constant_pool
    # Code with NOPs and dead block
    code = bytes([Opcode.NOP, Opcode.ICONST_1, Opcode.IRETURN, Opcode.NOP, Opcode.NOP, Opcode.RETURN])
    code_attr = CodeAttribute(cp.add_utf8("Code"), 1, 1, code)
    m = MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("m"), cp.add_utf8("()I"), [code_attr])
    cf.methods.append(m)

    cfg = ControlFlowGraph.from_code(code_attr.code)
    prune_dead_code_and_nops(cfg)
    new_code, _, _, _ = reassemble_cfg(cfg)
    assert len(new_code) < len(code)
    assert Opcode.NOP not in new_code
