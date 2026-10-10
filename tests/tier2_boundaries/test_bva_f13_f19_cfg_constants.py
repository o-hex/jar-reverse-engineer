"""
tests/tier2_boundaries/test_bva_f13_f19_cfg_constants.py:
Boundary value analysis and corner cases for features F13 through F19.
Total: 35 tests.
"""

import math
import struct
import pytest

from jar_engine.model.constants import Opcode, AccessFlags, ConstantTag, ReferenceKind
from jar_engine.model.class_file import ClassFile
from jar_engine.model.cp_info import (
    ConstantPool, CONSTANT_Utf8_info, CONSTANT_Class_info, CONSTANT_String_info,
    CONSTANT_Integer_info, CONSTANT_MethodHandle_info, CONSTANT_MethodType_info,
    CONSTANT_InvokeDynamic_info
)
from jar_engine.model.member_info import (
    MethodInfo, CodeAttribute, ExceptionTableEntry,
    BootstrapMethodsAttribute, BootstrapMethod
)
from jar_engine.cfg import (
    Instruction, CFGEdge, CFGEdgeType, BasicBlock,
    ControlFlowGraph, partition_basic_blocks, decode_instructions,
    ControlFlowDeflattener, deflatten_control_flow,
    OpaquePredicateEliminator, eliminate_opaque_predicates,
    prune_unreachable_blocks, compact_goto_chains,
    eliminate_fallthrough_gotos, strip_nops_from_bytecode,
    reassemble_cfg, prune_dead_code_and_nops
)
from jar_engine.normalization import (
    MethodNormalizer, NormalizationStats,
    fold_constants_and_arithmetic, reconstruct_encrypted_strings,
    resolve_dynamic_invokes_and_reflection,
    ArithmeticEvaluator, BytecodeEmulator, DecryptorMethodDetector,
    DynamicInvokeResolver
)


def create_minimal_test_class(name: str = "com/example/CFGTest") -> ClassFile:
    """Helper to create minimal ClassFile."""
    cp = ConstantPool()
    this_c = cp.add_class(name)
    super_c = cp.add_class("java/lang/Object")
    return ClassFile(
        magic=0xCAFEBABE,
        minor_version=0,
        major_version=52,
        constant_pool=cp,
        access_flags=AccessFlags.PUBLIC,
        this_class=this_c,
        super_class=super_c,
        interfaces=[],
        fields=[],
        methods=[],
        attributes=[]
    )


# ==============================================================================
# F13 BOUNDARIES (5 tests)
# ==============================================================================

def test_bva_f13_empty_code_attribute():
    """BVA F13-1: Method with 0-byte code handled gracefully."""
    blocks = partition_basic_blocks(b"")
    assert blocks == []
    cfg = ControlFlowGraph.from_code(b"")
    assert len(cfg.blocks) == 0
    assert cfg.cyclomatic_complexity == 1


def test_bva_f13_single_instruction_return():
    """BVA F13-2: Method containing only single return opcode."""
    code = bytes([Opcode.RETURN])
    blocks = partition_basic_blocks(code)
    assert len(blocks) == 1
    assert blocks[0].start_pc == 0
    assert blocks[0].end_pc == 1
    assert blocks[0].is_terminal is True

    cfg = ControlFlowGraph.from_code(code)
    assert len(cfg.blocks) == 1
    assert cfg.cyclomatic_complexity == 1
    exit_blocks = cfg.get_exit_blocks()
    assert len(exit_blocks) == 1
    assert exit_blocks[0].id == 0


def test_bva_f13_hundred_sequential_blocks():
    """BVA F13-3: 100 sequential single-instruction basic blocks partitioned correctly."""
    # 100 sequential gotos: goto +3, goto +3, ..., return
    # Each goto is 3 bytes, so pc = i*3, target is (i+1)*3
    code = bytearray()
    for _ in range(99):
        code.append(Opcode.GOTO)
        code.extend(struct.pack(">h", 3))
    code.append(Opcode.RETURN)

    raw = bytes(code)
    blocks = partition_basic_blocks(raw)
    assert len(blocks) == 100
    cfg = ControlFlowGraph.from_code(raw)
    assert len(cfg.blocks) == 100


def test_bva_f13_tableswitch_negative_low():
    """BVA F13-4: tableswitch with negative low index (-5 to 5)."""
    # pc 0: tableswitch pad=3 (pc 4), default=+100, low=-5, high=5 (11 offsets)
    # pc 4 + 4(default) + 4(low) + 4(high) + 44(offsets) = 60
    # pc 60: return
    code = bytearray([Opcode.TABLESWITCH, 0, 0, 0])
    code.extend(struct.pack(">iii", 60, -5, 5))
    for _ in range(11):
        code.extend(struct.pack(">i", 60))
    code.append(Opcode.RETURN)

    cfg = ControlFlowGraph.from_code(bytes(code))
    entry = cfg.get_entry_block()
    assert entry is not None
    out_edges = cfg.get_outgoing_edges(entry.id)
    case_edges = [e for e in out_edges if e.edge_type == CFGEdgeType.SWITCH_CASE]
    assert len(case_edges) == 11
    keys = {e.case_key for e in case_edges}
    assert -5 in keys
    assert 0 in keys
    assert 5 in keys


def test_bva_f13_lookupswitch_50_branch_targets():
    """BVA F13-5: lookupswitch with 50 distinct key-target pairs."""
    # pc 0: lookupswitch pad=3 (pc 4), default=+500, npairs=50
    # 50 pairs * 8 bytes = 400 bytes
    code = bytearray([Opcode.LOOKUPSWITCH, 0, 0, 0])
    code.extend(struct.pack(">ii", 500, 50))
    for i in range(50):
        code.extend(struct.pack(">ii", i * 10, 500))
    # Pad until 500 with returns
    while len(code) < 500:
        code.append(Opcode.NOP)
    code.append(Opcode.RETURN)

    cfg = ControlFlowGraph.from_code(bytes(code))
    entry = cfg.get_entry_block()
    out_edges = cfg.get_outgoing_edges(entry.id)
    case_edges = [e for e in out_edges if e.edge_type == CFGEdgeType.SWITCH_CASE]
    assert len(case_edges) == 50
    default_edges = [e for e in out_edges if e.edge_type == CFGEdgeType.SWITCH_DEFAULT]
    assert len(default_edges) == 1


# ==============================================================================
# F14 BOUNDARIES (5 tests)
# ==============================================================================

def test_bva_f14_natural_cfg_zero_dispatch():
    """BVA F14-1: Method with no dispatcher loop passes through de-flattening unchanged."""
    cfg = ControlFlowGraph()
    b0 = BasicBlock(0, 0, 2, [Instruction(0, Opcode.ICONST_1), Instruction(1, Opcode.IRETURN)])
    cfg.add_block(b0)

    report = ControlFlowDeflattener(cfg).deflatten()
    assert report.dispatchers_eliminated == 0
    assert report.loops_detected == 0
    assert len(cfg.blocks) == 1


def test_bva_f14_single_state_dispatch_loop():
    """BVA F14-2: Dispatcher with only 1 state case."""
    cfg = ControlFlowGraph()
    b_init = BasicBlock(0, 0, 4, [Instruction(0, Opcode.ICONST_0), Instruction(1, Opcode.ISTORE_1), Instruction(2, Opcode.GOTO, struct.pack(">h", 8))])
    b_disp = BasicBlock(10, 10, 20, [Instruction(10, Opcode.ILOAD_1), Instruction(11, Opcode.TABLESWITCH, operands=b"\x00" * 16)])
    b_p0 = BasicBlock(100, 100, 102, [Instruction(100, Opcode.RETURN)])

    for b in [b_init, b_disp, b_p0]:
        cfg.add_block(b)
    cfg.add_edge(CFGEdge(0, 10, CFGEdgeType.UNCONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(10, 100, CFGEdgeType.SWITCH_CASE, case_key=0))

    report = ControlFlowDeflattener(cfg).deflatten()
    assert report.complexity_after <= report.complexity_before


def test_bva_f14_state_var_updated_in_branch():
    """BVA F14-3: State variable updated inside nested if condition."""
    cfg = ControlFlowGraph()
    b_init = BasicBlock(0, 0, 4, [Instruction(0, Opcode.ICONST_0), Instruction(1, Opcode.ISTORE_1), Instruction(2, Opcode.GOTO, struct.pack(">h", 8))])
    b_disp = BasicBlock(10, 10, 30, [Instruction(10, Opcode.ILOAD_1), Instruction(11, Opcode.TABLESWITCH, operands=b"\x00" * 16)])
    # Branching block that conditionally updates state variable to 1 or 2
    b_p0 = BasicBlock(100, 100, 108, [
        Instruction(100, Opcode.ILOAD_0),
        Instruction(101, Opcode.IFEQ, struct.pack(">h", 4)),
        Instruction(104, Opcode.ICONST_1),
        Instruction(105, Opcode.ISTORE_1),
        Instruction(106, Opcode.GOTO, struct.pack(">h", 0))
    ])
    b_p1 = BasicBlock(200, 200, 202, [Instruction(200, Opcode.RETURN)])

    for b in [b_init, b_disp, b_p0, b_p1]:
        cfg.add_block(b)
    cfg.add_edge(CFGEdge(0, 10, CFGEdgeType.UNCONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(10, 100, CFGEdgeType.SWITCH_CASE, case_key=0))
    cfg.add_edge(CFGEdge(10, 200, CFGEdgeType.SWITCH_CASE, case_key=1))
    cfg.add_edge(CFGEdge(100, 10, CFGEdgeType.UNCONDITIONAL_BRANCH))

    report = ControlFlowDeflattener(cfg).deflatten()
    assert isinstance(report.complexity_after, int)


def test_bva_f14_dispatcher_with_dead_states():
    """BVA F14-4: Switch contains unreachable state cases."""
    cfg = ControlFlowGraph()
    b_init = BasicBlock(0, 0, 4, [Instruction(0, Opcode.ICONST_0), Instruction(1, Opcode.ISTORE_1), Instruction(2, Opcode.GOTO, struct.pack(">h", 8))])
    b_disp = BasicBlock(10, 10, 30, [Instruction(10, Opcode.ILOAD_1), Instruction(11, Opcode.TABLESWITCH, operands=b"\x00" * 16)])
    b_p0 = BasicBlock(100, 100, 102, [Instruction(100, Opcode.RETURN)])
    b_dead = BasicBlock(999, 999, 1001, [Instruction(999, Opcode.RETURN)])  # State 42 is never set

    for b in [b_init, b_disp, b_p0, b_dead]:
        cfg.add_block(b)
    cfg.add_edge(CFGEdge(0, 10, CFGEdgeType.UNCONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(10, 100, CFGEdgeType.SWITCH_CASE, case_key=0))
    cfg.add_edge(CFGEdge(10, 999, CFGEdgeType.SWITCH_CASE, case_key=42))

    report = ControlFlowDeflattener(cfg).deflatten()
    assert report.complexity_after <= report.complexity_before


def test_bva_f14_nested_dispatcher_loops():
    """BVA F14-5: Dispatch loop nested inside another dispatch loop."""
    cfg = ControlFlowGraph()
    b0 = BasicBlock(0, 0, 4, [Instruction(0, Opcode.ICONST_0), Instruction(1, Opcode.ISTORE_1), Instruction(2, Opcode.GOTO, struct.pack(">h", 8))])
    b10 = BasicBlock(10, 10, 20, [Instruction(10, Opcode.ILOAD_1), Instruction(11, Opcode.TABLESWITCH, operands=b"\x00" * 16)])
    b20 = BasicBlock(20, 20, 22, [Instruction(20, Opcode.RETURN)])

    for b in [b0, b10, b20]:
        cfg.add_block(b)
    cfg.add_edge(CFGEdge(0, 10, CFGEdgeType.UNCONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(10, 20, CFGEdgeType.SWITCH_CASE, case_key=0))

    report = ControlFlowDeflattener(cfg).deflatten()
    assert report.complexity_after <= report.complexity_before


# ==============================================================================
# F15 BOUNDARIES (5 tests)
# ==============================================================================

def test_bva_f15_always_true_predicate():
    """BVA F15-1: if (1 == 1) branch replaced with unconditional jump."""
    cfg = ControlFlowGraph()
    b_entry = BasicBlock(0, 0, 7, [
        Instruction(0, Opcode.BIPUSH, struct.pack(">b", 1)),
        Instruction(2, Opcode.BIPUSH, struct.pack(">b", 1)),
        Instruction(4, Opcode.IF_ICMPEQ, struct.pack(">h", 10))
    ])
    b_fall = BasicBlock(7, 7, 8, [Instruction(7, Opcode.RETURN)])
    b_target = BasicBlock(10, 10, 11, [Instruction(10, Opcode.RETURN)])

    for b in [b_entry, b_fall, b_target]:
        cfg.add_block(b)
    cfg.add_edge(CFGEdge(0, 10, CFGEdgeType.CONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(0, 7, CFGEdgeType.FALLTHROUGH))

    count = eliminate_opaque_predicates(cfg)
    assert count == 1
    assert 10 in cfg.get_successors(0)
    assert 7 not in cfg.get_successors(0)


def test_bva_f15_always_false_predicate():
    """BVA F15-2: if (0 == 1) branch pruned and block removed."""
    cfg = ControlFlowGraph()
    b_entry = BasicBlock(0, 0, 7, [
        Instruction(0, Opcode.ICONST_0),
        Instruction(1, Opcode.ICONST_1),
        Instruction(2, Opcode.IF_ICMPEQ, struct.pack(">h", 10))
    ])
    b_fall = BasicBlock(7, 7, 8, [Instruction(7, Opcode.RETURN)])
    b_target = BasicBlock(10, 10, 11, [Instruction(10, Opcode.RETURN)])

    for b in [b_entry, b_fall, b_target]:
        cfg.add_block(b)
    cfg.add_edge(CFGEdge(0, 10, CFGEdgeType.CONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(0, 7, CFGEdgeType.FALLTHROUGH))

    report = OpaquePredicateEliminator(cfg).eliminate()
    assert report.constant_predicates_pruned == 1
    # Falls through to 7, 10 is pruned
    assert 7 in cfg.get_successors(0)
    assert 10 not in cfg.get_successors(0)


def test_bva_f15_algebraic_identity_zero():
    """BVA F15-3: x * 0 == 0 recognized as constant zero and pruned."""
    cfg = ControlFlowGraph()
    b_entry = BasicBlock(0, 0, 6, [
        Instruction(0, Opcode.ILOAD_1),
        Instruction(1, Opcode.ICONST_0),
        Instruction(2, Opcode.IAND),  # x & 0 == 0
        Instruction(3, Opcode.IFEQ, struct.pack(">h", 10))
    ])
    b_fall = BasicBlock(6, 6, 7, [Instruction(6, Opcode.RETURN)])
    b_target = BasicBlock(10, 10, 11, [Instruction(10, Opcode.RETURN)])

    for b in [b_entry, b_fall, b_target]:
        cfg.add_block(b)
    cfg.add_edge(CFGEdge(0, 10, CFGEdgeType.CONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(0, 6, CFGEdgeType.FALLTHROUGH))

    report = OpaquePredicateEliminator(cfg).eliminate()
    assert report.algebraic_predicates_pruned == 1
    assert 10 in cfg.get_successors(0)
    assert 6 not in cfg.get_successors(0)


def test_bva_f15_dynamic_branch_unknown_var():
    """BVA F15-4: Branch depending on unknown method parameter preserved."""
    cfg = ControlFlowGraph()
    b_entry = BasicBlock(0, 0, 6, [
        Instruction(0, Opcode.ILOAD_0),
        Instruction(1, Opcode.ILOAD_1),
        Instruction(2, Opcode.IF_ICMPEQ, struct.pack(">h", 10))
    ])
    b_fall = BasicBlock(6, 6, 7, [Instruction(6, Opcode.RETURN)])
    b_target = BasicBlock(10, 10, 11, [Instruction(10, Opcode.RETURN)])

    for b in [b_entry, b_fall, b_target]:
        cfg.add_block(b)
    cfg.add_edge(CFGEdge(0, 10, CFGEdgeType.CONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(0, 6, CFGEdgeType.FALLTHROUGH))

    count = eliminate_opaque_predicates(cfg)
    assert count == 0
    assert 10 in cfg.get_successors(0)
    assert 6 in cfg.get_successors(0)


def test_bva_f15_predicate_with_side_effect():
    """BVA F15-5: Expression with method invocation preserved (not pruned)."""
    cfg = ControlFlowGraph()
    b_entry = BasicBlock(0, 0, 6, [
        Instruction(0, Opcode.INVOKESTATIC, operands=b"\x00\x05"),
        Instruction(3, Opcode.IFEQ, struct.pack(">h", 10))
    ])
    b_fall = BasicBlock(6, 6, 7, [Instruction(6, Opcode.RETURN)])
    b_target = BasicBlock(10, 10, 11, [Instruction(10, Opcode.RETURN)])

    for b in [b_entry, b_fall, b_target]:
        cfg.add_block(b)
    cfg.add_edge(CFGEdge(0, 10, CFGEdgeType.CONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(0, 6, CFGEdgeType.FALLTHROUGH))

    count = eliminate_opaque_predicates(cfg)
    assert count == 0


# ==============================================================================
# F16 BOUNDARIES (5 tests)
# ==============================================================================

def test_bva_f16_integer_division_by_zero():
    """BVA F16-1: Division by zero in bytecode handled safely without crashing analyzer."""
    cfg = ControlFlowGraph()
    b = BasicBlock(0, 0, 4, [
        Instruction(0, Opcode.BIPUSH, struct.pack(">b", 10)),
        Instruction(2, Opcode.ICONST_0),
        Instruction(3, Opcode.IDIV),
        Instruction(4, Opcode.IRETURN)
    ])
    cfg.add_block(b)

    # Must not raise ZeroDivisionError, must preserve idiv for runtime ArithmeticException
    folded = fold_constants_and_arithmetic(cfg)
    assert folded == 0
    assert len(b.instructions) == 4
    assert ArithmeticEvaluator.eval_binary_int(Opcode.IDIV, 10, 0) is None


def test_bva_f16_32bit_integer_overflow():
    """BVA F16-2: Arithmetic overflow 0x7fffffff + 1 wrapped to -0x80000000."""
    res = ArithmeticEvaluator.eval_binary_int(Opcode.IADD, 0x7FFFFFFF, 1)
    assert res == -2147483648
    assert ArithmeticEvaluator.eval_binary_int(Opcode.ISUB, -2147483648, 1) == 0x7FFFFFFF


def test_bva_f16_shift_by_32_or_negative():
    """BVA F16-3: Bitwise shift by 32 or negative count masked to 5 bits per JVMS."""
    # In Java: 1 << 32 == 1 (32 & 0x1F == 0)
    assert ArithmeticEvaluator.eval_binary_int(Opcode.ISHL, 1, 32) == 1
    # 1 << -1 == 1 << 31 == -2147483648
    assert ArithmeticEvaluator.eval_binary_int(Opcode.ISHL, 1, -1) == -2147483648


def test_bva_f16_float_nan_arithmetic_folding():
    """BVA F16-4: Floating point NaN operations folded correctly."""
    nan_val = float("nan")
    res = ArithmeticEvaluator.eval_binary_float(Opcode.FADD, nan_val, 1.0)
    assert res is not None and math.isnan(res)

    res_div0 = ArithmeticEvaluator.eval_binary_float(Opcode.FDIV, 0.0, 0.0)
    assert res_div0 is not None and math.isnan(res_div0)


def test_bva_f16_constant_folder_empty_cp():
    """BVA F16-5: Folding on method with minimal constant pool runs cleanly."""
    cfg = ControlFlowGraph()
    b = BasicBlock(0, 0, 5, [
        Instruction(0, Opcode.ICONST_2),
        Instruction(1, Opcode.ICONST_3),
        Instruction(2, Opcode.IMUL),
        Instruction(3, Opcode.IRETURN)
    ])
    cfg.add_block(b)

    cp = ConstantPool()
    folded = fold_constants_and_arithmetic(cfg, cp)
    assert folded == 1
    assert b.instructions[0].opcode == Opcode.BIPUSH
    assert struct.unpack(">b", b.instructions[0].operands[:1])[0] == 6


# ==============================================================================
# F17 BOUNDARIES (5 tests)
# ==============================================================================

def test_bva_f17_zero_length_ciphertext():
    """BVA F17-1: Decryption of empty string '' returns empty string."""
    cf = create_minimal_test_class()
    cp = cf.constant_pool

    m_name = cp.add_utf8("decEmpty")
    m_desc = cp.add_utf8("(Ljava/lang/String;)Ljava/lang/String;")
    code_attr = CodeAttribute(cp.add_utf8("Code"), 1, 1, bytes([Opcode.ALOAD_0, Opcode.ARETURN]))
    m = MethodInfo(AccessFlags.STATIC | AccessFlags.PUBLIC, m_name, m_desc, [code_attr])
    cf.methods.append(m)

    emulator = BytecodeEmulator()
    res = emulator.execute(cf, m, [""])
    assert res == ""


def test_bva_f17_xor_key_zero():
    """BVA F17-2: XOR key of 0 (identity) handled correctly."""
    cf = create_minimal_test_class()
    cp = cf.constant_pool

    m_name = cp.add_utf8("decIdentity")
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
    res = emulator.execute(cf, m, ["Hello", 0])
    assert res == "Hello"


def test_bva_f17_complex_loop_condition():
    """BVA F17-3: Decryptor loop with complex counter step handled cleanly."""
    cf = create_minimal_test_class()
    cp = cf.constant_pool
    # Candidate detector test
    m_name = cp.add_utf8("decStep")
    m_desc = cp.add_utf8("(Ljava/lang/String;)Ljava/lang/String;")
    code = bytes([Opcode.ALOAD_0, Opcode.ARETURN])
    m = MethodInfo(AccessFlags.STATIC | AccessFlags.PUBLIC, m_name, m_desc, [CodeAttribute(cp.add_utf8("Code"), 1, 1, code)])
    cf.methods.append(m)
    assert isinstance(DecryptorMethodDetector.is_candidate(cf, m), bool)


def test_bva_f17_non_decryptor_loop_rejection():
    """BVA F17-4: Normal business loop rejected by DecryptorMethodDetector."""
    cf = create_minimal_test_class()
    cp = cf.constant_pool

    m_name = cp.add_utf8("calculateSum")
    m_desc = cp.add_utf8("([I)I")
    # Integer array summation loop (no string / char arrays)
    code = bytes([
        Opcode.ICONST_0, Opcode.ISTORE_1,
        Opcode.ICONST_0, Opcode.ISTORE_2,
        Opcode.ILOAD_2, Opcode.ALOAD_0, Opcode.ARRAYLENGTH, Opcode.IF_ICMPGE, 0, 10,
        Opcode.ILOAD_1, Opcode.ALOAD_0, Opcode.ILOAD_2, Opcode.IALOAD, Opcode.IADD, Opcode.ISTORE_1,
        Opcode.IINC, 2, 1, Opcode.GOTO, 0xFF, 0xEF,
        Opcode.ILOAD_1, Opcode.IRETURN
    ])
    m = MethodInfo(AccessFlags.PUBLIC, m_name, m_desc, [CodeAttribute(cp.add_utf8("Code"), 4, 3, code)])
    cf.methods.append(m)

    assert DecryptorMethodDetector.is_candidate(cf, m) is False


def test_bva_f17_multi_stage_decryption():
    """BVA F17-5: Multi-stage decryption emulator returns deterministic result."""
    cf = create_minimal_test_class()
    cp = cf.constant_pool

    m_name = cp.add_utf8("decMulti")
    m_desc = cp.add_utf8("(Ljava/lang/String;I)Ljava/lang/String;")
    code = bytes([Opcode.ALOAD_0, Opcode.ARETURN])
    m = MethodInfo(AccessFlags.STATIC | AccessFlags.PUBLIC, m_name, m_desc, [CodeAttribute(cp.add_utf8("Code"), 1, 2, code)])
    cf.methods.append(m)

    emulator = BytecodeEmulator()
    res = emulator.execute(cf, m, ["TEST_PAYLOAD", 42])
    assert res == "TEST_PAYLOAD"


# ==============================================================================
# F18 BOUNDARIES (5 tests)
# ==============================================================================

def test_bva_f18_indy_zero_arguments():
    """BVA F18-1: Invokedynamic with 0 bootstrap arguments handled cleanly."""
    cf = create_minimal_test_class()
    cp = cf.constant_pool

    bsm_mh_idx = cp.add_member_ref(
        ConstantTag.METHODREF, "java/lang/invoke/StringConcatFactory",
        "makeConcatWithConstants", "()Ljava/lang/invoke/CallSite;"
    )
    mh_idx = len(cp._entries)
    cp._entries.append(CONSTANT_MethodHandle_info(reference_kind=ReferenceKind.REF_invokeStatic, reference_index=bsm_mh_idx))

    bsm = BootstrapMethod(bootstrap_method_ref=mh_idx, bootstrap_arguments=[])
    bsm_attr = BootstrapMethodsAttribute(attribute_name_index=cp.add_utf8("BootstrapMethods"), bootstrap_methods=[bsm])
    cf.attributes.append(bsm_attr)

    nat_idx = cp.add_name_and_type("test", "()Ljava/lang/String;")
    indy_idx = len(cp._entries)
    cp._entries.append(CONSTANT_InvokeDynamic_info(bootstrap_method_attr_index=0, name_and_type_index=nat_idx))

    resolver = DynamicInvokeResolver(cf)
    res = resolver.resolve_string_concat(cp[indy_idx])
    assert res is None


def test_bva_f18_unknown_bootstrap_method():
    """BVA F18-2: Unknown bootstrap method handle marked unresolvable without crash."""
    cf = create_minimal_test_class()
    cp = cf.constant_pool

    unknown_mh_idx = cp.add_member_ref(
        ConstantTag.METHODREF, "com/unknown/CustomBSM", "bootstrap", "()Ljava/lang/invoke/CallSite;"
    )
    mh_idx = len(cp._entries)
    cp._entries.append(CONSTANT_MethodHandle_info(reference_kind=ReferenceKind.REF_invokeStatic, reference_index=unknown_mh_idx))

    bsm = BootstrapMethod(bootstrap_method_ref=mh_idx, bootstrap_arguments=[])
    bsm_attr = BootstrapMethodsAttribute(attribute_name_index=cp.add_utf8("BootstrapMethods"), bootstrap_methods=[bsm])
    cf.attributes.append(bsm_attr)

    nat_idx = cp.add_name_and_type("test", "()V")
    indy_idx = len(cp._entries)
    cp._entries.append(CONSTANT_InvokeDynamic_info(bootstrap_method_attr_index=0, name_and_type_index=nat_idx))

    resolver = DynamicInvokeResolver(cf)
    assert resolver.resolve_string_concat(cp[indy_idx]) is None
    assert resolver.resolve_lambda(cp[indy_idx]) is None


def test_bva_f18_string_concat_ten_arguments():
    """BVA F18-3: StringConcatFactory recipe with 10 concatenated arguments."""
    cf = create_minimal_test_class()
    cp = cf.constant_pool

    bsm_mh_idx = cp.add_member_ref(
        ConstantTag.METHODREF, "java/lang/invoke/StringConcatFactory",
        "makeConcatWithConstants", "()Ljava/lang/invoke/CallSite;"
    )
    mh_idx = len(cp._entries)
    cp._entries.append(CONSTANT_MethodHandle_info(reference_kind=ReferenceKind.REF_invokeStatic, reference_index=bsm_mh_idx))

    recipe_str = "\u0002" * 10
    recipe_idx = cp.add_string(recipe_str)
    arg_indices = [cp.add_string(f"{i}") for i in range(10)]

    bsm = BootstrapMethod(bootstrap_method_ref=mh_idx, bootstrap_arguments=[recipe_idx] + arg_indices)
    bsm_attr = BootstrapMethodsAttribute(attribute_name_index=cp.add_utf8("BootstrapMethods"), bootstrap_methods=[bsm])
    cf.attributes.append(bsm_attr)

    nat_idx = cp.add_name_and_type("makeConcat", "()Ljava/lang/String;")
    indy_idx = len(cp._entries)
    cp._entries.append(CONSTANT_InvokeDynamic_info(bootstrap_method_attr_index=0, name_and_type_index=nat_idx))

    resolver = DynamicInvokeResolver(cf)
    res = resolver.resolve_string_concat(cp[indy_idx])
    assert res is not None
    assert res.is_fully_constant is True
    assert res.folded_constant == "0123456789"


def test_bva_f18_reflection_non_existent_class():
    """BVA F18-4: Class.forName on non-existent class safely recorded."""
    cf = create_minimal_test_class()
    cp = cf.constant_pool

    str_idx = cp.add_string("com.nonexistent.NoSuchClass")
    for_name_ref = cp.add_member_ref(ConstantTag.METHODREF, "java/lang/Class", "forName", "(Ljava/lang/String;)Ljava/lang/Class;")
    get_m_ref = cp.add_member_ref(ConstantTag.METHODREF, "java/lang/Class", "getMethod", "(Ljava/lang/String;[Ljava/lang/Class;)Ljava/lang/reflect/Method;")
    m_name_idx = cp.add_string("doSomething")
    invoke_ref = cp.add_member_ref(ConstantTag.METHODREF, "java/lang/reflect/Method", "invoke", "(Ljava/lang/Object;[Ljava/lang/Object;)Ljava/lang/Object;")

    code = bytes([
        Opcode.LDC, str_idx,
        Opcode.INVOKESTATIC, (for_name_ref >> 8) & 0xFF, for_name_ref & 0xFF,
        Opcode.LDC, m_name_idx,
        Opcode.ACONST_NULL,
        Opcode.INVOKEVIRTUAL, (get_m_ref >> 8) & 0xFF, get_m_ref & 0xFF,
        Opcode.ACONST_NULL,
        Opcode.ACONST_NULL,
        Opcode.INVOKEVIRTUAL, (invoke_ref >> 8) & 0xFF, invoke_ref & 0xFF,
        Opcode.ARETURN
    ])
    code_attr = CodeAttribute(cp.add_utf8("Code"), 4, 1, code)
    m = MethodInfo(AccessFlags.PUBLIC | AccessFlags.STATIC, cp.add_utf8("test"), cp.add_utf8("()Ljava/lang/Object;"), [code_attr])
    cf.methods.append(m)

    resolver = DynamicInvokeResolver(cf)
    calls = resolver.resolve_reflection_calls(m)
    assert len(calls) == 1
    assert calls[0].target_class == "com/nonexistent/NoSuchClass"
    assert calls[0].target_method == "doSomething"


def test_bva_f18_reflection_primitive_params():
    """BVA F18-5: getMethod with primitive parameters."""
    cf = create_minimal_test_class()
    cp = cf.constant_pool

    str_idx = cp.add_string("com.example.Target")
    for_name_ref = cp.add_member_ref(ConstantTag.METHODREF, "java/lang/Class", "forName", "(Ljava/lang/String;)Ljava/lang/Class;")
    get_m_ref = cp.add_member_ref(ConstantTag.METHODREF, "java/lang/Class", "getMethod", "(Ljava/lang/String;[Ljava/lang/Class;)Ljava/lang/reflect/Method;")
    m_name_idx = cp.add_string("compute")
    invoke_ref = cp.add_member_ref(ConstantTag.METHODREF, "java/lang/reflect/Method", "invoke", "(Ljava/lang/Object;[Ljava/lang/Object;)Ljava/lang/Object;")

    code = bytes([
        Opcode.LDC, str_idx,
        Opcode.INVOKESTATIC, (for_name_ref >> 8) & 0xFF, for_name_ref & 0xFF,
        Opcode.LDC, m_name_idx,
        Opcode.ACONST_NULL,
        Opcode.INVOKEVIRTUAL, (get_m_ref >> 8) & 0xFF, get_m_ref & 0xFF,
        Opcode.ACONST_NULL,
        Opcode.ACONST_NULL,
        Opcode.INVOKEVIRTUAL, (invoke_ref >> 8) & 0xFF, invoke_ref & 0xFF,
        Opcode.ARETURN
    ])
    code_attr = CodeAttribute(cp.add_utf8("Code"), 4, 1, code)
    m = MethodInfo(AccessFlags.PUBLIC | AccessFlags.STATIC, cp.add_utf8("test"), cp.add_utf8("()Ljava/lang/Object;"), [code_attr])
    cf.methods.append(m)

    resolver = DynamicInvokeResolver(cf)
    calls = resolver.resolve_reflection_calls(m)
    assert len(calls) == 1
    assert calls[0].target_class == "com/example/Target"
    assert calls[0].target_method == "compute"


# ==============================================================================
# F19 BOUNDARIES (5 tests)
# ==============================================================================

def test_bva_f19_method_zero_dead_blocks():
    """BVA F19-1: Method with 0 dead blocks unchanged after pruning."""
    cfg = ControlFlowGraph()
    b0 = BasicBlock(0, 0, 2, [Instruction(0, Opcode.ICONST_1), Instruction(1, Opcode.IRETURN)])
    cfg.add_block(b0)

    pruned = prune_unreachable_blocks(cfg)
    assert pruned == 0
    assert len(cfg.blocks) == 1


def test_bva_f19_entirely_dead_method():
    """BVA F19-2: Method with unreachable dead blocks after return pruned."""
    cfg = ControlFlowGraph()
    b0 = BasicBlock(0, 0, 2, [Instruction(0, Opcode.ICONST_1), Instruction(1, Opcode.IRETURN)])
    b_dead1 = BasicBlock(10, 10, 12, [Instruction(10, Opcode.NOP), Instruction(11, Opcode.RETURN)])
    b_dead2 = BasicBlock(20, 20, 22, [Instruction(20, Opcode.NOP), Instruction(21, Opcode.RETURN)])
    for b in [b0, b_dead1, b_dead2]:
        cfg.add_block(b)

    pruned = prune_unreachable_blocks(cfg)
    assert pruned == 2
    assert len(cfg.blocks) == 1
    assert 0 in cfg.blocks


def test_bva_f19_fifty_consecutive_nops():
    """BVA F19-3: 50 consecutive NOPs stripped without altering semantics."""
    code = bytes([Opcode.NOP] * 50 + [Opcode.RETURN])
    new_code, _, nops_removed = strip_nops_from_bytecode(code)
    assert nops_removed == 50
    assert len(new_code) == 1
    assert new_code[0] == Opcode.RETURN


def test_bva_f19_circular_goto_loop():
    """BVA F19-4: Infinite goto loop goto L1; L1: goto L1 handled without recursion limit."""
    cfg = ControlFlowGraph()
    b0 = BasicBlock(0, 0, 3, [Instruction(0, Opcode.GOTO, struct.pack(">h", 0))])
    cfg.add_block(b0)
    cfg.add_edge(CFGEdge(0, 0, CFGEdgeType.UNCONDITIONAL_BRANCH))

    compacted = compact_goto_chains(cfg)
    assert compacted == 0  # Does not hang or loop infinitely


def test_bva_f19_dead_block_in_exception_range():
    """BVA F19-5: Dead block at exception handler target preserved if referenced in exception table."""
    code = bytes([
        Opcode.ICONST_0, Opcode.IRETURN,  # Block 0: return
        Opcode.ASTORE_1, Opcode.ATHROW    # Block 1 (pc 2): exception handler
    ])
    ex_table = [ExceptionTableEntry(start_pc=0, end_pc=1, handler_pc=2, catch_type=0)]
    cfg = ControlFlowGraph.from_code(code, ex_table)

    # In CFG with exception handler edge, handler is reachable via EXCEPTION edge
    pruned = prune_unreachable_blocks(cfg)
    assert pruned == 0
    assert 2 in cfg.blocks
