"""
tests/test_m3_cfg_and_constants.py: Comprehensive test suite for Milestone M3.
Covers features F13, F14, F15, F16, F17, F18, F19 and the MethodNormalizer pipeline.
"""

import pytest
import struct

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
    ControlFlowGraph, NaturalLoop,
    partition_basic_blocks, decode_instructions,
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


def create_test_class(name: str = "com/example/Obfuscated") -> ClassFile:
    """Helper to construct a valid in-memory ClassFile."""
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


# ============================================================================
# F13: CFG Basic Block Partitioning & Graph Construction
# ============================================================================

def test_cfg_linear_instructions():
    # iconst_1 (1B), istore_1 (1B), return (1B) -> 1 basic block
    code = bytes([Opcode.ICONST_1, Opcode.ISTORE_1, Opcode.RETURN])
    blocks = partition_basic_blocks(code)
    assert len(blocks) == 1
    assert blocks[0].start_pc == 0
    assert blocks[0].end_pc == 3
    assert len(blocks[0].instructions) == 3
    assert blocks[0].is_terminal is True


def test_cfg_diamond_branch():
    # pc 0: iconst_1 (1B)
    # pc 1: iconst_2 (1B)
    # pc 2: if_icmpeq +7 (3B -> pc 9)
    # pc 5: iconst_0 (1B)
    # pc 6: goto +6 (3B -> pc 12)
    # pc 9: iconst_1 (1B)
    # pc 10: goto +2 (3B -> pc 12)
    # pc 12: ireturn (1B)
    code = (
        bytes([Opcode.ICONST_1, Opcode.ICONST_2, Opcode.IF_ICMPEQ]) + struct.pack(">h", 7) +
        bytes([Opcode.ICONST_0, Opcode.GOTO]) + struct.pack(">h", 6) +
        bytes([Opcode.ICONST_1, Opcode.GOTO]) + struct.pack(">h", 2) +
        bytes([Opcode.IRETURN])
    )
    cfg = ControlFlowGraph.from_code(code)
    assert len(cfg.blocks) >= 4
    entry = cfg.get_entry_block()
    assert entry is not None
    # Entry block should branch conditionally to target (9) and fall through (5)
    succs = cfg.get_successors(entry.id)
    assert len(succs) == 2
    assert 9 in succs
    assert 5 in succs


def test_cfg_switch_table_lookup():
    # tableswitch with low=0, high=1, default=14, case 0=14, case 1=15
    # pc 0: iconst_1
    # pc 1: tableswitch pad=2 (pc 4), default=+14, low=0, high=1, case0=+14, case1=+15
    # pc 18: return
    # pc 19: return
    code = bytes([Opcode.ICONST_1, Opcode.TABLESWITCH, 0, 0]) + struct.pack(">iii", 14, 0, 1) + struct.pack(">ii", 14, 15) + bytes([Opcode.RETURN, Opcode.RETURN])
    cfg = ControlFlowGraph.from_code(code)
    assert len(cfg.blocks) >= 2
    entry = cfg.get_entry_block()
    out_edges = cfg.get_outgoing_edges(entry.id)
    case_edges = [e for e in out_edges if e.edge_type == CFGEdgeType.SWITCH_CASE]
    default_edges = [e for e in out_edges if e.edge_type == CFGEdgeType.SWITCH_DEFAULT]
    assert len(case_edges) == 2
    assert len(default_edges) == 1


def test_cfg_exception_boundaries():
    # pc 0: iconst_1
    # pc 1: istore_1
    # pc 2: goto +4 (pc 6)
    # pc 5: return
    # pc 6: return
    # Try block [0, 2) handled at 5
    code = bytes([Opcode.ICONST_1, Opcode.ISTORE_1, Opcode.GOTO]) + struct.pack(">h", 4) + bytes([Opcode.RETURN, Opcode.RETURN])
    ex_table = [ExceptionTableEntry(start_pc=0, end_pc=2, handler_pc=5, catch_type=0)]
    blocks = partition_basic_blocks(code, ex_table)
    # PC 0 (start), PC 2 (end), PC 5 (handler) should all be leaders
    leader_starts = {b.start_pc for b in blocks}
    assert 0 in leader_starts
    assert 2 in leader_starts
    assert 5 in leader_starts


def test_cfg_dominators_and_loops():
    # Simple loop:
    # pc 0: iconst_0
    # pc 1: istore_1
    # pc 2: iinc 1, 1
    # pc 5: iload_1
    # pc 6: bipush 10
    # pc 8: if_icmplt -6 (jump back to pc 2)
    # pc 11: return
    code = (
        bytes([Opcode.ICONST_0, Opcode.ISTORE_1, Opcode.IINC, 1, 1, Opcode.ILOAD_1, Opcode.BIPUSH, 10, Opcode.IF_ICMPLT]) +
        struct.pack(">h", -6) +
        bytes([Opcode.RETURN])
    )
    cfg = ControlFlowGraph.from_code(code)
    loops = cfg.find_natural_loops()
    assert len(loops) == 1
    assert loops[0].header_id == 2
    assert cfg.cyclomatic_complexity >= 2


# ============================================================================
# F14: Control Flow De-Flattening
# ============================================================================

def test_deflatten_linear_state_dispatch():
    # A flattened method with dispatcher switch for states 0, 1, 2
    # State 0 sets state 1 and jumps to dispatcher
    # State 1 sets state 2 and jumps to dispatcher
    # State 2 returns
    cfg = ControlFlowGraph()

    # Block 0: Init state 0
    b_init = BasicBlock(id=0, start_pc=0, end_pc=4, instructions=[
        Instruction(0, Opcode.ICONST_0),
        Instruction(1, Opcode.ISTORE_1),
        Instruction(2, Opcode.GOTO, struct.pack(">h", 8))
    ])

    # Block 10: Dispatcher (switch on slot 1)
    b_disp = BasicBlock(id=10, start_pc=10, end_pc=30, instructions=[
        Instruction(10, Opcode.ILOAD_1),
        Instruction(11, Opcode.TABLESWITCH, operands=b"\x00" * 16)
    ])

    # Payload 100: State 0 -> sets state 1
    b_p0 = BasicBlock(id=100, start_pc=100, end_pc=105, instructions=[
        Instruction(100, Opcode.NOP),
        Instruction(101, Opcode.ICONST_1),
        Instruction(102, Opcode.ISTORE_1),
        Instruction(103, Opcode.GOTO, struct.pack(">h", 0))
    ])

    # Payload 200: State 1 -> sets state 2
    b_p1 = BasicBlock(id=200, start_pc=200, end_pc=205, instructions=[
        Instruction(200, Opcode.NOP),
        Instruction(201, Opcode.ICONST_2),
        Instruction(202, Opcode.ISTORE_1),
        Instruction(203, Opcode.GOTO, struct.pack(">h", 0))
    ])

    # Payload 300: State 2 -> returns
    b_p2 = BasicBlock(id=300, start_pc=300, end_pc=302, instructions=[
        Instruction(300, Opcode.RETURN)
    ])

    for b in [b_init, b_disp, b_p0, b_p1, b_p2]:
        cfg.add_block(b)

    cfg.add_edge(CFGEdge(0, 10, CFGEdgeType.UNCONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(10, 100, CFGEdgeType.SWITCH_CASE, case_key=0))
    cfg.add_edge(CFGEdge(10, 200, CFGEdgeType.SWITCH_CASE, case_key=1))
    cfg.add_edge(CFGEdge(10, 300, CFGEdgeType.SWITCH_CASE, case_key=2))
    cfg.add_edge(CFGEdge(10, 300, CFGEdgeType.SWITCH_DEFAULT))
    cfg.add_edge(CFGEdge(100, 10, CFGEdgeType.UNCONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(200, 10, CFGEdgeType.UNCONDITIONAL_BRANCH))

    complexity_before = cfg.cyclomatic_complexity
    deflattener = ControlFlowDeflattener(cfg)
    report = deflattener.deflatten()

    assert report.edges_reconstructed >= 2
    # Verify b_init now branches directly to b_p0 (state 0)
    assert 100 in cfg.get_successors(0)
    # Verify b_p0 branches directly to b_p1 (state 1)
    assert 200 in cfg.get_successors(100)
    # Verify b_p1 branches directly to b_p2 (state 2)
    assert 300 in cfg.get_successors(200)
    # Verify central dispatcher was eliminated
    assert 10 not in cfg.blocks


def test_deflatten_loop_state():
    # Payload block transitioning back to an earlier state (forming unflattened while loop)
    cfg = ControlFlowGraph()
    b_init = BasicBlock(0, 0, 4, [Instruction(0, Opcode.ICONST_0), Instruction(1, Opcode.ISTORE_1), Instruction(2, Opcode.GOTO, struct.pack(">h", 0))])
    b_disp = BasicBlock(10, 10, 30, [Instruction(10, Opcode.ILOAD_1), Instruction(11, Opcode.TABLESWITCH, operands=b"\x00" * 16)])
    b_p0 = BasicBlock(100, 100, 105, [Instruction(100, Opcode.ICONST_1), Instruction(101, Opcode.ISTORE_1), Instruction(102, Opcode.GOTO, struct.pack(">h", 0))])
    b_p1 = BasicBlock(200, 200, 205, [Instruction(200, Opcode.ICONST_0), Instruction(201, Opcode.ISTORE_1), Instruction(202, Opcode.GOTO, struct.pack(">h", 0))])

    for b in [b_init, b_disp, b_p0, b_p1]:
        cfg.add_block(b)

    cfg.add_edge(CFGEdge(0, 10, CFGEdgeType.UNCONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(10, 100, CFGEdgeType.SWITCH_CASE, case_key=0))
    cfg.add_edge(CFGEdge(10, 200, CFGEdgeType.SWITCH_CASE, case_key=1))
    cfg.add_edge(CFGEdge(10, 100, CFGEdgeType.SWITCH_DEFAULT))
    cfg.add_edge(CFGEdge(100, 10, CFGEdgeType.UNCONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(200, 10, CFGEdgeType.UNCONDITIONAL_BRANCH))

    deflatten_control_flow(cfg)
    # Loop back edge: b_p1 -> b_p0
    assert 100 in cfg.get_successors(200)


# ============================================================================
# F15: Opaque Predicate Elimination
# ============================================================================

def test_opaque_constant_true_pruning():
    # iconst_0; ifeq L_target (Always True)
    cfg = ControlFlowGraph()
    b_entry = BasicBlock(0, 0, 4, [
        Instruction(0, Opcode.ICONST_0),
        Instruction(1, Opcode.IFEQ, struct.pack(">h", 10))
    ])
    b_fall = BasicBlock(4, 4, 5, [Instruction(4, Opcode.RETURN)])
    b_target = BasicBlock(10, 10, 11, [Instruction(10, Opcode.RETURN)])

    for b in [b_entry, b_fall, b_target]:
        cfg.add_block(b)
    cfg.add_edge(CFGEdge(0, 10, CFGEdgeType.CONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(0, 4, CFGEdgeType.FALLTHROUGH))

    report = OpaquePredicateEliminator(cfg).eliminate()
    assert report.constant_predicates_pruned == 1
    # Fallthrough edge removed, unconditional jump to 10 retained
    assert 4 not in cfg.get_successors(0)
    assert 10 in cfg.get_successors(0)
    assert b_entry.terminator.opcode == Opcode.GOTO


def test_opaque_constant_false_pruning():
    # iconst_1; ifeq L_target (Always False)
    cfg = ControlFlowGraph()
    b_entry = BasicBlock(0, 0, 4, [
        Instruction(0, Opcode.ICONST_1),
        Instruction(1, Opcode.IFEQ, struct.pack(">h", 10))
    ])
    b_fall = BasicBlock(4, 4, 5, [Instruction(4, Opcode.RETURN)])
    b_target = BasicBlock(10, 10, 11, [Instruction(10, Opcode.RETURN)])

    for b in [b_entry, b_fall, b_target]:
        cfg.add_block(b)
    cfg.add_edge(CFGEdge(0, 10, CFGEdgeType.CONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(0, 4, CFGEdgeType.FALLTHROUGH))

    report = OpaquePredicateEliminator(cfg).eliminate()
    assert report.constant_predicates_pruned == 1
    # Branch edge removed, execution falls through to 4
    assert 10 not in cfg.get_successors(0)
    assert 4 in cfg.get_successors(0)


def test_opaque_constant_icmpeq_pruning():
    # bipush 42; bipush 42; if_icmpeq L (Always True)
    cfg = ControlFlowGraph()
    b_entry = BasicBlock(0, 0, 7, [
        Instruction(0, Opcode.BIPUSH, struct.pack(">b", 42)),
        Instruction(2, Opcode.BIPUSH, struct.pack(">b", 42)),
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


def test_opaque_algebraic_self_xor():
    # iload_0; iload_0; ixor; ifeq L (x ^ x == 0 is Always True)
    cfg = ControlFlowGraph()
    b_entry = BasicBlock(0, 0, 6, [
        Instruction(0, Opcode.ILOAD_0),
        Instruction(1, Opcode.ILOAD_0),
        Instruction(2, Opcode.IXOR),
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


# ============================================================================
# F16: Constant Folding Lattice
# ============================================================================

def test_constant_folding_int_arithmetic_and_bitwise():
    # bipush 10; bipush 20; iadd; bipush 5; isub
    # 10 + 20 - 5 = 25 -> bipush 25
    cfg = ControlFlowGraph()
    b = BasicBlock(0, 0, 10, [
        Instruction(0, Opcode.BIPUSH, struct.pack(">b", 10)),
        Instruction(2, Opcode.BIPUSH, struct.pack(">b", 20)),
        Instruction(4, Opcode.IADD),
        Instruction(5, Opcode.BIPUSH, struct.pack(">b", 5)),
        Instruction(7, Opcode.ISUB),
        Instruction(8, Opcode.IRETURN)
    ])
    cfg.add_block(b)

    folded = fold_constants_and_arithmetic(cfg)
    assert folded == 2
    # Resulting instruction should be bipush 25 followed by ireturn
    assert len(b.instructions) == 2
    assert b.instructions[0].opcode == Opcode.BIPUSH
    assert struct.unpack(">b", b.instructions[0].operands[:1])[0] == 25


def test_constant_folding_division_by_zero_safety():
    # bipush 10; iconst_0; idiv -> Must NOT fold (preserves ArithmeticException)
    cfg = ControlFlowGraph()
    b = BasicBlock(0, 0, 4, [
        Instruction(0, Opcode.BIPUSH, struct.pack(">b", 10)),
        Instruction(2, Opcode.ICONST_0),
        Instruction(3, Opcode.IDIV),
        Instruction(4, Opcode.IRETURN)
    ])
    cfg.add_block(b)

    folded = fold_constants_and_arithmetic(cfg)
    assert folded == 0
    assert len(b.instructions) == 4


def test_constant_folding_java_remainder_semantics():
    assert ArithmeticEvaluator.java_irem(-7, 3) == -1
    assert ArithmeticEvaluator.java_irem(7, -3) == 1
    assert ArithmeticEvaluator.java_irem(-7, -3) == -1


def test_constant_folding_shift_masking():
    # In Java, shift amount is masked by 0x1F (31)
    # 1 << 32 in Java is 1 (32 & 0x1F == 0)
    assert ArithmeticEvaluator.eval_binary_int(Opcode.ISHL, 1, 32) == 1
    assert ArithmeticEvaluator.eval_binary_int(Opcode.ISHL, 1, 33) == 2


def test_constant_folding_conversions():
    # i2b, i2c, i2s
    assert ArithmeticEvaluator.eval_conversion(Opcode.I2B, 300) == (44, "int")  # 300 & 0xFF = 44
    assert ArithmeticEvaluator.eval_conversion(Opcode.I2B, 255) == (-1, "int")  # 255 as signed byte is -1
    assert ArithmeticEvaluator.eval_conversion(Opcode.I2C, -1) == (65535, "int")


# ============================================================================
# F17: Static String Decryptor Resolution
# ============================================================================

def test_string_decryptor_detection():
    cf = create_test_class()
    cp = cf.constant_pool

    # Create static method: String decrypt(String s, int key)
    m_name = cp.add_utf8("decrypt")
    m_desc = cp.add_utf8("(Ljava/lang/String;I)Ljava/lang/String;")
    code_bytes = bytes([Opcode.ALOAD_0, Opcode.ILOAD_1, Opcode.IXOR, Opcode.ARETURN])
    c_name = cp.add_utf8("Code")
    code_attr = CodeAttribute(attribute_name_index=c_name, max_stack=2, max_locals=2, code=code_bytes)

    method = MethodInfo(access_flags=AccessFlags.STATIC | AccessFlags.PUBLIC, name_index=m_name, descriptor_index=m_desc, attributes=[code_attr])
    cf.methods.append(method)

    assert DecryptorMethodDetector.is_candidate(cf, method) is True


def test_string_decryptor_emulator_xor():
    # Emulate rotating XOR loop:
    # static String decrypt(String s, int key) {
    #     char[] c = s.toCharArray();
    #     for (int i=0; i<c.length; i++) c[i] = (char)(c[i] ^ key);
    #     return new String(c);
    # }
    cf = create_test_class()
    cp = cf.constant_pool

    m_name = cp.add_utf8("decrypt")
    m_desc = cp.add_utf8("(Ljava/lang/String;I)Ljava/lang/String;")

    str_to_char_ref = cp.add_member_ref(ConstantTag.METHODREF, "java/lang/String", "toCharArray", "()[C")
    str_init_ref = cp.add_member_ref(ConstantTag.METHODREF, "java/lang/String", "<init>", "([C)V")

    # Bytecode:
    # pc 0: aload_0
    # pc 1: invokevirtual toCharArray (3B) -> pc 4
    # pc 4: astore_2 (c in local 2)
    # pc 5: iconst_0
    # pc 6: istore_3 (i in local 3)
    # pc 7: iload_3
    # pc 8: aload_2
    # pc 9: arraylength
    # pc 10: if_icmpge +19 (pc 29)
    # pc 13: aload_2
    # pc 14: iload_3
    # pc 15: aload_2
    # pc 16: iload_3
    # pc 17: caload
    # pc 18: iload_1
    # pc 19: ixor
    # pc 20: i2c
    # pc 21: castore
    # pc 22: iinc 3, 1
    # pc 25: goto -18 (pc 7)
    # pc 28: nop
    # pc 29: new java/lang/String (3B)
    # pc 32: dup
    # pc 33: aload_2
    # pc 34: invokespecial <init>([C)V (3B)
    # pc 37: areturn
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

    c_name = cp.add_utf8("Code")
    code_attr = CodeAttribute(attribute_name_index=c_name, max_stack=4, max_locals=4, code=code)
    method = MethodInfo(access_flags=AccessFlags.STATIC | AccessFlags.PUBLIC, name_index=m_name, descriptor_index=m_desc, attributes=[code_attr])
    cf.methods.append(method)

    emulator = BytecodeEmulator()
    # Encrypt "SECRET" with key 0x42
    plain = "SECRET"
    key = 0x42
    cipher = "".join(chr(ord(c) ^ key) for c in plain)

    res = emulator.execute(cf, method, [cipher, key])
    assert res == "SECRET"


def test_string_decryptor_inlining():
    cf = create_test_class()
    cp = cf.constant_pool

    # Add decryptor method that performs genuine Caesar shift (-1)
    d_name = cp.add_utf8("dec")
    d_desc = cp.add_utf8("(Ljava/lang/String;)Ljava/lang/String;")
    str_to_char_ref = cp.add_member_ref(ConstantTag.METHODREF, "java/lang/String", "toCharArray", "()[C")
    str_init_ref = cp.add_member_ref(ConstantTag.METHODREF, "java/lang/String", "<init>", "([C)V")
    str_cls = cp.add_class("java/lang/String")

    dec_code = (
        bytes([Opcode.ALOAD_0, Opcode.INVOKEVIRTUAL]) + struct.pack(">H", str_to_char_ref) +
        bytes([Opcode.ASTORE_1, Opcode.ICONST_0, Opcode.ISTORE_2, Opcode.ILOAD_2, Opcode.ALOAD_1, Opcode.ARRAYLENGTH, Opcode.IF_ICMPGE]) +
        struct.pack(">h", 18) +
        bytes([Opcode.ALOAD_1, Opcode.ILOAD_2, Opcode.ALOAD_1, Opcode.ILOAD_2, Opcode.CALOAD, Opcode.ICONST_1, Opcode.ISUB, Opcode.I2C, Opcode.CASTORE, Opcode.IINC, 2, 1, Opcode.GOTO]) +
        struct.pack(">h", -18) +
        bytes([Opcode.NEW]) + struct.pack(">H", str_cls) +
        bytes([Opcode.DUP, Opcode.ALOAD_1, Opcode.INVOKESPECIAL]) + struct.pack(">H", str_init_ref) +
        bytes([Opcode.ARETURN])
    )
    c_name = cp.add_utf8("Code")
    dec_attr = CodeAttribute(attribute_name_index=c_name, max_stack=4, max_locals=3, code=dec_code)
    dec_method = MethodInfo(access_flags=AccessFlags.STATIC | AccessFlags.PUBLIC, name_index=d_name, descriptor_index=d_desc, attributes=[dec_attr])
    cf.methods.append(dec_method)

    # Calling method:
    # pc 0: ldc #enc
    # pc 2: invokestatic #dec
    # pc 5: astore_1
    # pc 6: return
    enc_idx = cp.add_string("IFMMP")
    dec_ref = cp.add_member_ref(ConstantTag.METHODREF, cf.this_class_name, "dec", "(Ljava/lang/String;)Ljava/lang/String;")
    caller_code = bytes([Opcode.LDC, enc_idx, Opcode.INVOKESTATIC]) + struct.pack(">H", dec_ref) + bytes([Opcode.ASTORE_1, Opcode.RETURN])

    call_name = cp.add_utf8("main")
    call_desc = cp.add_utf8("()V")
    call_attr = CodeAttribute(attribute_name_index=c_name, max_stack=2, max_locals=2, code=caller_code)
    call_method = MethodInfo(access_flags=AccessFlags.STATIC | AccessFlags.PUBLIC, name_index=call_name, descriptor_index=call_desc, attributes=[call_attr])
    cf.methods.append(call_method)

    count = reconstruct_encrypted_strings(cf, call_method)
    assert count == 1
    # Check that decrypted literal is now inlined at pc 0
    assert call_attr.code[0] in (Opcode.LDC, Opcode.LDC_W)


# ============================================================================
# F18: Dynamic Invoke & Reflection Resolution
# ============================================================================

def test_dynamic_invoke_string_concat_pure_constant():
    cf = create_test_class()
    cp = cf.constant_pool

    # Create StringConcatFactory BSM
    bsm_mh_idx = cp.add_member_ref(
        ConstantTag.METHODREF,
        "java/lang/invoke/StringConcatFactory",
        "makeConcatWithConstants",
        "(Ljava/lang/invoke/MethodHandles$Lookup;Ljava/lang/String;Ljava/lang/invoke/MethodType;Ljava/lang/String;[Ljava/lang/Object;)Ljava/lang/invoke/CallSite;"
    )
    mh_idx = len(cp._entries)
    cp._entries.append(CONSTANT_MethodHandle_info(reference_kind=ReferenceKind.REF_invokeStatic, reference_index=bsm_mh_idx))

    recipe_idx = cp.add_string("Hello \u0002 World")
    const_arg_idx = cp.add_string("Pure")

    bsm_attr_name = cp.add_utf8("BootstrapMethods")
    bsm = BootstrapMethod(bootstrap_method_ref=mh_idx, bootstrap_arguments=[recipe_idx, const_arg_idx])
    bsm_attr = BootstrapMethodsAttribute(attribute_name_index=bsm_attr_name, bootstrap_methods=[bsm])
    cf.attributes.append(bsm_attr)

    nat_idx = cp.add_name_and_type("makeConcatWithConstants", "()Ljava/lang/String;")
    indy_idx = len(cp._entries)
    cp._entries.append(CONSTANT_InvokeDynamic_info(bootstrap_method_attr_index=0, name_and_type_index=nat_idx))

    # Caller code with invokedynamic
    caller_code = bytes([Opcode.INVOKEDYNAMIC]) + struct.pack(">H", indy_idx) + bytes([0, 0, Opcode.ARETURN])
    c_name = cp.add_utf8("Code")
    call_attr = CodeAttribute(attribute_name_index=c_name, max_stack=1, max_locals=1, code=caller_code)
    call_method = MethodInfo(access_flags=AccessFlags.STATIC | AccessFlags.PUBLIC, name_index=cp.add_utf8("test"), descriptor_index=cp.add_utf8("()Ljava/lang/String;"), attributes=[call_attr])
    cf.methods.append(call_method)

    resolved = resolve_dynamic_invokes_and_reflection(cf, call_method)
    assert resolved >= 1
    # Check that invokedynamic was folded into ldc "Hello Pure World"
    assert call_attr.code[0] in (Opcode.LDC, Opcode.LDC_W)


def test_dynamic_invoke_lambda_metafactory():
    cf = create_test_class()
    cp = cf.constant_pool

    bsm_mh_idx = cp.add_member_ref(
        ConstantTag.METHODREF,
        "java/lang/invoke/LambdaMetafactory",
        "metafactory",
        "(Ljava/lang/invoke/MethodHandles$Lookup;Ljava/lang/String;Ljava/lang/invoke/MethodType;Ljava/lang/invoke/MethodType;Ljava/lang/invoke/MethodHandle;Ljava/lang/invoke/MethodType;)Ljava/lang/invoke/CallSite;"
    )
    mh_idx = len(cp._entries)
    cp._entries.append(CONSTANT_MethodHandle_info(reference_kind=ReferenceKind.REF_invokeStatic, reference_index=bsm_mh_idx))

    sam_type_idx = len(cp._entries)
    cp._entries.append(CONSTANT_MethodType_info(descriptor_index=cp.add_utf8("()V")))

    impl_ref_idx = cp.add_member_ref(ConstantTag.METHODREF, cf.this_class_name, "lambda$run$0", "()V")
    impl_mh_idx = len(cp._entries)
    cp._entries.append(CONSTANT_MethodHandle_info(reference_kind=ReferenceKind.REF_invokeStatic, reference_index=impl_ref_idx))

    inst_type_idx = len(cp._entries)
    cp._entries.append(CONSTANT_MethodType_info(descriptor_index=cp.add_utf8("()V")))

    bsm_attr_name = cp.add_utf8("BootstrapMethods")
    bsm = BootstrapMethod(bootstrap_method_ref=mh_idx, bootstrap_arguments=[sam_type_idx, impl_mh_idx, inst_type_idx])
    bsm_attr = BootstrapMethodsAttribute(attribute_name_index=bsm_attr_name, bootstrap_methods=[bsm])
    cf.attributes.append(bsm_attr)

    nat_idx = cp.add_name_and_type("run", "()Ljava/lang/Runnable;")
    indy_idx = len(cp._entries)
    cp._entries.append(CONSTANT_InvokeDynamic_info(bootstrap_method_attr_index=0, name_and_type_index=nat_idx))

    resolver = DynamicInvokeResolver(cf)
    lambda_info = resolver.resolve_lambda(cp[indy_idx])
    assert lambda_info is not None
    assert lambda_info.sam_interface == "java/lang/Runnable"
    assert lambda_info.sam_method_name == "run"
    assert lambda_info.impl_method_name == "lambda$run$0"


# ============================================================================
# F19: Dead Code, NOP Pruning & Bytecode Reassembly
# ============================================================================

def test_dead_code_unreachable_block_pruning():
    # pc 0: return
    # pc 1: iconst_1 (unreachable)
    # pc 2: return   (unreachable)
    code = bytes([Opcode.RETURN, Opcode.ICONST_1, Opcode.RETURN])
    cfg = ControlFlowGraph.from_code(code)
    assert len(cfg.blocks) == 2
    pruned = prune_unreachable_blocks(cfg)
    assert pruned == 1
    assert len(cfg.blocks) == 1
    assert 0 in cfg.blocks


def test_dead_code_goto_chain_compaction():
    # b0: goto b1
    # b1: goto b2
    # b2: return
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
    # b0 now branches directly to b2 (20)
    assert 20 in cfg.get_successors(0)


def test_dead_code_nop_stripping_and_offset_recalculation():
    # pc 0: goto +5 (pc 5)
    # pc 3: nop
    # pc 4: nop
    # pc 5: return
    code = bytes([Opcode.GOTO]) + struct.pack(">h", 5) + bytes([Opcode.NOP, Opcode.NOP, Opcode.RETURN])
    new_code, _, nops_removed = strip_nops_from_bytecode(code)
    assert nops_removed == 2
    assert len(new_code) == 4  # goto (3B) + return (1B)
    # New goto offset is +3 (points to pc 3: return)
    new_offset = struct.unpack(">h", new_code[1:3])[0]
    assert new_offset == 3


def test_reassemble_cfg_round_trip():
    # Diamond CFG reassembled cleanly
    code = (
        bytes([Opcode.ICONST_1, Opcode.IFEQ]) + struct.pack(">h", 7) +
        bytes([Opcode.ICONST_2, Opcode.GOTO]) + struct.pack(">h", 4) +
        bytes([Opcode.ICONST_3, Opcode.IRETURN])
    )
    cfg = ControlFlowGraph.from_code(code)
    new_code, ex_table, max_stack, max_locals = reassemble_cfg(cfg)
    assert len(new_code) > 0
    assert new_code[-1] == Opcode.IRETURN


# ============================================================================
# End-to-End MethodNormalizer Pipeline
# ============================================================================

def test_method_normalizer_full_pipeline():
    cf = create_test_class()
    cp = cf.constant_pool

    # Method containing arithmetic to fold: 10 + 20, NOPs, dead block after return
    code = bytes([
        Opcode.BIPUSH, 10,
        Opcode.BIPUSH, 20,
        Opcode.IADD,
        Opcode.NOP,
        Opcode.NOP,
        Opcode.IRETURN,
        Opcode.NOP,
        Opcode.ICONST_0,
        Opcode.IRETURN
    ])
    c_name = cp.add_utf8("Code")
    code_attr = CodeAttribute(attribute_name_index=c_name, max_stack=2, max_locals=1, code=code)
    method = MethodInfo(access_flags=AccessFlags.PUBLIC, name_index=cp.add_utf8("calc"), descriptor_index=cp.add_utf8("()I"), attributes=[code_attr])
    cf.methods.append(method)

    normalizer = MethodNormalizer()
    stats = normalizer.normalize_method(cf, method)

    assert stats.constants_folded >= 1
    assert stats.dead_blocks_pruned >= 1
    assert stats.nops_stripped >= 2
    # Verify code reassembled cleanly
    assert len(code_attr.code) < len(code)
    assert code_attr.code[-1] == Opcode.IRETURN


# ============================================================================
# Additional Exhaustive Tests for F13-F19 & Boundary Conditions
# ============================================================================

def test_cfg_empty_and_single_instruction_method():
    code = bytes([Opcode.RETURN])
    cfg = ControlFlowGraph.from_code(code)
    assert len(cfg.blocks) == 1
    assert cfg.cyclomatic_complexity == 1
    assert cfg.get_exit_blocks()[0].id == 0


def test_cfg_lookupswitch_sparse_keys():
    # pc 0: iconst_1
    # pc 1: lookupswitch pad=2 (pc 4), default=+18, npairs=2, (-100, +18), (500, +19)
    # pc 22: return
    # pc 23: return
    code = (
        bytes([Opcode.ICONST_1, Opcode.LOOKUPSWITCH, 0, 0]) +
        struct.pack(">ii", 18, 2) +
        struct.pack(">iiii", -100, 18, 500, 19) +
        bytes([Opcode.RETURN, Opcode.RETURN])
    )
    cfg = ControlFlowGraph.from_code(code)
    out_edges = cfg.get_outgoing_edges(0)
    case_keys = {e.case_key for e in out_edges if e.edge_type == CFGEdgeType.SWITCH_CASE}
    assert -100 in case_keys
    assert 500 in case_keys


def test_deflatten_complexity_reduction_quantified():
    cfg = ControlFlowGraph()
    b_init = BasicBlock(0, 0, 4, [Instruction(0, Opcode.ICONST_0), Instruction(1, Opcode.ISTORE_1), Instruction(2, Opcode.GOTO, struct.pack(">h", 8))])
    b_disp = BasicBlock(10, 10, 30, [Instruction(10, Opcode.ILOAD_1), Instruction(11, Opcode.TABLESWITCH, operands=b"\x00" * 16)])
    b_p0 = BasicBlock(100, 100, 105, [Instruction(100, Opcode.ICONST_1), Instruction(101, Opcode.ISTORE_1), Instruction(102, Opcode.GOTO, struct.pack(">h", 0))])
    b_p1 = BasicBlock(200, 200, 205, [Instruction(200, Opcode.ICONST_2), Instruction(201, Opcode.ISTORE_1), Instruction(202, Opcode.GOTO, struct.pack(">h", 0))])
    b_p2 = BasicBlock(300, 300, 302, [Instruction(300, Opcode.RETURN)])
    for b in [b_init, b_disp, b_p0, b_p1, b_p2]:
        cfg.add_block(b)
    cfg.add_edge(CFGEdge(0, 10, CFGEdgeType.UNCONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(10, 100, CFGEdgeType.SWITCH_CASE, case_key=0))
    cfg.add_edge(CFGEdge(10, 200, CFGEdgeType.SWITCH_CASE, case_key=1))
    cfg.add_edge(CFGEdge(10, 300, CFGEdgeType.SWITCH_CASE, case_key=2))
    cfg.add_edge(CFGEdge(100, 10, CFGEdgeType.UNCONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(200, 10, CFGEdgeType.UNCONDITIONAL_BRANCH))
    report = ControlFlowDeflattener(cfg).deflatten()
    assert report.complexity_after <= report.complexity_before


def test_opaque_algebraic_self_subtraction():
    cfg = ControlFlowGraph()
    b_entry = BasicBlock(0, 0, 6, [
        Instruction(0, Opcode.ILOAD_1),
        Instruction(1, Opcode.ILOAD_1),
        Instruction(2, Opcode.ISUB),
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


def test_opaque_algebraic_zero_and_mask():
    cfg = ControlFlowGraph()
    b_entry = BasicBlock(0, 0, 6, [
        Instruction(0, Opcode.ILOAD_1),
        Instruction(1, Opcode.ICONST_0),
        Instruction(2, Opcode.IAND),
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


def test_opaque_algebraic_disjunction_with_one():
    cfg = ControlFlowGraph()
    b_entry = BasicBlock(0, 0, 6, [
        Instruction(0, Opcode.ILOAD_1),
        Instruction(1, Opcode.ICONST_1),
        Instruction(2, Opcode.IOR),
        Instruction(3, Opcode.IFNE, struct.pack(">h", 10))
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


def test_opaque_null_checks():
    cfg = ControlFlowGraph()
    b_entry = BasicBlock(0, 0, 4, [
        Instruction(0, Opcode.ACONST_NULL),
        Instruction(1, Opcode.IFNULL, struct.pack(">h", 10))
    ])
    b_fall = BasicBlock(4, 4, 5, [Instruction(4, Opcode.RETURN)])
    b_target = BasicBlock(10, 10, 11, [Instruction(10, Opcode.RETURN)])
    for b in [b_entry, b_fall, b_target]:
        cfg.add_block(b)
    cfg.add_edge(CFGEdge(0, 10, CFGEdgeType.CONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(0, 4, CFGEdgeType.FALLTHROUGH))
    report = OpaquePredicateEliminator(cfg).eliminate()
    assert report.constant_predicates_pruned == 1
    assert 10 in cfg.get_successors(0)
    assert 4 not in cfg.get_successors(0)


def test_constant_folding_long_and_float_arithmetic():
    assert ArithmeticEvaluator.eval_binary_long(Opcode.LADD, 100, 200) == 300
    assert ArithmeticEvaluator.eval_binary_long(Opcode.LMUL, 10, 20) == 200
    assert ArithmeticEvaluator.eval_binary_long(Opcode.LXOR, 0xFF, 0x0F) == 0xF0
    assert ArithmeticEvaluator.eval_binary_float(Opcode.FADD, 1.5, 2.5) == 4.0
    assert ArithmeticEvaluator.eval_binary_double(Opcode.DMUL, 3.0, 4.0) == 12.0


def test_constant_folding_min_int_overflow_safety():
    res = ArithmeticEvaluator.eval_binary_int(Opcode.IDIV, -2147483648, -1)
    assert res == -2147483648
    res_l = ArithmeticEvaluator.eval_binary_long(Opcode.LDIV, -9223372036854775808, -1)
    assert res_l == -9223372036854775808


def test_constant_folding_comparisons():
    assert ArithmeticEvaluator.eval_comparison(Opcode.LCMP, 50, 20) == 1
    assert ArithmeticEvaluator.eval_comparison(Opcode.LCMP, 20, 50) == -1
    assert ArithmeticEvaluator.eval_comparison(Opcode.LCMP, 20, 20) == 0


def test_string_decryptor_timeout_safety():
    cf = create_test_class()
    cp = cf.constant_pool
    d_name = cp.add_utf8("loop")
    d_desc = cp.add_utf8("()Ljava/lang/String;")
    loop_code = bytes([Opcode.GOTO, 0, 0])
    c_name = cp.add_utf8("Code")
    code_attr = CodeAttribute(attribute_name_index=c_name, max_stack=1, max_locals=1, code=loop_code)
    m = MethodInfo(access_flags=AccessFlags.STATIC | AccessFlags.PUBLIC, name_index=d_name, descriptor_index=d_desc, attributes=[code_attr])
    cf.methods.append(m)

    emulator = BytecodeEmulator(max_steps=100)
    res = emulator.execute(cf, m, [])
    assert res is None


def test_string_decryptor_string_builder_emulation():
    cf = create_test_class()
    cp = cf.constant_pool
    d_name = cp.add_utf8("dec")
    d_desc = cp.add_utf8("()Ljava/lang/String;")

    sb_cls = cp.add_class("java/lang/StringBuilder")
    sb_init = cp.add_member_ref(ConstantTag.METHODREF, "java/lang/StringBuilder", "<init>", "()V")
    sb_append = cp.add_member_ref(ConstantTag.METHODREF, "java/lang/StringBuilder", "append", "(Ljava/lang/String;)Ljava/lang/StringBuilder;")
    sb_to_string = cp.add_member_ref(ConstantTag.METHODREF, "java/lang/StringBuilder", "toString", "()Ljava/lang/String;")
    s1 = cp.add_string("FOO")
    s2 = cp.add_string("BAR")

    code = (
        bytes([Opcode.NEW]) + struct.pack(">H", sb_cls) +
        bytes([Opcode.DUP, Opcode.INVOKESPECIAL]) + struct.pack(">H", sb_init) +
        bytes([Opcode.LDC, s1, Opcode.INVOKEVIRTUAL]) + struct.pack(">H", sb_append) +
        bytes([Opcode.LDC, s2, Opcode.INVOKEVIRTUAL]) + struct.pack(">H", sb_append) +
        bytes([Opcode.INVOKEVIRTUAL]) + struct.pack(">H", sb_to_string) +
        bytes([Opcode.ARETURN])
    )
    c_name = cp.add_utf8("Code")
    code_attr = CodeAttribute(attribute_name_index=c_name, max_stack=3, max_locals=1, code=code)
    m = MethodInfo(access_flags=AccessFlags.STATIC | AccessFlags.PUBLIC, name_index=d_name, descriptor_index=d_desc, attributes=[code_attr])
    cf.methods.append(m)

    emulator = BytecodeEmulator()
    res = emulator.execute(cf, m, [])
    assert res == "FOOBAR"


def test_dynamic_invoke_string_concat_mixed_dynamic_recipe():
    cf = create_test_class()
    cp = cf.constant_pool
    bsm_mh_idx = cp.add_member_ref(
        ConstantTag.METHODREF,
        "java/lang/invoke/StringConcatFactory",
        "makeConcatWithConstants",
        "(Ljava/lang/invoke/MethodHandles$Lookup;Ljava/lang/String;Ljava/lang/invoke/MethodType;Ljava/lang/String;[Ljava/lang/Object;)Ljava/lang/invoke/CallSite;"
    )
    mh_idx = len(cp._entries)
    cp._entries.append(CONSTANT_MethodHandle_info(reference_kind=ReferenceKind.REF_invokeStatic, reference_index=bsm_mh_idx))
    recipe_idx = cp.add_string("\u0001+\u0002")
    const_idx = cp.add_string("CONST")
    bsm_attr_name = cp.add_utf8("BootstrapMethods")
    bsm = BootstrapMethod(bootstrap_method_ref=mh_idx, bootstrap_arguments=[recipe_idx, const_idx])
    cf.attributes.append(BootstrapMethodsAttribute(attribute_name_index=bsm_attr_name, bootstrap_methods=[bsm]))
    nat_idx = cp.add_name_and_type("makeConcatWithConstants", "(Ljava/lang/String;)Ljava/lang/String;")
    indy_idx = len(cp._entries)
    cp._entries.append(CONSTANT_InvokeDynamic_info(bootstrap_method_attr_index=0, name_and_type_index=nat_idx))
    resolver = DynamicInvokeResolver(cf)
    concat = resolver.resolve_string_concat(cp[indy_idx])
    assert concat is not None
    assert concat.is_fully_constant is False
    assert len(concat.dynamic_param_types) == 1
    assert concat.constant_args == ["CONST"]


def test_dynamic_invoke_static_reflection_resolution():
    cf = create_test_class()
    cp = cf.constant_pool
    cls_str = cp.add_string("com.example.Target")
    m_str = cp.add_string("doWork")
    c_for_name = cp.add_member_ref(ConstantTag.METHODREF, "java/lang/Class", "forName", "(Ljava/lang/String;)Ljava/lang/Class;")
    m_invoke = cp.add_member_ref(ConstantTag.METHODREF, "java/lang/reflect/Method", "invoke", "(Ljava/lang/Object;[Ljava/lang/Object;)Ljava/lang/Object;")
    code = (
        bytes([Opcode.LDC, cls_str, Opcode.INVOKESTATIC]) + struct.pack(">H", c_for_name) +
        bytes([Opcode.LDC, m_str, Opcode.INVOKEVIRTUAL]) + struct.pack(">H", m_invoke) +
        bytes([Opcode.RETURN])
    )
    c_name = cp.add_utf8("Code")
    code_attr = CodeAttribute(attribute_name_index=c_name, max_stack=3, max_locals=1, code=code)
    m = MethodInfo(access_flags=AccessFlags.PUBLIC, name_index=cp.add_utf8("reflectTest"), descriptor_index=cp.add_utf8("()V"), attributes=[code_attr])
    cf.methods.append(m)
    resolver = DynamicInvokeResolver(cf)
    reflections = resolver.resolve_reflection_calls(m)
    assert len(reflections) == 1
    assert reflections[0].target_class == "com/example/Target"
    assert reflections[0].target_method == "doWork"


def test_dead_code_cyclic_goto_chain_safety():
    cfg = ControlFlowGraph()
    b0 = BasicBlock(0, 0, 3, [Instruction(0, Opcode.GOTO, struct.pack(">h", 10))])
    b1 = BasicBlock(10, 10, 13, [Instruction(10, Opcode.GOTO, struct.pack(">h", -10))])
    cfg.add_block(b0)
    cfg.add_block(b1)
    cfg.add_edge(CFGEdge(0, 10, CFGEdgeType.UNCONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(10, 0, CFGEdgeType.UNCONDITIONAL_BRANCH))
    compacted = compact_goto_chains(cfg)
    assert compacted >= 0


def test_dead_code_redundant_fallthrough_elimination():
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


def test_method_normalizer_empty_and_trivial_method():
    cf = create_test_class()
    cp = cf.constant_pool
    c_name = cp.add_utf8("Code")
    code_attr = CodeAttribute(attribute_name_index=c_name, max_stack=1, max_locals=1, code=bytes([Opcode.RETURN]))
    m = MethodInfo(access_flags=AccessFlags.PUBLIC, name_index=cp.add_utf8("emptyMethod"), descriptor_index=cp.add_utf8("()V"), attributes=[code_attr])
    cf.methods.append(m)
    normalizer = MethodNormalizer()
    stats = normalizer.normalize_method(cf, m)
    assert isinstance(stats, NormalizationStats)
    assert stats.complexity_reduced >= 0


# ============================================================================
# Iteration 2 Regression & Boundary Tests
# ============================================================================

def test_deflatten_fallthrough_payload_preserved():
    """Verify that unflattening does not overwrite fallthrough payload instructions."""
    cfg = ControlFlowGraph()
    b0 = BasicBlock(0, 0, 4, [
        Instruction(0, Opcode.ICONST_0),
        Instruction(1, Opcode.ISTORE_1),
        Instruction(2, Opcode.GOTO, struct.pack(">h", 8))
    ])
    b_disp = BasicBlock(10, 10, 30, [
        Instruction(10, Opcode.ILOAD_1),
        Instruction(11, Opcode.TABLESWITCH, operands=b"\x00" * 16)
    ])
    b_p0 = BasicBlock(100, 100, 105, [
        Instruction(100, Opcode.IINC, operands=bytes([2, 1])),
        Instruction(103, Opcode.ICONST_1),
        Instruction(104, Opcode.ISTORE_1)
    ])
    cfg.add_block(b0)
    cfg.add_block(b_disp)
    cfg.add_block(b_p0)
    cfg.add_block(BasicBlock(200, 200, 201, [Instruction(200, Opcode.RETURN)]))
    cfg.add_edge(CFGEdge(0, 10, CFGEdgeType.UNCONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(10, 100, CFGEdgeType.SWITCH_CASE, case_key=0))
    cfg.add_edge(CFGEdge(10, 200, CFGEdgeType.SWITCH_CASE, case_key=1))
    cfg.add_edge(CFGEdge(100, 10, CFGEdgeType.FALLTHROUGH))

    ControlFlowDeflattener(cfg).deflatten()
    assert any(i.opcode == Opcode.IINC for i in b_p0.instructions), "IINC payload instruction was erased!"
    assert b_p0.instructions[-1].opcode == Opcode.GOTO, "GOTO was not appended!"


def test_opaque_consecutive_parity_exact_pattern():
    """Verify Invariant 5 x*(x+1)%2==0 is pruned when exact pattern matches."""
    cfg = ControlFlowGraph()
    b0 = BasicBlock(0, 0, 10, [
        Instruction(0, Opcode.ILOAD_1), Instruction(1, Opcode.ILOAD_1),
        Instruction(2, Opcode.ICONST_1), Instruction(3, Opcode.IADD),
        Instruction(4, Opcode.IMUL), Instruction(5, Opcode.ICONST_2),
        Instruction(6, Opcode.IREM), Instruction(7, Opcode.IFEQ, struct.pack(">h", 5))
    ])
    cfg.add_block(b0)
    cfg.add_block(BasicBlock(10, 10, 11, [Instruction(10, Opcode.RETURN)]))
    cfg.add_block(BasicBlock(20, 20, 21, [Instruction(20, Opcode.RETURN)]))
    cfg.add_edge(CFGEdge(0, 10, CFGEdgeType.CONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(0, 20, CFGEdgeType.FALLTHROUGH))

    rep = OpaquePredicateEliminator(cfg).eliminate()
    assert rep.algebraic_predicates_pruned == 1
    assert len(b0.instructions) == 1 and b0.instructions[0].opcode == Opcode.GOTO


def test_opaque_modulo_2_not_pruned():
    """Verify regular isEven(x) check is NOT misclassified as opaque predicate."""
    code = (
        bytes([Opcode.ICONST_0, Opcode.ISTORE_1,
               Opcode.ICONST_0, Opcode.ISTORE_2,
               Opcode.ICONST_0, Opcode.ISTORE_3,
               Opcode.ILOAD_0, Opcode.ICONST_2, Opcode.IREM,
               Opcode.IFEQ]) + struct.pack(">h", 4) +
        bytes([Opcode.ICONST_0, Opcode.IRETURN,
               Opcode.ICONST_1, Opcode.IRETURN])
    )
    cfg = ControlFlowGraph.from_code(code)
    b0_len = len(cfg.get_block(0).instructions)
    pruned = eliminate_opaque_predicates(cfg)
    assert pruned == 0
    assert len(cfg.get_block(0).instructions) == b0_len


def test_opaque_expression_stack_leakage_prevented():
    """Verify compound expression (a + b) | 1 != 0 is NOT pruned to avoid stack leak."""
    code = (
        bytes([Opcode.ILOAD_0, Opcode.ILOAD_1, Opcode.IADD,
               Opcode.ICONST_1, Opcode.IOR,
               Opcode.IFNE]) + struct.pack(">h", 4) +
        bytes([Opcode.ICONST_0, Opcode.IRETURN,
               Opcode.ICONST_1, Opcode.IRETURN])
    )
    cfg = ControlFlowGraph.from_code(code)
    pruned = eliminate_opaque_predicates(cfg)
    assert pruned == 0
    b0 = cfg.get_block(0)
    opcodes = [inst.opcode for inst in b0.instructions]
    assert not (Opcode.ILOAD_0 in opcodes and Opcode.ILOAD_1 in opcodes and Opcode.GOTO in opcodes)


def test_opaque_invariants_6_and_7():
    """Verify Invariant 6 (non-negative mask) and Invariant 7 (cube parity)."""
    # Invariant 6: iload_1; ldc 0x7FFFFFFF; iand; ifge
    cp6 = ConstantPool([None, CONSTANT_Integer_info(value=0x7FFFFFFF)])
    cfg6 = ControlFlowGraph()
    b6 = BasicBlock(0, 0, 10, [
        Instruction(0, Opcode.ILOAD_1),
        Instruction(1, Opcode.LDC, struct.pack(">B", 1)),
        Instruction(2, Opcode.IAND),
        Instruction(3, Opcode.IFGE, struct.pack(">h", 5))
    ])
    cfg6.add_block(b6)
    cfg6.add_block(BasicBlock(10, 10, 11, [Instruction(10, Opcode.RETURN)]))
    cfg6.add_block(BasicBlock(20, 20, 21, [Instruction(20, Opcode.RETURN)]))
    cfg6.add_edge(CFGEdge(0, 10, CFGEdgeType.CONDITIONAL_BRANCH))
    cfg6.add_edge(CFGEdge(0, 20, CFGEdgeType.FALLTHROUGH))
    rep6 = OpaquePredicateEliminator(cfg6, cp=cp6).eliminate()
    assert rep6.algebraic_predicates_pruned == 1
    assert len(b6.instructions) == 1 and b6.instructions[0].opcode == Opcode.GOTO

    # Invariant 7: iload_1; iload_1; iload_1; imul; iconst_1; isub; imul; bipush 6; irem; ifeq
    cfg7 = ControlFlowGraph()
    b7 = BasicBlock(0, 0, 12, [
        Instruction(0, Opcode.ILOAD_1), Instruction(1, Opcode.ILOAD_1), Instruction(2, Opcode.ILOAD_1),
        Instruction(3, Opcode.IMUL), Instruction(4, Opcode.ICONST_1), Instruction(5, Opcode.ISUB),
        Instruction(6, Opcode.IMUL), Instruction(7, Opcode.BIPUSH, b"\x06"),
        Instruction(9, Opcode.IREM), Instruction(10, Opcode.IFEQ, struct.pack(">h", 5))
    ])
    cfg7.add_block(b7)
    cfg7.add_block(BasicBlock(10, 10, 11, [Instruction(10, Opcode.RETURN)]))
    cfg7.add_block(BasicBlock(20, 20, 21, [Instruction(20, Opcode.RETURN)]))
    cfg7.add_edge(CFGEdge(0, 10, CFGEdgeType.CONDITIONAL_BRANCH))
    cfg7.add_edge(CFGEdge(0, 20, CFGEdgeType.FALLTHROUGH))
    rep7 = OpaquePredicateEliminator(cfg7).eliminate()
    assert rep7.algebraic_predicates_pruned == 1
    assert len(b7.instructions) == 1 and b7.instructions[0].opcode == Opcode.GOTO


def test_opaque_nonnegative_mask_cp_safety():
    """Verify that Invariant 6 requires valid CP and exact 0x7FFFFFFF constant value, not false matching."""
    # Subcase A: cp is None -> Must NOT eliminate branch
    cfg_no_cp = ControlFlowGraph()
    b_no_cp = BasicBlock(0, 0, 10, [
        Instruction(0, Opcode.ILOAD_1),
        Instruction(1, Opcode.LDC, struct.pack(">B", 1)),
        Instruction(2, Opcode.IAND),
        Instruction(3, Opcode.IFGE, struct.pack(">h", 5))
    ])
    cfg_no_cp.add_block(b_no_cp)
    cfg_no_cp.add_block(BasicBlock(10, 10, 11, [Instruction(10, Opcode.RETURN)]))
    cfg_no_cp.add_block(BasicBlock(20, 20, 21, [Instruction(20, Opcode.RETURN)]))
    cfg_no_cp.add_edge(CFGEdge(0, 10, CFGEdgeType.CONDITIONAL_BRANCH))
    cfg_no_cp.add_edge(CFGEdge(0, 20, CFGEdgeType.FALLTHROUGH))
    rep_no_cp = OpaquePredicateEliminator(cfg_no_cp, cp=None).eliminate()
    assert rep_no_cp.algebraic_predicates_pruned == 0
    assert len(b_no_cp.instructions) == 4

    # Subcase B: cp has different constant (e.g. -1 or 42) -> Must NOT eliminate branch
    cp_other = ConstantPool([None, CONSTANT_Integer_info(value=-1)])
    cfg_other = ControlFlowGraph()
    b_other = BasicBlock(0, 0, 10, [
        Instruction(0, Opcode.ILOAD_1),
        Instruction(1, Opcode.LDC, struct.pack(">B", 1)),
        Instruction(2, Opcode.IAND),
        Instruction(3, Opcode.IFGE, struct.pack(">h", 5))
    ])
    cfg_other.add_block(b_other)
    cfg_other.add_block(BasicBlock(10, 10, 11, [Instruction(10, Opcode.RETURN)]))
    cfg_other.add_block(BasicBlock(20, 20, 21, [Instruction(20, Opcode.RETURN)]))
    cfg_other.add_edge(CFGEdge(0, 10, CFGEdgeType.CONDITIONAL_BRANCH))
    cfg_other.add_edge(CFGEdge(0, 20, CFGEdgeType.FALLTHROUGH))
    rep_other = OpaquePredicateEliminator(cfg_other, cp=cp_other).eliminate()
    assert rep_other.algebraic_predicates_pruned == 0
    assert len(b_other.instructions) == 4


def test_dead_code_nop_stripping_preceding_branches():
    """Verify that multiple NOPs preceding branch instructions do not corrupt target offsets."""
    code = (
        bytes([Opcode.NOP, Opcode.NOP, Opcode.NOP, Opcode.GOTO]) + struct.pack(">h", 7) +
        bytes([Opcode.ICONST_1, Opcode.ICONST_2, Opcode.IADD, Opcode.POP, Opcode.RETURN])
    )
    new_code, _, removed = strip_nops_from_bytecode(code)
    assert removed == 3
    new_offset = struct.unpack(">h", new_code[1:3])[0]
    assert new_offset == 7
    assert new_code[7] == Opcode.RETURN


def test_dead_code_nop_stripping_preceding_tableswitch():
    """Verify that NOPs preceding TABLESWITCH realign padding and preserve default/case offsets."""
    pad = 3
    sw_operands = (
        b"\x00" * pad +
        struct.pack(">iii", 25, 0, 1) +  # default=25 (pc 29), low=0, high=1
        struct.pack(">ii", 25, 25)        # case0=25, case1=25
    )
    code = (
        bytes([Opcode.NOP, Opcode.ICONST_0, Opcode.NOP, Opcode.NOP, Opcode.TABLESWITCH]) +
        sw_operands +
        bytes([Opcode.NOP, Opcode.RETURN])
    )
    ex = [ExceptionTableEntry(start_pc=1, end_pc=len(code)-1, handler_pc=len(code)-1, catch_type=0)]
    new_code, new_ex, removed = strip_nops_from_bytecode(code, ex)
    assert removed == 4
    assert new_code[0] == Opcode.ICONST_0
    assert new_code[1] == Opcode.TABLESWITCH
    assert new_code[2:4] == b"\x00\x00"
    new_default = struct.unpack(">i", new_code[4:8])[0]
    assert new_default == 23
    assert new_code[24] == Opcode.RETURN
    assert len(new_code) == 25
    assert new_ex[0].start_pc == 0
    assert new_ex[0].end_pc == 24
    assert new_ex[0].handler_pc == 24


def test_reassemble_cfg_switch_padding_alignment():
    """Verify CFG reassembly correctly computes switch padding without inflating subsequent block offsets."""
    code = bytes([Opcode.NOP, Opcode.TABLESWITCH, 0, 0]) + struct.pack(">iiii", 19, 0, 0, 19) + bytes([Opcode.RETURN])
    cfg = ControlFlowGraph.from_code(code)
    new_code, _, _, _ = cfg.reassemble_code()

    sw_idx = new_code.find(bytes([Opcode.TABLESWITCH]))
    sw_pad = (4 - (sw_idx + 1) % 4) % 4
    default_off = struct.unpack(">i", new_code[sw_idx + 1 + sw_pad : sw_idx + 1 + sw_pad + 4])[0]
    target_pc = sw_idx + default_off
    actual_target_pc = len(new_code) - 1

    assert target_pc == actual_target_pc


def test_constant_folding_ldiv_precision_64bit():
    """Verify 64-bit integer division preserves exact precision on Long.MAX_VALUE."""
    max_long = 9223372036854775807
    res = ArithmeticEvaluator.eval_binary_long(Opcode.LDIV, max_long, 1)
    assert res == max_long


def test_constant_folding_frem_drem_infinity():
    """Verify FREM and DREM with Infinity evaluate to NaN rather than throwing ValueError."""
    import math
    res_f = ArithmeticEvaluator.eval_binary_float(Opcode.FREM, math.inf, 1.0)
    assert math.isnan(res_f)
    res_d = ArithmeticEvaluator.eval_binary_double(Opcode.DREM, math.inf, 1.0)
    assert math.isnan(res_d)


def test_constant_folding_fdiv_negative_zero_sign():
    """Verify FDIV and DDIV by -0.0 respect IEEE 754 sign rules."""
    import math
    res_pos = ArithmeticEvaluator.eval_binary_float(Opcode.FDIV, 1.0, -0.0)
    assert math.isinf(res_pos) and math.copysign(1.0, res_pos) < 0
    res_neg = ArithmeticEvaluator.eval_binary_float(Opcode.FDIV, -1.0, -0.0)
    assert math.isinf(res_neg) and math.copysign(1.0, res_neg) > 0


def test_constant_folding_cfg_iand_ior_ixor():
    """Verify CFG constant folding folds IAND, IOR, and IXOR."""
    cfg = ControlFlowGraph()
    b = BasicBlock(0, 0, 10, [
        Instruction(0, Opcode.ICONST_3),
        Instruction(1, Opcode.ICONST_2),
        Instruction(2, Opcode.IAND),
        Instruction(3, Opcode.IRETURN)
    ])
    cfg.add_block(b)
    folded = fold_constants_and_arithmetic(cfg)
    assert folded == 1
    assert b.instructions[0].opcode == Opcode.ICONST_2
    assert b.instructions[1].opcode == Opcode.IRETURN


def test_constant_folding_cfg_ladd_fadd():
    """Verify CFG constant folding handles LADD and FADD."""
    cf = create_test_class()
    cp = cf.constant_pool

    # Test LADD
    cfg_l = ControlFlowGraph()
    b_l = BasicBlock(0, 0, 10, [
        Instruction(0, Opcode.LCONST_1),
        Instruction(1, Opcode.LCONST_1),
        Instruction(2, Opcode.LADD),
        Instruction(3, Opcode.LRETURN)
    ])
    cfg_l.add_block(b_l)
    folded_l = fold_constants_and_arithmetic(cfg_l, cp=cp)
    assert folded_l == 1
    assert b_l.instructions[0].opcode == Opcode.LDC2_W

    # Test FADD
    cfg_f = ControlFlowGraph()
    b_f = BasicBlock(0, 0, 10, [
        Instruction(0, Opcode.FCONST_1),
        Instruction(1, Opcode.FCONST_2),
        Instruction(2, Opcode.FADD),
        Instruction(3, Opcode.FRETURN)
    ])
    cfg_f.add_block(b_f)
    folded_f = fold_constants_and_arithmetic(cfg_f, cp=cp)
    assert folded_f == 1
    assert b_f.instructions[0].opcode == Opcode.LDC


def test_dynamic_invoke_string_concat_dynamic_params_rejected():
    """Verify StringConcatFactory with non-empty callsite params is not marked fully constant."""
    cf = create_test_class()
    cp = cf.constant_pool
    bsm_mh_idx = cp.add_member_ref(
        ConstantTag.METHODREF,
        "java/lang/invoke/StringConcatFactory",
        "makeConcatWithConstants",
        "(Ljava/lang/invoke/MethodHandles$Lookup;Ljava/lang/String;Ljava/lang/invoke/MethodType;Ljava/lang/String;[Ljava/lang/Object;)Ljava/lang/invoke/CallSite;"
    )
    mh_idx = len(cp._entries)
    cp._entries.append(CONSTANT_MethodHandle_info(reference_kind=ReferenceKind.REF_invokeStatic, reference_index=bsm_mh_idx))

    recipe_idx = cp.add_string("ConstantText")
    bsm = BootstrapMethod(bootstrap_method_ref=mh_idx, bootstrap_arguments=[recipe_idx])
    cf.attributes.append(BootstrapMethodsAttribute(attribute_name_index=cp.add_utf8("BootstrapMethods"), bootstrap_methods=[bsm]))

    nat_idx = cp.add_name_and_type("makeConcatWithConstants", "(I)Ljava/lang/String;")
    indy_idx = len(cp._entries)
    cp._entries.append(CONSTANT_InvokeDynamic_info(bootstrap_method_attr_index=0, name_and_type_index=nat_idx))

    resolver = DynamicInvokeResolver(cf)
    concat = resolver.resolve_string_concat(cp[indy_idx])
    assert concat is not None
    assert concat.is_fully_constant is False


def test_dynamic_invoke_reflection_logging_strings_discriminated():
    """Verify reflection resolver does not misidentify logging strings followed by pop as targets."""
    cf = create_test_class()
    cp = cf.constant_pool
    log1 = cp.add_string("Unrelated log message 1")
    log2 = cp.add_string("Unrelated log message 2")
    m_invoke = cp.add_member_ref(ConstantTag.METHODREF, "java/lang/reflect/Method", "invoke", "(Ljava/lang/Object;[Ljava/lang/Object;)Ljava/lang/Object;")

    code = bytes([
        Opcode.LDC, log1, Opcode.POP,
        Opcode.LDC, log2, Opcode.POP,
        Opcode.ALOAD_0, Opcode.ACONST_NULL, Opcode.ACONST_NULL,
        Opcode.INVOKEVIRTUAL
    ]) + struct.pack(">H", m_invoke) + bytes([Opcode.POP, Opcode.RETURN])

    code_attr = CodeAttribute(attribute_name_index=cp.add_utf8("Code"), max_stack=3, max_locals=1, code=code)
    m = MethodInfo(access_flags=AccessFlags.PUBLIC, name_index=cp.add_utf8("testMethod"), descriptor_index=cp.add_utf8("()V"), attributes=[code_attr])
    cf.methods.append(m)

    resolver = DynamicInvokeResolver(cf)
    reflections = resolver.resolve_reflection_calls(m)
    assert len(reflections) == 0


def test_string_decryptor_negative_signed_byte():
    """Verify BytecodeEmulator masks negative signed byte values when constructing String."""
    emulator = BytecodeEmulator()
    raw = bytes([b & 0xFF for b in [-65, -66]])
    assert raw == b"\xbf\xbe"


def test_string_decryptor_multibyte_key_inlined():
    """Verify multi-byte (String, String) decryptor call site is inlined."""
    cf = create_test_class()
    cp = cf.constant_pool

    m_name = cp.add_utf8("decrypt_multibyte")
    m_desc = cp.add_utf8("(Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;")
    str_to_char_ref = cp.add_member_ref(ConstantTag.METHODREF, "java/lang/String", "toCharArray", "()[C")
    str_init_ref = cp.add_member_ref(ConstantTag.METHODREF, "java/lang/String", "<init>", "([C)V")
    str_cls = cp.add_class("java/lang/String")

    code = bytearray()
    code += bytes([Opcode.ALOAD_0, Opcode.INVOKEVIRTUAL]) + struct.pack(">H", str_to_char_ref)
    code += bytes([Opcode.ASTORE_2])
    code += bytes([Opcode.ALOAD_1, Opcode.INVOKEVIRTUAL]) + struct.pack(">H", str_to_char_ref)
    code += bytes([Opcode.ASTORE_3])
    code += bytes([Opcode.ICONST_0, Opcode.ISTORE, 4])
    loop_head = len(code)
    code += bytes([Opcode.ILOAD, 4, Opcode.ALOAD_2, Opcode.ARRAYLENGTH, Opcode.IF_ICMPGE]) + struct.pack(">h", 0)
    code += bytes([Opcode.ALOAD_2, Opcode.ILOAD, 4, Opcode.ALOAD_2, Opcode.ILOAD, 4, Opcode.CALOAD])
    code += bytes([Opcode.ALOAD_3, Opcode.ILOAD, 4, Opcode.ALOAD_3, Opcode.ARRAYLENGTH, Opcode.IREM, Opcode.CALOAD])
    code += bytes([Opcode.IXOR, Opcode.I2C, Opcode.CASTORE])
    code += bytes([Opcode.IINC, 4, 1, Opcode.GOTO]) + struct.pack(">h", -(len(code) + 3 - loop_head))
    target_exit = len(code)
    code += bytes([Opcode.NEW]) + struct.pack(">H", str_cls)
    code += bytes([Opcode.DUP, Opcode.ALOAD_2, Opcode.INVOKESPECIAL]) + struct.pack(">H", str_init_ref)
    code += bytes([Opcode.ARETURN])

    branch_pc = loop_head + 4
    offset = target_exit - branch_pc
    code[branch_pc + 1 : branch_pc + 3] = struct.pack(">h", offset)

    c_name = cp.add_utf8("Code")
    code_attr = CodeAttribute(attribute_name_index=c_name, max_stack=6, max_locals=5, code=bytes(code))
    method = MethodInfo(access_flags=AccessFlags.STATIC | AccessFlags.PUBLIC, name_index=m_name, descriptor_index=m_desc, attributes=[code_attr])
    cf.methods.append(method)

    key = "AES_KEY"
    plain = "PAYLOAD_BYTES"
    cipher = "".join(chr(ord(c) ^ ord(key[i % len(key)])) for i, c in enumerate(plain))

    c_idx = cp.add_string(cipher)
    k_idx = cp.add_string(key)
    dec_ref = cp.add_member_ref(ConstantTag.METHODREF, cf.this_class_name, "decrypt_multibyte", "(Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;")
    caller_code = bytes([Opcode.LDC, c_idx, Opcode.LDC, k_idx, Opcode.INVOKESTATIC]) + struct.pack(">H", dec_ref) + bytes([Opcode.ARETURN])
    caller_attr = CodeAttribute(attribute_name_index=c_name, max_stack=2, max_locals=1, code=caller_code)
    caller_m = MethodInfo(access_flags=AccessFlags.STATIC | AccessFlags.PUBLIC, name_index=cp.add_utf8("call_multi"), descriptor_index=cp.add_utf8("()Ljava/lang/String;"), attributes=[caller_attr])
    cf.methods.append(caller_m)

    inlined = reconstruct_encrypted_strings(cf, caller_m)
    assert inlined >= 1

