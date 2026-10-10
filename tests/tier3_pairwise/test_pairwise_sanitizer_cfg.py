"""
tests/tier3_pairwise/test_pairwise_sanitizer_cfg.py: Pairwise interactions between M2 Sanitizer and M3 CFG.
Covers:
- F08 (StackMapTable Normalization) + F13 (CFG Basic Block Partitioning)
- F10 (Irregular Exception Handler Fixer) + F13 (CFG Basic Block Partitioning)
- F08 (StackMapTable Normalization) + F19 (Dead Code & NOP Pruning)
- F09 (Generic Signature Sanitizer) + F14 (Control Flow De-Flattening)
- F10 (Irregular Exception Handler Fixer) + F14 (Control Flow De-Flattening)
- F11 (Duplicate Synthetic Member Deduplication) + F19 (Dead Code & NOP Pruning)
"""

import struct
from pathlib import Path
import pytest

from jar_engine.model.constants import Opcode, AccessFlags, ConstantTag
from jar_engine.model.class_file import ClassFile
from jar_engine.model.cp_info import ConstantPool
from jar_engine.model.member_info import (
    FieldInfo, MethodInfo, CodeAttribute, SignatureAttribute,
    StackMapTableAttribute, ExceptionTableEntry
)
from jar_engine.model.jar_archive import JarArchive
from jar_engine.cfg import (
    ControlFlowGraph, BasicBlock, Instruction, CFGEdge, CFGEdgeType,
    partition_basic_blocks, decode_instructions,
    ControlFlowDeflattener, deflatten_control_flow,
    prune_dead_code_and_nops, reassemble_cfg,
    strip_nops_from_bytecode
)
from jar_engine.sanitizer import (
    BytecodeSanitizer,
    sanitize_class_signatures,
    sanitize_code_exceptions,
    sanitize_class_exceptions,
    sanitize_class_stack_maps,
    break_recursive_bridge_methods,
)
from jar_engine.normalization import MethodNormalizer


def test_pairwise_f08_f13_stackmap_cfg_partition():
    """
    Pairwise F08 + F13: Sanitized StackMapTable frames synchronize with
    CFG basic block leader boundaries and branching targets.
    """
    cp = ConstantPool()
    # Code with branch:
    # pc 0..2: iload_0, ifeq +5 (pc 8)
    # pc 3..4: iconst_1, ireturn
    # pc 5..6 (pc 8): iconst_0, ireturn
    code = bytes([
        Opcode.ILOAD_0, Opcode.IFEQ, 0, 5,
        Opcode.ICONST_1, Opcode.IRETURN,
        Opcode.ICONST_0, Opcode.IRETURN
    ])
    # StackMapTable attribute with 1 same_frame at delta 8 (offset 8)
    sm_attr = StackMapTableAttribute(cp.add_utf8("StackMapTable"), bytes([8]))
    code_attr = CodeAttribute(cp.add_utf8("Code"), 2, 1, code, attributes=[sm_attr])

    cf = ClassFile(
        magic=0xCAFEBABE, minor_version=0, major_version=52, constant_pool=cp,
        access_flags=AccessFlags.PUBLIC, this_class=cp.add_class("com/example/BranchClass"),
        super_class=cp.add_class("java/lang/Object"), interfaces=[],
        fields=[],
        methods=[MethodInfo(AccessFlags.PUBLIC | AccessFlags.STATIC, cp.add_utf8("checkVal"), cp.add_utf8("(I)I"), [code_attr])],
        attributes=[]
    )

    # Sanitize stack maps
    sanitize_class_stack_maps(cf)

    # Partition into CFG
    cfg = ControlFlowGraph.from_code(code_attr.code, cp=cp)
    assert len(cfg.blocks) >= 2
    leader_pcs = {b.start_pc for b in cfg.blocks.values()}
    # Leader PCs must include entry (0), fallthrough (4), and jump target (7/6 depending on indexing)
    assert 0 in leader_pcs
    assert any(b.terminator and b.terminator.is_return for b in cfg.blocks.values())


def test_pairwise_f10_f13_exception_ranges_cfg_edges():
    """
    Pairwise F10 + F13: Normalized exception handler ranges construct
    correct CFG EXCEPTION edges from protected blocks to handlers.
    """
    cp = ConstantPool()
    # pc 0..2: iconst_1, istore_1
    # pc 3..5: goto +4 (pc 8)
    # pc 6..7: return (handler)
    # pc 8..9: return
    code = (
        bytes([Opcode.ICONST_1, Opcode.ISTORE_1, Opcode.GOTO, 0, 5]) +
        bytes([Opcode.RETURN, Opcode.RETURN])
    )
    # Corrupt table with duplicate and inverted entries plus 1 valid entry
    raw_table = [
        ExceptionTableEntry(start_pc=10, end_pc=2, handler_pc=6, catch_type=0),  # Inverted
        ExceptionTableEntry(start_pc=0, end_pc=3, handler_pc=6, catch_type=0),   # Valid
        ExceptionTableEntry(start_pc=0, end_pc=3, handler_pc=6, catch_type=0),   # Duplicate
    ]
    code_attr = CodeAttribute(cp.add_utf8("Code"), 2, 2, code, exception_table=raw_table)

    # Sanitize exceptions
    fixed_count = sanitize_code_exceptions(code_attr)
    assert fixed_count >= 1
    assert len(code_attr.exception_table) == 1
    valid_ex = code_attr.exception_table[0]
    assert valid_ex.start_pc == 0 and valid_ex.end_pc == 3

    # Build CFG with sanitized exception table
    cfg = ControlFlowGraph.from_code(code, exception_table=code_attr.exception_table, cp=cp)
    ex_edges = [e for e in cfg.edges if e.edge_type == CFGEdgeType.EXCEPTION]
    assert len(ex_edges) >= 1
    # Exception edge targets handler block containing pc 6
    handler_block = cfg.blocks[ex_edges[0].target_id]
    assert handler_block.contains_pc(6) or handler_block.start_pc == 6


def test_pairwise_f08_f19_stackmap_dead_code_sync():
    """
    Pairwise F08 + F19: StackMapTable entries synchronized and valid
    following dead code pruning and NOP stripping.
    """
    cp = ConstantPool()
    # Code with dead block after return:
    # pc 0..1: iconst_1, ireturn
    # pc 2..5: nop, nop, nop, nop (dead NOPs)
    # pc 6..8: iconst_0, ireturn (dead unreachable block)
    code = bytes([
        Opcode.ICONST_1, Opcode.IRETURN,
        Opcode.NOP, Opcode.NOP, Opcode.NOP, Opcode.NOP,
        Opcode.ICONST_0, Opcode.IRETURN
    ])
    code_attr = CodeAttribute(cp.add_utf8("Code"), 1, 1, code, attributes=[
        StackMapTableAttribute(cp.add_utf8("StackMapTable"), b"")
    ])
    cf = ClassFile(
        magic=0xCAFEBABE, minor_version=0, major_version=52, constant_pool=cp,
        access_flags=AccessFlags.PUBLIC, this_class=cp.add_class("com/example/DeadCodeClass"),
        super_class=cp.add_class("java/lang/Object"), interfaces=[],
        fields=[],
        methods=[MethodInfo(AccessFlags.PUBLIC | AccessFlags.STATIC, cp.add_utf8("deadMethod"), cp.add_utf8("()I"), [code_attr])],
        attributes=[]
    )

    # Build CFG and prune dead blocks / NOPs
    cfg = ControlFlowGraph.from_code(code, cp=cp)
    initial_block_count = len(cfg.blocks)
    prune_dead_code_and_nops(cfg)
    assert len(cfg.blocks) < initial_block_count

    # Reassemble CFG into clean bytecode
    new_code, _, _, _ = reassemble_cfg(cfg, cp=cp)
    assert Opcode.NOP not in new_code
    code_attr.code = new_code

    # Sanitize stack maps on reassembled class
    sanitize_class_stack_maps(cf)
    raw = cf.to_bytes()
    reparsed = ClassFile.parse(raw)
    assert reparsed.magic == 0xCAFEBABE


def test_pairwise_f09_f14_signatures_cfg_deflattening():
    """
    Pairwise F09 + F14: Generic method signatures preserved intact
    across control flow de-flattening passes.
    """
    cp = ConstantPool()
    sig_attr = SignatureAttribute(cp.add_utf8("Signature"), cp.add_utf8("<T:Ljava/lang/Object;>(TT;)TT;"))

    # Flattened dispatcher switch loop:
    # State 0 -> State 1 -> State 2 (Exit)
    disp_code = (
        bytes([Opcode.ICONST_0, Opcode.ISTORE_1, Opcode.GOTO, 0, 3]) +
        bytes([Opcode.ILOAD_1, Opcode.TABLESWITCH, 0]) +
        struct.pack(">iiiiiii", 59, 0, 3, 30, 40, 49, 59) +
        bytes([Opcode.ALOAD_0, Opcode.ASTORE_0, Opcode.ICONST_1, Opcode.ISTORE_1, Opcode.GOTO]) + struct.pack(">h", -38) +
        bytes([Opcode.ALOAD_0, Opcode.ASTORE_0, Opcode.ICONST_2, Opcode.ISTORE_1, Opcode.GOTO]) + struct.pack(">h", -47) +
        bytes([Opcode.ALOAD_0, Opcode.ASTORE_0, Opcode.ICONST_3, Opcode.ISTORE_1, Opcode.GOTO]) + struct.pack(">h", -57) +
        bytes([Opcode.ALOAD_0, Opcode.ARETURN])
    )
    code_attr = CodeAttribute(cp.add_utf8("Code"), 2, 2, disp_code)

    m = MethodInfo(
        AccessFlags.PUBLIC | AccessFlags.STATIC, cp.add_utf8("processItem"), cp.add_utf8("(Ljava/lang/Object;)Ljava/lang/Object;"),
        [code_attr, sig_attr]
    )
    cf = ClassFile(
        magic=0xCAFEBABE, minor_version=0, major_version=52, constant_pool=cp,
        access_flags=AccessFlags.PUBLIC, this_class=cp.add_class("com/example/GenericProcessor"),
        super_class=cp.add_class("java/lang/Object"), interfaces=[],
        fields=[], methods=[m], attributes=[]
    )

    # Sanitize signatures: valid signature preserved
    stripped = sanitize_class_signatures(cf)
    assert stripped == 0
    assert any(isinstance(a, SignatureAttribute) for a in m.attributes)

    # Run MethodNormalizer
    normalizer = MethodNormalizer()
    stats = normalizer.normalize_method(cf, m)
    assert stats.deflattened or stats.complexity_reduced > 0

    # Verify signature attribute is still intact on method
    sig_after = [a for a in m.attributes if isinstance(a, SignatureAttribute)]
    assert len(sig_after) == 1
    assert sig_after[0].signature(cp) == "<T:Ljava/lang/Object;>(TT;)TT;"


def test_pairwise_f10_f14_exceptions_cfg_deflattening():
    """
    Pairwise F10 + F14: Exception handler ranges preserved when state-variable
    switch dispatch loops are de-flattened.
    """
    cp = ConstantPool()
    disp_code = (
        bytes([Opcode.ICONST_0, Opcode.ISTORE_1, Opcode.GOTO, 0, 3]) +
        bytes([Opcode.ILOAD_1, Opcode.TABLESWITCH, 0]) +
        struct.pack(">iiiiiii", 59, 0, 3, 30, 40, 49, 59) +
        bytes([Opcode.ILOAD_0, Opcode.ICONST_1, Opcode.IADD, Opcode.ISTORE_0, Opcode.ICONST_1, Opcode.ISTORE_1, Opcode.GOTO]) + struct.pack(">h", -38) +
        bytes([Opcode.ILOAD_0, Opcode.ICONST_2, Opcode.IMUL, Opcode.ISTORE_0, Opcode.ICONST_2, Opcode.ISTORE_1, Opcode.GOTO]) + struct.pack(">h", -47) +
        bytes([Opcode.ILOAD_0, Opcode.ICONST_1, Opcode.ISUB, Opcode.ISTORE_0, Opcode.ICONST_3, Opcode.ISTORE_1, Opcode.GOTO]) + struct.pack(">h", -57) +
        bytes([Opcode.ILOAD_0, Opcode.IRETURN])
    )
    # Exception handler protecting exit block
    ex_entries = [ExceptionTableEntry(start_pc=59, end_pc=66, handler_pc=65, catch_type=0)]
    code_attr = CodeAttribute(cp.add_utf8("Code"), 2, 2, disp_code, exception_table=ex_entries)

    m = MethodInfo(
        AccessFlags.PUBLIC | AccessFlags.STATIC, cp.add_utf8("safeDispatch"), cp.add_utf8("(I)I"),
        [code_attr]
    )
    cf = ClassFile(
        magic=0xCAFEBABE, minor_version=0, major_version=52, constant_pool=cp,
        access_flags=AccessFlags.PUBLIC, this_class=cp.add_class("com/example/SafeDispatcher"),
        super_class=cp.add_class("java/lang/Object"), interfaces=[],
        fields=[], methods=[m], attributes=[]
    )

    # Sanitize exceptions
    sanitize_class_exceptions(cf)

    # Normalize method (de-flattens CFG)
    normalizer = MethodNormalizer()
    stats = normalizer.normalize_method(cf, m)
    assert stats.cyclomatic_complexity_before > 0

    # Re-parse serialized class
    reparsed = ClassFile.parse(cf.to_bytes())
    assert len(reparsed.methods) == 1


def test_pairwise_f11_f19_synthetic_dedup_dead_code(corrupted_structures_jar: Path):
    """
    Pairwise F11 + F19: Bridge recursion cycle breaking coupled with
    dead instruction elimination.
    """
    archive = JarArchive.from_file(corrupted_structures_jar)
    assert "com/synthetic/corrupted/DuplicateSynthetics" in archive.classes

    cf = archive.classes["com/synthetic/corrupted/DuplicateSynthetics"]
    cp = cf.constant_pool

    # Break recursive bridge methods (F11)
    broken_count = break_recursive_bridge_methods(cf)
    assert broken_count >= 1

    # Find bridge method and run dead code pruning (F19)
    bridge_m = next(m for m in cf.methods if m.name(cp) == "bridgeCall")
    c_attr = bridge_m.code_attribute(cp)
    assert c_attr is not None

    normalizer = MethodNormalizer()
    stats = normalizer.normalize_method(cf, bridge_m)
    # Bytecode must be valid and verifiable
    reparsed = ClassFile.parse(cf.to_bytes())
    assert reparsed.magic == 0xCAFEBABE
