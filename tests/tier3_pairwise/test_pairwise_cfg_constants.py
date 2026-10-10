"""
tests/tier3_pairwise/test_pairwise_cfg_constants.py: Pairwise interactions between M3 CFG and M3 Normalization.
Covers:
- F13 (CFG Basic Block Partitioning) + F16 (Constant Folding Lattice)
- F15 (Opaque Predicate Elimination) + F16 (Constant Folding Lattice)
- F14 (Control Flow De-Flattening) + F16 (Constant Folding Lattice)
- F16 (Constant Folding Lattice) + F17 (Static String Decryptor Resolution)
- F16 (Constant Folding Lattice) + F18 (Dynamic Invoke & Reflection Resolution)
- F15 (Opaque Predicate Elimination) + F19 (Dead Code & NOP Pruning)
"""

import struct
from pathlib import Path
import pytest

from jar_engine.model.constants import Opcode, AccessFlags, ConstantTag
from jar_engine.model.class_file import ClassFile
from jar_engine.model.cp_info import ConstantPool, CONSTANT_String_info
from jar_engine.model.member_info import MethodInfo, CodeAttribute
from jar_engine.model.jar_archive import JarArchive
from jar_engine.cfg import (
    ControlFlowGraph, BasicBlock, Instruction, CFGEdge, CFGEdgeType,
    partition_basic_blocks, ControlFlowDeflattener,
    eliminate_opaque_predicates, prune_dead_code_and_nops,
    prune_unreachable_blocks, compact_goto_chains
)
from jar_engine.normalization import (
    MethodNormalizer,
    fold_constants_and_arithmetic,
    reconstruct_encrypted_strings,
    resolve_dynamic_invokes_and_reflection,
)


def test_pairwise_f13_f16_cfg_constant_folding():
    """
    Pairwise F13 + F16: Basic block partitioning feeds abstract
    interpretation constant folding lattice to simplify multi-opcode expressions.
    """
    # Expression: (20 + 30) * 2 = 100
    code = bytes([
        Opcode.BIPUSH, 20,
        Opcode.BIPUSH, 30,
        Opcode.IADD,
        Opcode.BIPUSH, 2,
        Opcode.IMUL,
        Opcode.IRETURN
    ])
    cfg = ControlFlowGraph.from_code(code)
    assert len(cfg.blocks) >= 1

    folded_count = fold_constants_and_arithmetic(cfg)
    assert folded_count >= 1

    entry_block = cfg.blocks[cfg.entry_block_id]
    # Block should now contain folded push (bipush 100) and ireturn
    opcodes = [inst.opcode for inst in entry_block.instructions]
    assert Opcode.IADD not in opcodes
    assert Opcode.IMUL not in opcodes
    assert Opcode.IRETURN in opcodes
    # Verify constant value is 100
    push_inst = entry_block.instructions[0]
    assert push_inst.opcode in (Opcode.BIPUSH, Opcode.SIPUSH, Opcode.LDC)
    if push_inst.opcode == Opcode.BIPUSH:
        assert push_inst.operands[0] == 100


def test_pairwise_f15_f16_opaque_predicates_constant_folding():
    """
    Pairwise F15 + F16: Constant folding lattice detects algebraic invariants
    ((x ^ x) == 0) to prune bogus opaque predicate branches.
    """
    # (x ^ x) == 0 is always 0, ifeq branch always taken
    code = (
        bytes([Opcode.ILOAD_0, Opcode.ILOAD_0, Opcode.IXOR, Opcode.IFEQ, 0, 7]) +
        bytes([Opcode.BIPUSH, 99, Opcode.IRETURN]) +
        bytes([Opcode.ILOAD_0, Opcode.IRETURN])
    )
    cfg = ControlFlowGraph.from_code(code)
    assert len(cfg.blocks) >= 2

    pruned_count = eliminate_opaque_predicates(cfg)
    assert pruned_count >= 1

    # Verify dead bogus return block is disconnected from entry successors
    entry_succs = cfg.get_successors(cfg.entry_block_id)
    # Target block is successor, fallthrough block is removed
    prune_unreachable_blocks(cfg)
    remaining_ret_values = []
    for b in cfg.blocks.values():
        for inst in b.instructions:
            if inst.opcode == Opcode.BIPUSH and inst.operands == bytes([99]):
                remaining_ret_values.append(99)
    assert 99 not in remaining_ret_values


def test_pairwise_f14_f16_state_variable_constant_propagation():
    """
    Pairwise F14 + F16: Constant propagation across state variable assignments
    resolves dispatch switch targets in control flow de-flattening.
    """
    # State-variable switch dispatch loop:
    # State 0 sets state 1 and returns to header
    # State 1 sets state 2 and returns to header
    # State 2 exits
    disp_code = (
        bytes([Opcode.ICONST_0, Opcode.ISTORE_1, Opcode.GOTO, 0, 3]) +
        bytes([Opcode.ILOAD_1, Opcode.TABLESWITCH, 0]) +
        struct.pack(">iiiiiii", 59, 0, 3, 30, 40, 49, 59) +
        bytes([Opcode.ILOAD_0, Opcode.BIPUSH, 10, Opcode.IADD, Opcode.ISTORE_0, Opcode.ICONST_1, Opcode.ISTORE_1, Opcode.GOTO]) + struct.pack(">h", -38) +
        bytes([Opcode.ILOAD_0, Opcode.ICONST_2, Opcode.IMUL, Opcode.ISTORE_0, Opcode.ICONST_2, Opcode.ISTORE_1, Opcode.GOTO]) + struct.pack(">h", -47) +
        bytes([Opcode.ILOAD_0, Opcode.BIPUSH, 5, Opcode.ISUB, Opcode.ISTORE_0, Opcode.ICONST_3, Opcode.ISTORE_1, Opcode.GOTO]) + struct.pack(">h", -57) +
        bytes([Opcode.ILOAD_0, Opcode.IRETURN])
    )
    cfg = ControlFlowGraph.from_code(disp_code)
    initial_complexity = cfg.cyclomatic_complexity
    assert initial_complexity >= 4

    deflattener = ControlFlowDeflattener(cfg)
    report = deflattener.deflatten()
    assert report.dispatchers_eliminated >= 1


def test_pairwise_f16_f17_constant_folding_string_decryption(constant_obfuscated_jar: Path):
    """
    Pairwise F16 + F17: Constant arithmetic folding calculates cipher keys and
    inlines statically decrypted strings into constant pool.
    """
    archive = JarArchive.from_file(constant_obfuscated_jar)
    assert "com/synthetic/constants/StringObfuscated" in archive.classes

    cf = archive.classes["com/synthetic/constants/StringObfuscated"]
    cp = cf.constant_pool

    m_secret = next(m for m in cf.methods if m.name(cp) == "getSecretMessage")
    m_key = next(m for m in cf.methods if m.name(cp) == "getDatabaseKey")

    normalizer = MethodNormalizer()

    # Reconstruct getSecretMessage -> dec("IFMMP") -> "HELLO"
    stats1 = normalizer.normalize_method(cf, m_secret)
    assert stats1.strings_decrypted == 1

    # Reconstruct getDatabaseKey -> dec("TFDSFU") -> "SECRET"
    stats2 = normalizer.normalize_method(cf, m_key)
    assert stats2.strings_decrypted == 1

    # Verify constant pool now contains plaintext strings
    all_strings = [cp.resolve_utf8(entry.string_index) for _, entry in cp.entries() if isinstance(entry, CONSTANT_String_info)]
    assert "HELLO" in all_strings
    assert "SECRET" in all_strings


def test_pairwise_f16_f18_constant_folding_dynamic_invoke(constant_obfuscated_jar: Path):
    """
    Pairwise F16 + F18: Constant folding evaluates StringConcatFactory recipe
    strings and resolves invokedynamic call sites to constant LDC.
    """
    archive = JarArchive.from_file(constant_obfuscated_jar)
    assert "com/synthetic/constants/DynamicInvoker" in archive.classes

    cf = archive.classes["com/synthetic/constants/DynamicInvoker"]
    cp = cf.constant_pool

    m_concat = next(m for m in cf.methods if m.name(cp) == "concatSecret")
    code_attr = m_concat.code_attribute(cp)
    assert code_attr is not None
    assert Opcode.INVOKEDYNAMIC in code_attr.code

    normalizer = MethodNormalizer()
    stats = normalizer.normalize_method(cf, m_concat)
    assert stats.dynamic_invokes_resolved >= 1

    # Inlined string "Secret: 42" should now be present in constant pool
    all_strings = [cp.resolve_utf8(entry.string_index) for _, entry in cp.entries() if isinstance(entry, CONSTANT_String_info)]
    assert any("Secret: 42" in s for s in all_strings)


def test_pairwise_f15_f19_opaque_predicates_dead_code_pruning():
    """
    Pairwise F15 + F19: Dead blocks created by opaque predicate pruning
    are eliminated by unreachable block pruning and GOTO compaction.
    """
    # (x - x) == 0 algebraic invariant followed by dead unreachable block (bipush 42)
    code = (
        bytes([Opcode.ILOAD_0, Opcode.ILOAD_0, Opcode.ISUB, Opcode.IFEQ, 0, 7]) +
        bytes([Opcode.BIPUSH, 42, Opcode.IRETURN]) +
        bytes([Opcode.ILOAD_0, Opcode.ICONST_1, Opcode.IADD, Opcode.IRETURN])
    )
    cfg = ControlFlowGraph.from_code(code)
    initial_blocks = len(cfg.blocks)

    # 1. Eliminate opaque predicates
    pruned_count = eliminate_opaque_predicates(cfg)
    assert pruned_count >= 1

    # 2. Prune unreachable blocks and compact goto chains
    pruned_count = prune_unreachable_blocks(cfg)
    compact_goto_chains(cfg)

    # Blocks must be strictly fewer than initial count
    assert len(cfg.blocks) < initial_blocks
    # No block should contain dead value 42
    for b in cfg.blocks.values():
        for inst in b.instructions:
            if inst.opcode == Opcode.BIPUSH:
                assert inst.operands != bytes([42])
