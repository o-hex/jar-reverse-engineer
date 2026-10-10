"""
Tier 1: Feature Coverage for F8 - F12 (Bytecode Preprocessing & Decompiler Sanitizer).
Tests 25 isolated features (5 per feature F8 through F12) per explorer_e2e_3 specification.
"""

from __future__ import annotations
import struct
import io
from pathlib import Path
from typing import Dict, List

import pytest

from jar_engine.model.constants import AccessFlags, Opcode, ConstantTag
from jar_engine.model.cp_info import ConstantPool
from jar_engine.model.class_file import ClassFile
from jar_engine.model.member_info import (
    FieldInfo, MethodInfo, CodeAttribute, ExceptionTableEntry,
    SignatureAttribute, StackMapTableAttribute
)
from jar_engine.model.jar_archive import JarArchive
from jar_engine.sanitizer import (
    BytecodeSanitizer,
    is_valid_class_signature,
    is_valid_method_signature,
    is_valid_field_signature,
    sanitize_class_signatures,
    sanitize_code_exceptions,
    sanitize_class_exceptions,
    sanitize_class_stack_maps,
    validate_stack_map_table,
    deduplicate_fields,
    deduplicate_methods,
    break_recursive_bridge_methods,
    verify_sanitized_class_structural_integrity
)
from jar_engine.sanitizer.stack_map import validate_verification_type_info
from jar_engine.sanitizer.decompiler_check import (
    find_decompiler_command,
    run_decompiler_check
)


# ============================================================================
# F8: StackMapTable Normalization
# ============================================================================

def test_f08_legacy_class_stripping(make_test_class):
    """F8.1: Strip StackMapTable for classes with version <= 50 (Java 6)."""
    cf = make_test_class("pkg/Legacy")
    cf.major_version = 50  # Java 6
    cp = cf.constant_pool
    sm_idx = cp.add_utf8("StackMapTable")
    code_idx = cp.add_utf8("Code")

    valid_payload = struct.pack(">HB", 1, 0)
    code = CodeAttribute(
        attribute_name_index=code_idx,
        max_stack=1, max_locals=1,
        code=bytes([Opcode.RETURN]),
        attributes=[StackMapTableAttribute(sm_idx, valid_payload)]
    )
    cf.methods.append(MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("m"), cp.add_utf8("()V"), [code]))

    stripped = sanitize_class_stack_maps(cf)
    assert stripped == 1
    assert code.get_attribute("StackMapTable", cp) is None


def test_f08_java7_validation(make_test_class):
    """F8.2: Validate StackMapTable presence and frame consistency for version >= 51."""
    cf = make_test_class("pkg/Modern")
    cf.major_version = 52  # Java 8
    cp = cf.constant_pool
    sm_idx = cp.add_utf8("StackMapTable")
    code_idx = cp.add_utf8("Code")

    valid_payload = struct.pack(">HB", 1, 0)  # 1 same_frame (offset 0)
    code = CodeAttribute(
        attribute_name_index=code_idx,
        max_stack=1, max_locals=1,
        code=bytes([Opcode.RETURN]),
        attributes=[StackMapTableAttribute(sm_idx, valid_payload)]
    )
    cf.methods.append(MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("m"), cp.add_utf8("()V"), [code]))

    stripped = sanitize_class_stack_maps(cf)
    assert stripped == 0
    assert code.get_attribute("StackMapTable", cp) is not None


def test_f08_corrupt_offset_delta_repair(make_test_class):
    """F8.3: Detect and repair invalid frame offset deltas exceeding bytecode length."""
    cf = make_test_class("pkg/CorruptDelta")
    cf.major_version = 52
    cp = cf.constant_pool
    sm_idx = cp.add_utf8("StackMapTable")
    code_idx = cp.add_utf8("Code")

    # Bytecode length is 5, but frame offset delta is 50 -> exceeds code length
    oob_payload = struct.pack(">HB", 1, 50)
    code = CodeAttribute(
        attribute_name_index=code_idx,
        max_stack=1, max_locals=1,
        code=bytes([Opcode.NOP] * 4 + [Opcode.RETURN]),
        attributes=[StackMapTableAttribute(sm_idx, oob_payload)]
    )
    cf.methods.append(MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("m"), cp.add_utf8("()V"), [code]))

    stripped = sanitize_class_stack_maps(cf)
    assert stripped == 1
    assert code.get_attribute("StackMapTable", cp) is None


def test_f08_frame_type_consistency():
    """F8.4: Verify verification type tags match variable types (Object, Integer, Top)."""
    # Test valid verification types via validate_verification_type_info
    # Top (tag 0), Integer (tag 1), Null (tag 5)
    for tag in [0, 1, 2, 3, 4, 5, 6]:
        stream = io.BytesIO(bytes([tag]))
        assert validate_verification_type_info(stream, 10, None)

    # Object_variable_info (tag 7) requires valid CP class entry
    cp = ConstantPool()
    cls_idx = cp.add_class("java/lang/String")
    obj_stream = io.BytesIO(bytes([7]) + struct.pack(">H", cls_idx))
    assert validate_verification_type_info(obj_stream, 10, cp)

    # Invalid tag 99 fails
    bad_stream = io.BytesIO(bytes([99]))
    assert not validate_verification_type_info(bad_stream, 10, cp)


def test_f08_sanitizer_stack_map_pass(fixture_jars):
    """F8.5: Verify sanitize_class_stack_maps produces clean class files from corrupted_structures.jar."""
    arc = JarArchive.from_file(fixture_jars["corrupted_structures"])
    cf = arc.get_class("com/synthetic/corrupted/CorruptStackMaps")
    assert cf is not None

    stripped = sanitize_class_stack_maps(cf)
    assert stripped >= 1

    # Verify no corrupt stack maps remain
    valid, issues = verify_sanitized_class_structural_integrity(cf)
    stack_map_issues = [i for i in issues if "StackMap" in i]
    assert len(stack_map_issues) == 0


# ============================================================================
# F9: Generic Signature Sanitizer
# ============================================================================

def test_f09_valid_class_signatures_preserved(make_test_class):
    """F9.1: JVMS §4.7.9 compliant class signatures retained intact."""
    cf = make_test_class("pkg/GenericClass")
    cp = cf.constant_pool
    sig_idx = cp.add_utf8("Signature")
    valid_sig = "<T:Ljava/lang/Object;>Ljava/lang/Object;Ljava/lang/Comparable<TT;>;"
    cf.attributes.append(SignatureAttribute(sig_idx, cp.add_utf8(valid_sig)))

    assert is_valid_class_signature(valid_sig)
    stripped = sanitize_class_signatures(cf)
    assert stripped == 0
    attr = cf.get_attribute("Signature")
    assert attr is not None
    assert attr.signature(cp) == valid_sig


def test_f09_corrupt_class_signatures_stripped(make_test_class):
    """F9.2: Truncated or syntactically malformed signatures stripped."""
    cf = make_test_class("pkg/CorruptClass")
    cp = cf.constant_pool
    sig_idx = cp.add_utf8("Signature")
    bad_sig = "<T:Ljava/lang/Object"  # unclosed parameter bracket
    cf.attributes.append(SignatureAttribute(sig_idx, cp.add_utf8(bad_sig)))

    assert not is_valid_class_signature(bad_sig)
    stripped = sanitize_class_signatures(cf)
    assert stripped == 1
    assert cf.get_attribute("Signature") is None


def test_f09_valid_method_signatures_preserved(make_test_class):
    """F9.3: Method signatures with type parameters and bounds preserved."""
    cf = make_test_class("pkg/MethodClass")
    cp = cf.constant_pool
    sig_idx = cp.add_utf8("Signature")
    valid_m_sig = "<E:Ljava/lang/Number;>(TE;)Ljava/util/List<TE;>;"
    assert is_valid_method_signature(valid_m_sig)

    m = MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("process"), cp.add_utf8("(Ljava/lang/Number;)Ljava/util/List;"), [
        SignatureAttribute(sig_idx, cp.add_utf8(valid_m_sig)),
        CodeAttribute(cp.add_utf8("Code"), 1, 1, bytes([Opcode.ACONST_NULL, Opcode.ARETURN]))
    ])
    cf.methods.append(m)

    stripped = sanitize_class_signatures(cf)
    assert stripped == 0
    attr = m.get_attribute("Signature", cp)
    assert attr is not None
    assert attr.signature(cp) == valid_m_sig


def test_f09_corrupt_method_signatures_stripped(make_test_class):
    """F9.4: Method signatures with unclosed brackets stripped."""
    cf = make_test_class("pkg/CorruptMethodClass")
    cp = cf.constant_pool
    sig_idx = cp.add_utf8("Signature")
    bad_m_sig = "(Ljava/lang/String;)Ljava/util/List<Ljava/lang/String;"  # missing '>' and ';'
    assert not is_valid_method_signature(bad_m_sig)

    m = MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("broken"), cp.add_utf8("(Ljava/lang/String;)Ljava/util/List;"), [
        SignatureAttribute(sig_idx, cp.add_utf8(bad_m_sig)),
        CodeAttribute(cp.add_utf8("Code"), 1, 1, bytes([Opcode.ACONST_NULL, Opcode.ARETURN]))
    ])
    cf.methods.append(m)

    stripped = sanitize_class_signatures(cf)
    assert stripped == 1
    assert m.get_attribute("Signature", cp) is None


def test_f09_valid_and_corrupt_field_signatures(make_test_class):
    """F9.5: Field signature verification and stripping of corrupted tags."""
    cf = make_test_class("pkg/FieldClass")
    cp = cf.constant_pool
    sig_idx = cp.add_utf8("Signature")

    good_sig = "Ljava/util/Map<Ljava/lang/String;Ljava/lang/Integer;>;"
    bad_sig = "Lbroken/Field"  # missing terminal ';'

    assert is_valid_field_signature(good_sig)
    assert not is_valid_field_signature(bad_sig)

    f_good = FieldInfo(AccessFlags.PRIVATE, cp.add_utf8("goodField"), cp.add_utf8("Ljava/util/Map;"), [
        SignatureAttribute(sig_idx, cp.add_utf8(good_sig))
    ])
    f_bad = FieldInfo(AccessFlags.PRIVATE, cp.add_utf8("badField"), cp.add_utf8("Ljava/lang/Object;"), [
        SignatureAttribute(sig_idx, cp.add_utf8(bad_sig))
    ])
    cf.fields.extend([f_good, f_bad])

    stripped = sanitize_class_signatures(cf)
    assert stripped == 1
    assert f_good.get_attribute("Signature", cp) is not None
    assert f_bad.get_attribute("Signature", cp) is None


# ============================================================================
# F10: Irregular Exception Handler Fixer
# ============================================================================

def test_f10_inverted_range_removal():
    """F10.1: Remove handlers where start_pc >= end_pc."""
    code_bytes = bytes([Opcode.NOP] * 30 + [Opcode.RETURN])
    code_attr = CodeAttribute(
        attribute_name_index=1,
        max_stack=1, max_locals=1,
        code=code_bytes,
        exception_table=[
            ExceptionTableEntry(start_pc=20, end_pc=10, handler_pc=25, catch_type=0),  # Inverted
            ExceptionTableEntry(start_pc=15, end_pc=15, handler_pc=25, catch_type=0),  # Zero-length
        ]
    )

    fixed = sanitize_code_exceptions(code_attr)
    assert fixed == 2
    assert len(code_attr.exception_table) == 0


def test_f10_out_of_bounds_pc_removal():
    """F10.2: Remove handlers referencing PCs beyond bytecode length."""
    code_bytes = bytes([Opcode.NOP] * 20 + [Opcode.RETURN])  # length = 21
    code_attr = CodeAttribute(
        attribute_name_index=1,
        max_stack=1, max_locals=1,
        code=code_bytes,
        exception_table=[
            ExceptionTableEntry(start_pc=0, end_pc=50, handler_pc=10, catch_type=0),   # end_pc OOB
            ExceptionTableEntry(start_pc=0, end_pc=10, handler_pc=100, catch_type=0),  # handler_pc OOB
        ]
    )

    fixed = sanitize_code_exceptions(code_attr)
    assert fixed == 2
    assert len(code_attr.exception_table) == 0


def test_f10_self_catching_loop_breaker():
    """F10.3: Break recursive exception loops where handler_pc == start_pc or within range."""
    code_bytes = bytes([Opcode.NOP] * 40 + [Opcode.RETURN])
    code_attr = CodeAttribute(
        attribute_name_index=1,
        max_stack=1, max_locals=1,
        code=code_bytes,
        exception_table=[
            # Handler inside protected region: 5 <= 15 < 25 -> self catching
            ExceptionTableEntry(start_pc=5, end_pc=25, handler_pc=15, catch_type=0),
        ]
    )

    fixed = sanitize_code_exceptions(code_attr)
    assert fixed >= 1
    # Truncated to end before handler_pc
    assert len(code_attr.exception_table) == 1
    entry = code_attr.exception_table[0]
    assert entry.start_pc == 5
    assert entry.end_pc == 15
    assert entry.handler_pc == 15


def test_f10_valid_exception_ranges_preserved():
    """F10.4: Legitimate try-catch blocks preserved intact."""
    code_bytes = bytes([Opcode.NOP] * 50 + [Opcode.RETURN])
    code_attr = CodeAttribute(
        attribute_name_index=1,
        max_stack=1, max_locals=1,
        code=code_bytes,
        exception_table=[
            ExceptionTableEntry(start_pc=0, end_pc=20, handler_pc=30, catch_type=0),
            ExceptionTableEntry(start_pc=20, end_pc=30, handler_pc=40, catch_type=0),
        ]
    )

    fixed = sanitize_code_exceptions(code_attr)
    assert fixed == 0
    assert len(code_attr.exception_table) == 2
    assert code_attr.exception_table[0].start_pc == 0
    assert code_attr.exception_table[0].end_pc == 20
    assert code_attr.exception_table[1].start_pc == 20
    assert code_attr.exception_table[1].end_pc == 30


def test_f10_whole_class_exception_sanitization(make_test_class):
    """F10.5: Verify sanitize_class_exceptions processes all methods in class."""
    cf = make_test_class("pkg/MultiEx")
    cp = cf.constant_pool

    code1 = CodeAttribute(cp.add_utf8("Code"), 1, 1, bytes([Opcode.RETURN] * 10), exception_table=[
        ExceptionTableEntry(start_pc=8, end_pc=2, handler_pc=1, catch_type=0)  # Inverted
    ])
    code2 = CodeAttribute(cp.add_utf8("Code"), 1, 1, bytes([Opcode.RETURN] * 10), exception_table=[
        ExceptionTableEntry(start_pc=0, end_pc=50, handler_pc=1, catch_type=0)  # OOB
    ])
    cf.methods.append(MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("m1"), cp.add_utf8("()V"), [code1]))
    cf.methods.append(MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("m2"), cp.add_utf8("()V"), [code2]))

    fixed = sanitize_class_exceptions(cf)
    assert fixed == 2
    assert len(code1.exception_table) == 0
    assert len(code2.exception_table) == 0


# ============================================================================
# F11: Duplicate Synthetic Member Deduplication
# ============================================================================

def test_f11_duplicate_synthetic_field_dedup(make_test_class):
    """F11.1: Deduplicate identical synthetic fields generated by obfuscators."""
    cf = make_test_class("pkg/DupFields")
    cp = cf.constant_pool
    name_idx = cp.add_utf8("secret")
    desc_i = cp.add_utf8("I")
    desc_s = cp.add_utf8("Ljava/lang/String;")

    f_orig = FieldInfo(AccessFlags.PUBLIC, name_idx, desc_i, [])
    f_synth = FieldInfo(AccessFlags.PUBLIC | AccessFlags.SYNTHETIC, name_idx, desc_s, [])
    cf.fields.extend([f_orig, f_synth])

    renamed = deduplicate_fields(cf)
    assert renamed == 1
    assert f_orig.name(cp) == "secret"
    assert f_synth.name(cp) == "secret$dup$1"


def test_f11_duplicate_synthetic_method_dedup(make_test_class):
    """F11.2: Deduplicate duplicate synthetic methods."""
    cf = make_test_class("pkg/DupMethods")
    cp = cf.constant_pool
    name_idx = cp.add_utf8("runAction")
    desc_idx = cp.add_utf8("()V")

    m_real = MethodInfo(AccessFlags.PUBLIC, name_idx, desc_idx, [
        CodeAttribute(cp.add_utf8("Code"), 1, 1, bytes([Opcode.RETURN]))
    ])
    m_synth = MethodInfo(AccessFlags.PUBLIC | AccessFlags.SYNTHETIC, name_idx, desc_idx, [])
    cf.methods.extend([m_real, m_synth])

    pruned = deduplicate_methods(cf)
    assert pruned == 1
    assert len(cf.methods) == 1
    assert cf.methods[0] is m_real


def test_f11_distinct_member_preservation(make_test_class):
    """F11.3: Ensure distinct non-conflicting members are not deduplicated."""
    cf = make_test_class("pkg/DistinctMembers")
    cp = cf.constant_pool

    f1 = FieldInfo(AccessFlags.PRIVATE, cp.add_utf8("x"), cp.add_utf8("I"), [])
    f2 = FieldInfo(AccessFlags.PRIVATE, cp.add_utf8("y"), cp.add_utf8("I"), [])
    cf.fields.extend([f1, f2])

    m1 = MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("getX"), cp.add_utf8("()I"), [
        CodeAttribute(cp.add_utf8("Code"), 1, 1, bytes([Opcode.ICONST_0, Opcode.IRETURN]))
    ])
    m2 = MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("getY"), cp.add_utf8("()I"), [
        CodeAttribute(cp.add_utf8("Code"), 1, 1, bytes([Opcode.ICONST_0, Opcode.IRETURN]))
    ])
    cf.methods.extend([m1, m2])

    assert deduplicate_fields(cf) == 0
    assert deduplicate_methods(cf) == 0
    assert len(cf.fields) == 2
    assert len(cf.methods) == 2


def test_f11_recursive_bridge_cycle_breaking(make_test_class):
    """F11.4: Detect and break self-referential bridge method loops."""
    cf = make_test_class("pkg/BridgeRecurse")
    cp = cf.constant_pool
    m_name = cp.add_utf8("bridgeLoop")
    m_desc = cp.add_utf8("()V")

    mref_idx = cp.add_member_ref(ConstantTag.METHODREF, "pkg/BridgeRecurse", "bridgeLoop", "()V")
    code_bytes = bytes([Opcode.INVOKEVIRTUAL]) + struct.pack(">H", mref_idx) + bytes([Opcode.RETURN])

    code_attr = CodeAttribute(cp.add_utf8("Code"), 1, 1, code_bytes)
    bridge_m = MethodInfo(AccessFlags.PUBLIC | AccessFlags.BRIDGE, m_name, m_desc, [code_attr])
    cf.methods.append(bridge_m)

    cycles_broken = break_recursive_bridge_methods(cf)
    assert cycles_broken == 1
    assert code_attr.code == bytes([Opcode.RETURN])


def test_f11_non_synthetic_member_preservation(make_test_class):
    """F11.5: Non-synthetic members with identical names/types preserved or handled gracefully."""
    cf = make_test_class("pkg/NonSyntheticMembers")
    cp = cf.constant_pool

    m1 = MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("overloaded"), cp.add_utf8("(I)V"), [
        CodeAttribute(cp.add_utf8("Code"), 1, 1, bytes([Opcode.RETURN]))
    ])
    m2 = MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("overloaded"), cp.add_utf8("(Ljava/lang/String;)V"), [
        CodeAttribute(cp.add_utf8("Code"), 1, 1, bytes([Opcode.RETURN]))
    ])
    cf.methods.extend([m1, m2])

    # Distinct descriptors are valid overloads, should not be pruned
    pruned = deduplicate_methods(cf)
    assert pruned == 0
    assert len(cf.methods) == 2


# ============================================================================
# F12: Decompiler Backend Verification
# ============================================================================

def test_f12_structural_integrity_check(make_test_class):
    """F12.1: Run verify_sanitized_class_structural_integrity on sanitized class."""
    cf = make_test_class("pkg/SanitizedClass")
    valid, issues = verify_sanitized_class_structural_integrity(cf)
    assert valid
    assert len(issues) == 0


def test_f12_decompiler_detection_harness():
    """F12.2: Verify find_decompiler_command detects available backends or falls back cleanly."""
    for backend in ["cfr", "vineflower", "procyon"]:
        cmd = find_decompiler_command(backend)
        # Should return list of strings or None, without raising exceptions
        assert cmd is None or isinstance(cmd, list)


def test_f12_decompiler_run_fallback(make_test_class, temp_workspace):
    """F12.3: Verify run_decompiler_check succeeds with available=False when tools are absent."""
    cf = make_test_class("pkg/FallbackCheck")
    cf_path = temp_workspace / "FallbackCheck.class"
    cf_path.write_bytes(cf.to_bytes())

    results = run_decompiler_check(cf_path)
    assert "cfr" in results
    assert "vineflower" in results
    assert "procyon" in results
    for b, res in results.items():
        assert isinstance(res.available, bool)
        assert isinstance(res.success, bool)


def test_f12_clean_class_zero_issues(make_test_class):
    """F12.4: Confirm clean bytecode produces zero structural integrity issues."""
    cf = make_test_class("pkg/CleanClass")
    cp = cf.constant_pool
    cf.fields.append(FieldInfo(AccessFlags.PUBLIC, cp.add_utf8("name"), cp.add_utf8("Ljava/lang/String;"), []))
    cf.methods.append(MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("getName"), cp.add_utf8("()Ljava/lang/String;"), [
        CodeAttribute(cp.add_utf8("Code"), 1, 1, bytes([Opcode.ACONST_NULL, Opcode.ARETURN]))
    ]))

    valid, issues = verify_sanitized_class_structural_integrity(cf)
    assert valid
    assert issues == []


def test_f12_corrupted_class_issue_detection(make_test_class):
    """F12.5: Confirm malformed signatures and ranges are detected before sanitization."""
    cf = make_test_class("pkg/CorruptClass")
    cp = cf.constant_pool

    # Corrupt class signature
    cf.attributes.append(SignatureAttribute(cp.add_utf8("Signature"), cp.add_utf8("<Unclosed")))

    # Corrupt exception handler
    code = CodeAttribute(cp.add_utf8("Code"), 1, 1, bytes([Opcode.RETURN]), exception_table=[
        ExceptionTableEntry(start_pc=10, end_pc=2, handler_pc=1, catch_type=0)
    ])
    cf.methods.append(MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("corruptM"), cp.add_utf8("()V"), [code]))

    valid, issues = verify_sanitized_class_structural_integrity(cf)
    assert not valid
    assert len(issues) >= 2
    assert any("signature" in i.lower() for i in issues)
    assert any("exception" in i.lower() for i in issues)
