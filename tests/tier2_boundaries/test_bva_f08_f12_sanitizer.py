"""
tests/tier2_boundaries/test_bva_f08_f12_sanitizer.py:
Boundary value analysis and corner cases for features F8 through F12.
Total: 25 tests.
"""

import struct
import pytest
from pathlib import Path

from jar_engine.model.constants import AccessFlags, ConstantTag, Opcode
from jar_engine.model.class_file import ClassFile
from jar_engine.model.cp_info import (
    ConstantPool, CONSTANT_Utf8_info, CONSTANT_Class_info
)
from jar_engine.model.member_info import (
    FieldInfo, MethodInfo, CodeAttribute, ExceptionTableEntry,
    SignatureAttribute, StackMapTableAttribute
)
from jar_engine.sanitizer import (
    BytecodeSanitizer,
    sanitize_class_signatures,
    is_valid_class_signature,
    is_valid_method_signature,
    is_valid_field_signature,
    sanitize_code_exceptions,
    sanitize_class_exceptions,
    sanitize_class_stack_maps,
    validate_stack_map_table,
    deduplicate_fields,
    deduplicate_methods,
    break_recursive_bridge_methods,
    verify_sanitized_class_structural_integrity
)
from jar_engine.sanitizer.decompiler_check import (
    find_decompiler_command,
    run_decompiler_check,
)


def create_minimal_test_class(name: str = "com/example/TestSanitize") -> ClassFile:
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
# F8 BOUNDARIES (5 tests)
# ==============================================================================

def test_bva_f08_java5_class_strips_stackmap():
    """BVA F8-1: Java 1.5 class (major_version <= 50) with StackMapTable attribute has it stripped."""
    cf = create_minimal_test_class()
    cf.major_version = 49  # Java 1.5
    cp = cf.constant_pool
    sm_name_idx = cp.add_utf8("StackMapTable")
    code_name_idx = cp.add_utf8("Code")

    payload = struct.pack(">HB", 1, 0)  # valid frame for java 7, but invalid for java 5
    code = CodeAttribute(
        attribute_name_index=code_name_idx,
        max_stack=1,
        max_locals=1,
        code=bytes([Opcode.RETURN]),
        attributes=[StackMapTableAttribute(sm_name_idx, payload)]
    )
    method = MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("m"), cp.add_utf8("()V"), [code])
    cf.methods.append(method)

    stripped = sanitize_class_stack_maps(cf)
    assert stripped == 1
    assert code.get_attribute("StackMapTable", cp) is None


def test_bva_f08_java7_empty_frames():
    """BVA F8-2: Java 7 class (major_version >= 51) with empty stack map entries handled cleanly."""
    cf = create_minimal_test_class()
    cf.major_version = 51  # Java 7
    cp = cf.constant_pool
    sm_name_idx = cp.add_utf8("StackMapTable")
    code_name_idx = cp.add_utf8("Code")

    empty_payload = struct.pack(">H", 0)  # 0 entries
    assert validate_stack_map_table(empty_payload, 50, cp)

    code = CodeAttribute(
        attribute_name_index=code_name_idx,
        max_stack=1,
        max_locals=1,
        code=bytes([Opcode.RETURN]),
        attributes=[StackMapTableAttribute(sm_name_idx, empty_payload)]
    )
    cf.methods.append(MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("m"), cp.add_utf8("()V"), [code]))

    stripped = sanitize_class_stack_maps(cf)
    assert stripped == 0
    assert code.get_attribute("StackMapTable", cp) is not None


def test_bva_f08_offset_delta_beyond_code_length():
    """BVA F8-3: Frame with offset_delta > len(code) stripped or repaired."""
    code_len = 10
    oob_payload = struct.pack(">HB", 1, 20)  # offset delta 20 > code_len 10
    assert not validate_stack_map_table(oob_payload, code_len, None)

    cf = create_minimal_test_class()
    cp = cf.constant_pool
    code = CodeAttribute(
        attribute_name_index=cp.add_utf8("Code"),
        max_stack=1,
        max_locals=1,
        code=bytes([Opcode.NOP] * code_len),
        attributes=[StackMapTableAttribute(cp.add_utf8("StackMapTable"), oob_payload)]
    )
    cf.methods.append(MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("m"), cp.add_utf8("()V"), [code]))

    stripped = sanitize_class_stack_maps(cf)
    assert stripped == 1
    assert code.get_attribute("StackMapTable", cp) is None


def test_bva_f08_all_verification_type_tags():
    """BVA F8-4: Handle all verification types (Top, Integer, Float, Double, Long, Null, UninitializedThis, Object, Uninitialized)."""
    cp = ConstantPool()
    c_obj = cp.add_class("java/lang/Object")

    # Build a full_frame (tag 255) containing all tags in locals:
    # 0: Top, 1: Integer, 2: Float, 3: Double, 4: Long, 5: Null, 6: UninitializedThis
    # 7: Object (c_obj), 8: Uninitialized (offset 0)
    buf = bytearray()
    buf.extend(struct.pack(">H", 1))       # 1 frame entry
    buf.append(255)                        # full_frame tag
    buf.extend(struct.pack(">H", 0))       # offset_delta = 0
    buf.extend(struct.pack(">H", 9))       # 9 locals
    # Append tags 0 to 6
    for tag in range(7):
        buf.append(tag)
    # Tag 7: Object
    buf.append(7)
    buf.extend(struct.pack(">H", c_obj))
    # Tag 8: Uninitialized
    buf.append(8)
    buf.extend(struct.pack(">H", 0))

    buf.extend(struct.pack(">H", 0))       # 0 stack items
    payload = bytes(buf)

    assert validate_stack_map_table(payload, 50, cp)


def test_bva_f08_full_frame_zero_locals_zero_stack():
    """BVA F8-5: Full frame with 0 locals and 0 stack items."""
    buf = bytearray()
    buf.extend(struct.pack(">H", 1))    # 1 frame
    buf.append(255)                     # full_frame tag
    buf.extend(struct.pack(">H", 5))    # offset_delta = 5
    buf.extend(struct.pack(">H", 0))    # 0 locals
    buf.extend(struct.pack(">H", 0))    # 0 stack items
    payload = bytes(buf)

    assert validate_stack_map_table(payload, 20, None)


# ==============================================================================
# F9 BOUNDARIES (5 tests)
# ==============================================================================

def test_bva_f09_empty_string_signature():
    """BVA F9-1: Signature attribute pointing to empty string '' stripped."""
    assert not is_valid_class_signature("")
    assert not is_valid_method_signature("")
    assert not is_valid_field_signature("")

    cf = create_minimal_test_class()
    cp = cf.constant_pool
    sig_name = cp.add_utf8("Signature")
    empty_str = cp.add_utf8("")
    cf.attributes.append(SignatureAttribute(sig_name, empty_str))

    stripped = sanitize_class_signatures(cf)
    assert stripped == 1
    assert cf.get_attribute("Signature") is None


def test_bva_f09_deeply_nested_type_arguments():
    """BVA F9-2: Valid signature with 10 nested type arguments LList<LList<...>>; preserved."""
    nested = "Ljava/lang/String;"
    for _ in range(10):
        nested = f"Ljava/util/List<{nested}>;"

    assert is_valid_class_signature(nested)
    assert is_valid_field_signature(nested)

    cf = create_minimal_test_class()
    cp = cf.constant_pool
    sig_name = cp.add_utf8("Signature")
    sig_idx = cp.add_utf8(nested)
    cf.attributes.append(SignatureAttribute(sig_name, sig_idx))

    stripped = sanitize_class_signatures(cf)
    assert stripped == 0
    assert cf.get_attribute("Signature") is not None


def test_bva_f09_unclosed_angle_bracket():
    """BVA F9-3: Corrupted signature Ljava/util/List< stripped."""
    assert not is_valid_class_signature("Ljava/util/List<")
    assert not is_valid_class_signature("<T:Ljava/lang/Object")

    cf = create_minimal_test_class()
    cp = cf.constant_pool
    sig_name = cp.add_utf8("Signature")
    bad_sig = cp.add_utf8("Ljava/util/List<")
    cf.attributes.append(SignatureAttribute(sig_name, bad_sig))

    stripped = sanitize_class_signatures(cf)
    assert stripped == 1
    assert cf.get_attribute("Signature") is None


def test_bva_f09_void_no_arg_method_signature():
    """BVA F9-4: Method signature ()V correctly verified."""
    assert is_valid_method_signature("()V")
    assert not is_valid_method_signature("()X")
    assert not is_valid_method_signature("(V)V")


def test_bva_f09_formal_type_param_multiple_bounds():
    """BVA F9-5: Formal type parameter <T:LNumber;:LComparable;> validated."""
    sig = "<T:Ljava/lang/Number;:Ljava/lang/Comparable<TT;>;>Ljava/lang/Object;"
    assert is_valid_class_signature(sig)

    cf = create_minimal_test_class()
    cp = cf.constant_pool
    sig_name = cp.add_utf8("Signature")
    sig_idx = cp.add_utf8(sig)
    cf.attributes.append(SignatureAttribute(sig_name, sig_idx))

    stripped = sanitize_class_signatures(cf)
    assert stripped == 0
    assert cf.get_attribute("Signature") is not None


# ==============================================================================
# F10 BOUNDARIES (5 tests)
# ==============================================================================

def test_bva_f10_zero_length_exception_range():
    """BVA F10-1: Exception entry where start_pc == end_pc stripped."""
    code_attr = CodeAttribute(
        attribute_name_index=1,
        max_stack=1, max_locals=1,
        code=bytes([Opcode.NOP] * 20),
        exception_table=[
            ExceptionTableEntry(start_pc=5, end_pc=5, handler_pc=10, catch_type=0)
        ]
    )
    fixed = sanitize_code_exceptions(code_attr)
    assert fixed == 1
    assert len(code_attr.exception_table) == 0


def test_bva_f10_handler_pc_out_of_bounds():
    """BVA F10-2: handler_pc >= len(code) stripped."""
    code_attr = CodeAttribute(
        attribute_name_index=1,
        max_stack=1, max_locals=1,
        code=bytes([Opcode.NOP] * 20),
        exception_table=[
            ExceptionTableEntry(start_pc=0, end_pc=10, handler_pc=20, catch_type=0)  # 20 >= 20
        ]
    )
    fixed = sanitize_code_exceptions(code_attr)
    assert fixed == 1
    assert len(code_attr.exception_table) == 0


def test_bva_f10_handler_pc_equals_start_pc():
    """BVA F10-3: Infinite catching loop where handler_pc == start_pc normalized or removed."""
    code_attr = CodeAttribute(
        attribute_name_index=1,
        max_stack=1, max_locals=1,
        code=bytes([Opcode.NOP] * 30),
        exception_table=[
            # Handler inside protected region [5, 20): handler_pc=5
            ExceptionTableEntry(start_pc=5, end_pc=20, handler_pc=5, catch_type=0)
        ]
    )
    fixed = sanitize_code_exceptions(code_attr)
    assert fixed >= 1
    # When start_pc == handler_pc, protected range before handler is [5, 5) which is zero-length, so entry is removed
    assert len(code_attr.exception_table) == 0


def test_bva_f10_finally_block_catch_type_zero():
    """BVA F10-4: catch_type = 0 (any/finally) preserved."""
    code_attr = CodeAttribute(
        attribute_name_index=1,
        max_stack=1, max_locals=1,
        code=bytes([Opcode.NOP] * 30),
        exception_table=[
            ExceptionTableEntry(start_pc=0, end_pc=10, handler_pc=15, catch_type=0)
        ]
    )
    fixed = sanitize_code_exceptions(code_attr)
    assert fixed == 0
    assert len(code_attr.exception_table) == 1
    assert code_attr.exception_table[0].catch_type == 0


def test_bva_f10_multiple_overlapping_ranges():
    """BVA F10-5: Multiple identical overlapping ranges deduplicated."""
    code_attr = CodeAttribute(
        attribute_name_index=1,
        max_stack=1, max_locals=1,
        code=bytes([Opcode.NOP] * 30),
        exception_table=[
            ExceptionTableEntry(start_pc=0, end_pc=10, handler_pc=15, catch_type=0),
            ExceptionTableEntry(start_pc=0, end_pc=10, handler_pc=15, catch_type=0),
            ExceptionTableEntry(start_pc=0, end_pc=10, handler_pc=15, catch_type=0),
        ]
    )
    fixed = sanitize_code_exceptions(code_attr)
    assert fixed == 2
    assert len(code_attr.exception_table) == 1


# ==============================================================================
# F11 BOUNDARIES (5 tests)
# ==============================================================================

def test_bva_f11_ten_duplicate_synthetic_fields():
    """BVA F11-1: 10 duplicate synthetic fields reduced to unique non-conflicting names."""
    cf = create_minimal_test_class()
    cp = cf.constant_pool
    f_name = cp.add_utf8("syntheticField")
    f_desc = cp.add_utf8("I")

    for _ in range(10):
        cf.fields.append(FieldInfo(AccessFlags.PUBLIC | AccessFlags.SYNTHETIC, f_name, f_desc, []))

    renamed = deduplicate_fields(cf)
    assert renamed == 9
    names = [f.name(cp) for f in cf.fields]
    assert len(set(names)) == 10
    assert "syntheticField" in names
    assert "syntheticField$dup$1" in names
    assert "syntheticField$dup$9" in names


def test_bva_f11_duplicate_synthetic_methods():
    """BVA F11-2: Identical synthetic methods deduplicated."""
    cf = create_minimal_test_class()
    cp = cf.constant_pool
    m_name = cp.add_utf8("dupMethod")
    m_desc = cp.add_utf8("()V")

    # 1 regular method
    cf.methods.append(MethodInfo(AccessFlags.PUBLIC, m_name, m_desc, [CodeAttribute(cp.add_utf8("Code"), 1, 1, bytes([Opcode.RETURN]))]))
    # 3 duplicate synthetic methods
    for _ in range(3):
        cf.methods.append(MethodInfo(AccessFlags.PUBLIC | AccessFlags.SYNTHETIC, m_name, m_desc, []))

    pruned = deduplicate_methods(cf)
    assert pruned == 3
    assert len(cf.methods) == 1
    assert not (cf.methods[0].access_flags & AccessFlags.SYNTHETIC)


def test_bva_f11_self_calling_bridge_method():
    """BVA F11-3: Bridge method calling itself eliminated."""
    cf = create_minimal_test_class("com/example/BridgeLoop")
    cp = cf.constant_pool
    m_name = cp.add_utf8("bridgeAction")
    m_desc = cp.add_utf8("()V")

    mref = cp.add_member_ref(ConstantTag.METHODREF, "com/example/BridgeLoop", "bridgeAction", "()V")
    code_bytes = bytes([Opcode.INVOKEVIRTUAL]) + struct.pack(">H", mref) + bytes([Opcode.RETURN])
    code_attr = CodeAttribute(cp.add_utf8("Code"), 1, 1, code_bytes, [], [])
    bridge_m = MethodInfo(AccessFlags.PUBLIC | AccessFlags.BRIDGE, m_name, m_desc, [code_attr])
    cf.methods.append(bridge_m)

    broken = break_recursive_bridge_methods(cf)
    assert broken == 1
    assert code_attr.code == bytes([Opcode.RETURN])


def test_bva_f11_non_synthetic_duplicate_members():
    """BVA F11-4: Non-synthetic duplicates handled without data loss."""
    cf = create_minimal_test_class()
    cp = cf.constant_pool
    f_name = cp.add_utf8("val")
    # Two non-synthetic fields with different descriptors
    f1 = FieldInfo(AccessFlags.PUBLIC, f_name, cp.add_utf8("I"), [])
    f2 = FieldInfo(AccessFlags.PUBLIC, f_name, cp.add_utf8("Ljava/lang/String;"), [])
    cf.fields.extend([f1, f2])

    renamed = deduplicate_fields(cf)
    # Secondary field is renamed to prevent decompiler collision without dropping data
    assert renamed == 1
    assert len(cf.fields) == 2
    assert cf.fields[0].name(cp) == "val"
    assert cf.fields[1].name(cp) == "val$dup$1"
    assert cf.fields[0].descriptor(cp) == "I"
    assert cf.fields[1].descriptor(cp) == "Ljava/lang/String;"


def test_bva_f11_bridge_method_covariant_return():
    """BVA F11-5: Bridge method delegating to method with covariant return type preserved."""
    cf = create_minimal_test_class("com/example/Covariant")
    cp = cf.constant_pool

    # Bridge method: get()Ljava/lang/Object; calls get()Ljava/lang/String;
    target_mref = cp.add_member_ref(ConstantTag.METHODREF, "com/example/Covariant", "get", "()Ljava/lang/String;")
    bridge_code = bytes([
        Opcode.ALOAD_0,
        Opcode.INVOKEVIRTUAL, (target_mref >> 8) & 0xFF, target_mref & 0xFF,
        Opcode.ARETURN
    ])
    code_attr = CodeAttribute(cp.add_utf8("Code"), 1, 1, bridge_code, [], [])
    bridge_m = MethodInfo(
        AccessFlags.PUBLIC | AccessFlags.BRIDGE,
        cp.add_utf8("get"),
        cp.add_utf8("()Ljava/lang/Object;"),
        [code_attr]
    )
    cf.methods.append(bridge_m)

    broken = break_recursive_bridge_methods(cf)
    assert broken == 0  # not a self-recursive loop because descriptor differs
    assert code_attr.code == bridge_code


# ==============================================================================
# F12 BOUNDARIES (5 tests)
# ==============================================================================

def test_bva_f12_zero_timeout_decompiler_check(tmp_path: Path):
    """BVA F12-1: Zero/fractional timeout handled cleanly with timeout status."""
    dummy_jar = tmp_path / "test.jar"
    dummy_jar.write_bytes(b"")
    results = run_decompiler_check(dummy_jar, timeout_seconds=0)
    assert isinstance(results, dict)
    assert "cfr" in results
    assert "vineflower" in results
    assert "procyon" in results


def test_bva_f12_missing_decompiler_binary(tmp_path: Path):
    """BVA F12-2: Graceful fallback when decompiler binary does not exist on PATH."""
    dummy_jar = tmp_path / "test.jar"
    dummy_jar.write_bytes(b"")
    results = run_decompiler_check(dummy_jar)
    for b in ["cfr", "vineflower", "procyon"]:
        res = results[b]
        # Should not throw exception, should indicate available status cleanly
        assert isinstance(res.available, bool)
        assert isinstance(res.success, bool)


def test_bva_f12_structural_integrity_minimal_class():
    """BVA F12-3: Structural integrity check on minimal valid class passes."""
    cf = create_minimal_test_class()
    valid, issues = verify_sanitized_class_structural_integrity(cf)
    assert valid
    assert len(issues) == 0


def test_bva_f12_structural_integrity_corrupt_class():
    """BVA F12-4: Structural integrity check on corrupt class reports all issues."""
    cf = create_minimal_test_class()
    cp = cf.constant_pool
    # Add corrupted signature
    sig_name = cp.add_utf8("Signature")
    bad_sig = cp.add_utf8("<Corrupted")
    cf.attributes.append(SignatureAttribute(sig_name, bad_sig))

    valid, issues = verify_sanitized_class_structural_integrity(cf)
    assert not valid
    assert len(issues) >= 1
    assert any("signature" in iss.lower() for iss in issues)


def test_bva_f12_multiple_structural_issues():
    """BVA F12-5: Multiple concurrent structural issues captured in single report."""
    cf = create_minimal_test_class()
    cp = cf.constant_pool
    # 1. Corrupt signature
    cf.attributes.append(SignatureAttribute(cp.add_utf8("Signature"), cp.add_utf8("<Unclosed")))

    # 2. Duplicate field
    f_name = cp.add_utf8("dupField")
    f_desc = cp.add_utf8("I")
    cf.fields.append(FieldInfo(AccessFlags.PUBLIC, f_name, f_desc, []))
    cf.fields.append(FieldInfo(AccessFlags.PUBLIC, f_name, f_desc, []))

    valid, issues = verify_sanitized_class_structural_integrity(cf)
    assert not valid
    assert len(issues) >= 2
