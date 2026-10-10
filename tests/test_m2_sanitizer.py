"""
tests/test_m2_sanitizer.py: Comprehensive test suite for Milestone M2 (Bytecode Preprocessing & Decompiler Sanitizer).
Covers features F8, F9, F10, F11, F12.
"""

import pytest
import struct

from jar_engine.model.constants import ConstantTag, AccessFlags, Opcode
from jar_engine.model.class_file import ClassFile
from jar_engine.model.cp_info import (
    ConstantPool, CONSTANT_Utf8_info, CONSTANT_Class_info, CONSTANT_Methodref_info, CONSTANT_NameAndType_info
)
from jar_engine.model.member_info import (
    FieldInfo, MethodInfo, CodeAttribute, ExceptionTableEntry,
    SignatureAttribute, StackMapTableAttribute, RawAttribute
)
from jar_engine.model.jar_archive import JarArchive
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


def create_minimal_class(name: str = "com/example/Test") -> ClassFile:
    """Helper to create a valid minimal ClassFile in memory."""
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


# --- F9: Generic Signature Tests ---

def test_signature_grammars():
    # Valid class signatures
    assert is_valid_class_signature("Ljava/lang/Object;")
    assert is_valid_class_signature("<T:Ljava/lang/Object;>Ljava/lang/Object;")
    assert is_valid_class_signature("<K:Ljava/lang/Object;V:Ljava/lang/Object;>Ljava/lang/Object;Ljava/util/Map<TK;TV;>;")
    assert is_valid_class_signature("Ljava/util/ArrayList<Ljava/lang/String;>;")

    # Invalid class signatures (obfuscator corruption tricks)
    assert not is_valid_class_signature("<T:Ljava/lang/Object")  # unclosed parameter
    assert not is_valid_class_signature("Ljava/lang/Object")     # missing terminal ';'
    assert not is_valid_class_signature("<>Ljava/lang/Object;")  # empty type parameters
    assert not is_valid_class_signature("NOT_A_SIGNATURE")

    # Method signatures
    assert is_valid_method_signature("()V")
    assert is_valid_method_signature("(Ljava/lang/String;)I")
    assert is_valid_method_signature("<T:Ljava/lang/Object;>(TT;)V")
    assert is_valid_method_signature("()Ljava/util/List<Ljava/lang/String;>;")
    assert not is_valid_method_signature("(Ljava/lang/String;)")  # missing return type
    assert not is_valid_method_signature("<T>(TT;)V")             # missing class bound colon

    # Field signatures
    assert is_valid_field_signature("Ljava/lang/String;")
    assert is_valid_field_signature("TT;")
    assert is_valid_field_signature("[Ljava/util/List<Ljava/lang/String;>;")
    assert not is_valid_field_signature("I")  # BaseType is not ReferenceTypeSignature per JVMS
    assert not is_valid_field_signature("Ljava/lang/String")


def test_sanitize_signatures_strips_corrupted():
    cf = create_minimal_class()
    cp = cf.constant_pool
    sig_name_idx = cp.add_utf8("Signature")

    # Corrupt class signature
    bad_class_sig_idx = cp.add_utf8("<T:corrupted_unclosed")
    cf.attributes.append(SignatureAttribute(sig_name_idx, bad_class_sig_idx))

    # Field with corrupt signature
    f_name_idx = cp.add_utf8("corruptField")
    f_desc_idx = cp.add_utf8("Ljava/lang/Object;")
    bad_field_sig_idx = cp.add_utf8("Lbroken/Field")  # missing ';'
    field = FieldInfo(AccessFlags.PRIVATE, f_name_idx, f_desc_idx, [
        SignatureAttribute(sig_name_idx, bad_field_sig_idx)
    ])
    cf.fields.append(field)

    # Method with valid signature
    m_name_idx = cp.add_utf8("goodMethod")
    m_desc_idx = cp.add_utf8("()V")
    good_m_sig_idx = cp.add_utf8("()V")
    method = MethodInfo(AccessFlags.PUBLIC, m_name_idx, m_desc_idx, [
        SignatureAttribute(sig_name_idx, good_m_sig_idx)
    ])
    cf.methods.append(method)

    stripped = sanitize_class_signatures(cf)
    assert stripped == 2  # 1 bad class sig + 1 bad field sig

    # Bad signatures removed
    assert cf.get_attribute("Signature") is None
    assert field.get_attribute("Signature", cp) is None
    # Good method signature preserved
    assert method.get_attribute("Signature", cp) is not None


# --- F10: Exception Handler Normalization Tests ---

def test_sanitize_exceptions_inverted_and_oob():
    code_bytes = bytes([Opcode.NOP] * 50)
    code_attr = CodeAttribute(
        attribute_name_index=1,
        max_stack=2,
        max_locals=2,
        code=code_bytes,
        exception_table=[
            ExceptionTableEntry(start_pc=10, end_pc=5, handler_pc=20, catch_type=0),   # Inverted range
            ExceptionTableEntry(start_pc=5, end_pc=5, handler_pc=20, catch_type=0),    # Zero-length
            ExceptionTableEntry(start_pc=0, end_pc=60, handler_pc=20, catch_type=0),   # end_pc OOB (> 50)
            ExceptionTableEntry(start_pc=0, end_pc=20, handler_pc=55, catch_type=0),   # handler_pc OOB (> 50)
            ExceptionTableEntry(start_pc=0, end_pc=10, handler_pc=20, catch_type=0),   # Valid 1
            ExceptionTableEntry(start_pc=0, end_pc=10, handler_pc=20, catch_type=0),   # Duplicate of Valid 1
        ]
    )

    fixed = sanitize_code_exceptions(code_attr)
    assert fixed == 5
    assert len(code_attr.exception_table) == 1
    assert code_attr.exception_table[0].start_pc == 0
    assert code_attr.exception_table[0].end_pc == 10
    assert code_attr.exception_table[0].handler_pc == 20


def test_sanitize_exceptions_self_catching_loop():
    # Handler PC falls within [start_pc, end_pc)
    code_bytes = bytes([Opcode.NOP] * 30)
    code_attr = CodeAttribute(
        attribute_name_index=1,
        max_stack=2,
        max_locals=2,
        code=code_bytes,
        exception_table=[
            # Handler inside protected region: 0 <= 10 < 20
            ExceptionTableEntry(start_pc=0, end_pc=20, handler_pc=10, catch_type=0),
        ]
    )

    fixed = sanitize_code_exceptions(code_attr)
    assert fixed >= 1
    assert len(code_attr.exception_table) == 1
    # End PC should be truncated to before handler_pc (10)
    entry = code_attr.exception_table[0]
    assert entry.start_pc == 0
    assert entry.end_pc == 10
    assert entry.handler_pc == 10


# --- F8: StackMapTable Frame Sanitizer Tests ---

def test_stack_map_validation():
    # Valid StackMapTable payload: 1 same_frame (tag 10)
    valid_payload = struct.pack(">HB", 1, 10)
    assert validate_stack_map_table(valid_payload, 50, None)

    # Invalid: frame offset delta goes past code_len
    oob_payload = struct.pack(">HB", 1, 60)
    assert not validate_stack_map_table(oob_payload, 50, None)

    # Invalid: reserved frame tag 200
    bad_tag_payload = struct.pack(">HB", 1, 200)
    assert not validate_stack_map_table(bad_tag_payload, 50, None)

    # Truncated payload
    assert not validate_stack_map_table(b"\x00", 50, None)


def test_stack_map_stripping_on_corrupt():
    cf = create_minimal_class()
    cp = cf.constant_pool
    sm_name_idx = cp.add_utf8("StackMapTable")

    # Corrupt stack map attribute on method
    corrupt_data = struct.pack(">HB", 1, 220)  # reserved tag 220
    code = CodeAttribute(
        attribute_name_index=cp.add_utf8("Code"),
        max_stack=1,
        max_locals=1,
        code=bytes([Opcode.RETURN]),
        attributes=[StackMapTableAttribute(sm_name_idx, corrupt_data)]
    )
    method = MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("test"), cp.add_utf8("()V"), [code])
    cf.methods.append(method)

    stripped = sanitize_class_stack_maps(cf)
    assert stripped == 1
    assert code.get_attribute("StackMapTable", cp) is None


def test_stack_map_stripping_on_legacy_java():
    cf = create_minimal_class()
    cf.major_version = 50  # Java 6
    cp = cf.constant_pool
    sm_name_idx = cp.add_utf8("StackMapTable")

    valid_payload = struct.pack(">HB", 1, 0)
    code = CodeAttribute(
        attribute_name_index=cp.add_utf8("Code"),
        max_stack=1,
        max_locals=1,
        code=bytes([Opcode.RETURN]),
        attributes=[StackMapTableAttribute(sm_name_idx, valid_payload)]
    )
    method = MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("test"), cp.add_utf8("()V"), [code])
    cf.methods.append(method)

    stripped = sanitize_class_stack_maps(cf)
    assert stripped == 1
    assert code.get_attribute("StackMapTable", cp) is None


# --- F11: Synthetic Member Deduplication Tests ---

def test_deduplicate_synthetic_fields():
    cf = create_minimal_class()
    cp = cf.constant_pool

    name_idx = cp.add_utf8("value")
    desc_i = cp.add_utf8("I")
    desc_s = cp.add_utf8("Ljava/lang/String;")

    # Two fields with same name "value", one synthetic
    f1 = FieldInfo(AccessFlags.PUBLIC, name_idx, desc_i, [])
    f2 = FieldInfo(AccessFlags.PUBLIC | AccessFlags.SYNTHETIC, name_idx, desc_s, [])
    cf.fields.extend([f1, f2])

    renamed = deduplicate_fields(cf)
    assert renamed == 1
    assert f1.name(cp) == "value"
    assert f2.name(cp) == "value$dup$1"


def test_deduplicate_synthetic_methods():
    cf = create_minimal_class()
    cp = cf.constant_pool

    name_idx = cp.add_utf8("execute")
    desc_idx = cp.add_utf8("()V")

    # Regular method with code
    m1 = MethodInfo(AccessFlags.PUBLIC, name_idx, desc_idx, [
        CodeAttribute(cp.add_utf8("Code"), 1, 1, bytes([Opcode.RETURN]), [], [])
    ])
    # Duplicate synthetic method
    m2 = MethodInfo(AccessFlags.PUBLIC | AccessFlags.SYNTHETIC, name_idx, desc_idx, [])
    cf.methods.extend([m1, m2])

    pruned = deduplicate_methods(cf)
    assert pruned == 1
    assert len(cf.methods) == 1
    assert cf.methods[0] is m1


def test_break_recursive_bridge_method():
    cf = create_minimal_class("com/example/BridgeTest")
    cp = cf.constant_pool

    name_idx = cp.add_utf8("bridgeCall")
    desc_idx = cp.add_utf8("()V")

    # Create self-recursive invokevirtual call in bridge method
    mref_idx = cp.add_member_ref(ConstantTag.METHODREF, "com/example/BridgeTest", "bridgeCall", "()V")
    # Bytecode: invokevirtual mref_idx, return
    code_bytes = bytes([Opcode.INVOKEVIRTUAL]) + struct.pack(">H", mref_idx) + bytes([Opcode.RETURN])

    code_attr = CodeAttribute(cp.add_utf8("Code"), 1, 1, code_bytes, [], [])
    bridge_m = MethodInfo(AccessFlags.PUBLIC | AccessFlags.BRIDGE, name_idx, desc_idx, [code_attr])
    cf.methods.append(bridge_m)

    cycles_broken = break_recursive_bridge_methods(cf)
    assert cycles_broken == 1
    # Bytecode replaced with clean return
    assert code_attr.code == bytes([Opcode.RETURN])


def test_break_recursive_bridge_method_array_and_object_returns():
    """Verify recursive bridge method cycle breaking produces type-correct returns for arrays, objects, and primitives."""
    cf = create_minimal_class("com/example/BridgeReturnTypes")
    cp = cf.constant_pool

    test_cases = [
        ("bridgeArr", "()[I", bytes([Opcode.ACONST_NULL, Opcode.ARETURN])),
        ("bridgeMultiArr", "()[[Ljava/lang/String;", bytes([Opcode.ACONST_NULL, Opcode.ARETURN])),
        ("bridgeObj", "()Ljava/lang/String;", bytes([Opcode.ACONST_NULL, Opcode.ARETURN])),
        ("bridgeLong", "()J", bytes([Opcode.LCONST_0, Opcode.LRETURN])),
        ("bridgeDouble", "()D", bytes([Opcode.DCONST_0, Opcode.DRETURN])),
        ("bridgeFloat", "()F", bytes([Opcode.FCONST_0, Opcode.FRETURN])),
        ("bridgeInt", "()I", bytes([Opcode.ICONST_0, Opcode.IRETURN])),
    ]

    for m_name, m_desc, expected_code in test_cases:
        mref = cp.add_member_ref(ConstantTag.METHODREF, "com/example/BridgeReturnTypes", m_name, m_desc)
        code = bytes([Opcode.INVOKEVIRTUAL]) + struct.pack(">H", mref) + bytes([Opcode.RETURN])
        code_attr = CodeAttribute(cp.add_utf8("Code"), 1, 1, code, [], [])
        m = MethodInfo(
            AccessFlags.PUBLIC | AccessFlags.BRIDGE,
            cp.add_utf8(m_name),
            cp.add_utf8(m_desc),
            [code_attr]
        )
        cf.methods.append(m)

    broken = break_recursive_bridge_methods(cf)
    assert broken == len(test_cases)

    for i, (_, _, expected_code) in enumerate(test_cases):
        code_attr = next(a for a in cf.methods[i].attributes if isinstance(a, CodeAttribute))
        assert code_attr.code == expected_code


# --- End-to-End BytecodeSanitizer Tests ---

def test_bytecode_sanitizer_end_to_end_archive():
    archive = JarArchive()
    cf = create_minimal_class("com/example/Broken")
    cp = cf.constant_pool

    # Corrupt signature
    sig_name_idx = cp.add_utf8("Signature")
    bad_sig_idx = cp.add_utf8("<Corrupt")
    cf.attributes.append(SignatureAttribute(sig_name_idx, bad_sig_idx))

    # Corrupt exception table
    code_attr = CodeAttribute(
        attribute_name_index=cp.add_utf8("Code"),
        max_stack=1, max_locals=1,
        code=bytes([Opcode.RETURN] * 10),
        exception_table=[ExceptionTableEntry(10, 5, 2, 0)]  # Inverted
    )
    cf.methods.append(MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("m"), cp.add_utf8("()V"), [code_attr]))

    archive.add_class(cf)

    sanitizer = BytecodeSanitizer()
    report = sanitizer.sanitize_archive(archive)

    assert report.total_classes == 1
    assert report.classes_modified == 1
    assert report.total_signatures_stripped == 1
    assert report.total_exceptions_fixed == 1

    # Verify structural integrity
    valid, issues = verify_sanitized_class_structural_integrity(cf)
    assert valid, f"Structural issues remaining: {issues}"

    # Verify binary serialization round-trip succeeds
    serialized = cf.to_bytes()
    reparsed = ClassFile.parse(serialized)
    assert reparsed.this_class_name == "com/example/Broken"
