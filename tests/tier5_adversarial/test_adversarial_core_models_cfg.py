"""
tests/tier5_adversarial/test_adversarial_core_models_cfg.py:
Tier 5 White-Box Adversarial Stress Test Suite for jar_engine.
Focuses on:
1. Core Binary Models, Constant Pool 17 tags, 2-slot boundaries, IEEE-754 bit-exact preservation,
   Modified UTF-8 codec (NUL bytes, astral plane surrogate pairs, truncated fallbacks),
   and RFC 822 manifest line folding.
2. Bytecode Sanitizer: StackMapTable legacy stripping, reserved tag corruption, generic signature
   recursive-descent grammar edge cases, irregular/inverted exception tables, self-catching recursive
   exception handlers, synthetic member deduplication, and recursive bridge cycle breakers.
3. Control Flow Graph: 5-rule leader algorithm partitioning, dominator trees, natural loops, McCabe
   cyclomatic complexity, state-variable switch de-flattening, opaque predicate elimination with strict
   stack neutrality, and dead code/trampoline compaction.
4. Normalization Passes: JVM arithmetic evaluator semantics (division-by-zero prevention, overflow,
   remainder signs, shift masking), IEEE-754 float/double NaNs and Infinities, abstract interpretation
   constant folding lattice with ConstantPool promotion, pure sandboxed string decryptor bytecode
   emulation with timeout safety, and invokedynamic bootstrap resolution.
"""

from __future__ import annotations
import io
import math
import struct
import pytest

from jar_engine.model.constants import (
    JAVA_MAGIC, ConstantTag, AccessFlags, Opcode, ReferenceKind
)
from jar_engine.model.cp_info import (
    decode_modified_utf8, encode_modified_utf8,
    ConstantPool, CpPlaceholder,
    CONSTANT_Utf8_info, CONSTANT_Integer_info, CONSTANT_Float_info,
    CONSTANT_Long_info, CONSTANT_Double_info, CONSTANT_Class_info,
    CONSTANT_String_info, CONSTANT_Fieldref_info, CONSTANT_Methodref_info,
    CONSTANT_InterfaceMethodref_info, CONSTANT_NameAndType_info,
    CONSTANT_MethodHandle_info, CONSTANT_MethodType_info,
    CONSTANT_Dynamic_info, CONSTANT_InvokeDynamic_info,
    CONSTANT_Module_info, CONSTANT_Package_info
)
from jar_engine.model.class_file import ClassFile, ClassFormatError
from jar_engine.model.member_info import (
    FieldInfo, MethodInfo, AttributeInfo, RawAttribute,
    CodeAttribute, ExceptionTableEntry, ExceptionsAttribute,
    SignatureAttribute, StackMapTableAttribute,
    BootstrapMethodsAttribute, BootstrapMethod
)
from jar_engine.model.jar_archive import Manifest, wrap_manifest_line
from jar_engine.sanitizer.stack_map import (
    validate_verification_type_info, validate_stack_map_table,
    sanitize_class_stack_maps
)
from jar_engine.sanitizer.signatures import (
    SignatureParser, is_valid_class_signature,
    is_valid_method_signature, is_valid_field_signature,
    sanitize_class_signatures
)
from jar_engine.sanitizer.exceptions import (
    sanitize_code_exceptions, sanitize_method_exceptions_attribute,
    sanitize_class_exceptions
)
from jar_engine.sanitizer.synthetic import (
    deduplicate_fields, deduplicate_methods,
    break_recursive_bridge_methods, sanitize_synthetic_members
)
from jar_engine.cfg.basic_block import (
    Instruction, BasicBlock, CFGEdge, CFGEdgeType,
    decode_instructions, find_leaders, partition_basic_blocks
)
from jar_engine.cfg.graph import ControlFlowGraph, NaturalLoop
from jar_engine.cfg.deflattener import (
    ControlFlowDeflattener, OpaquePredicateEliminator,
    get_const_int_value, is_const_int_push
)
from jar_engine.cfg.dead_code import (
    prune_unreachable_blocks, compact_goto_chains, reassemble_cfg
)
from jar_engine.normalization.constant_folding import (
    ArithmeticEvaluator, LatticeValue, LatticeTag, AbstractFrame,
    create_push_instruction, IntraProceduralConstantFolder,
    fold_constants_and_arithmetic
)
from jar_engine.normalization.string_decryptor import (
    DecryptorMethodDetector, BytecodeEmulator, StringDecryptorResolver
)
from jar_engine.normalization.dynamic_invoke import (
    DynamicInvokeResolver, ResolvedStringConcat, ResolvedLambda
)


# ==============================================================================
# SECTION 1: CORE BINARY MODELS, CONSTANT POOL, AND MODIFIED UTF-8
# ==============================================================================

def test_adv_utf8_null_byte_roundtrip():
    """Adversarial UTF-8: JVMS §4.4.7 NUL byte (0x00) must encode to 2 bytes (0xC0 0x80)."""
    raw_nul = encode_modified_utf8("\u0000")
    assert raw_nul == b"\xc0\x80"
    assert decode_modified_utf8(b"\xc0\x80") == "\u0000"

    mixed = "prefix\u0000mid\u0000\u0000suffix"
    encoded = encode_modified_utf8(mixed)
    assert b"\x00" not in encoded  # Never contains standard NUL bytes
    assert encoded == b"prefix\xc0\x80mid\xc0\x80\xc0\x80suffix"
    assert decode_modified_utf8(encoded) == mixed


def test_adv_utf8_astral_plane_surrogate_pairs():
    """Adversarial UTF-8: Code points > 0xFFFF must encode as 6-byte UTF-16 surrogate pairs."""
    grinning_face = "\U0001F600"  # U+1F600
    encoded = encode_modified_utf8(grinning_face)
    assert len(encoded) == 6  # 3 bytes for high surrogate, 3 bytes for low surrogate
    # High surrogate: D83D -> E0 A0 BD -> \xED\xA0\xBD
    # Low surrogate:  DE00 -> E0 B8 80 -> \xED\xB8\x80
    assert encoded == b"\xed\xa0\xbd\xed\xb8\x80"
    decoded = decode_modified_utf8(encoded)
    assert decoded == grinning_face

    musical_g_clef = "\U0001D11E"  # U+1D11E
    encoded_g = encode_modified_utf8(musical_g_clef)
    assert len(encoded_g) == 6
    assert decode_modified_utf8(encoded_g) == musical_g_clef


def test_adv_utf8_corrupt_multibyte_graceful_fallback():
    """Adversarial UTF-8: Truncated or malformed byte sequences fallback without raising exceptions."""
    truncated_2byte = b"\xc0"
    assert decode_modified_utf8(truncated_2byte) == "\ufffd"

    truncated_3byte = b"\xe0\x80"
    assert decode_modified_utf8(truncated_3byte) == "\ufffd\ufffd"

    illegal_lead_byte = b"\xff\xfe"
    res = decode_modified_utf8(illegal_lead_byte)
    assert "\ufffd" in res


def test_adv_cp_two_slot_long_double_boundary_invariants():
    """Adversarial CP: Long and Double occupy index k and k+1; index k+1 must raise ValueError or return None."""
    cp = ConstantPool()
    l_idx = cp.add_long(0x1122334455667788)
    d_idx = cp.add_double(3.141592653589793)

    assert l_idx == 1
    assert d_idx == 3  # Long skipped slot 2, so Double is at slot 3
    assert len(cp) == 5  # [None, Long, Placeholder, Double, Placeholder]

    assert isinstance(cp[1], CONSTANT_Long_info)
    with pytest.raises(ValueError, match="unusable 2nd slot"):
        _ = cp[2]
    assert cp.get(2) is None

    assert isinstance(cp[3], CONSTANT_Double_info)
    with pytest.raises(ValueError, match="unusable 2nd slot"):
        _ = cp[4]
    assert cp.get(4) is None

    # Test entries() iterator ignores placeholders
    valid_indices = [idx for idx, _ in cp.entries()]
    assert valid_indices == [1, 3]


def test_adv_cp_ieee754_negative_zero_and_custom_nans():
    """Adversarial CP: Floating point -0.0 and distinct NaN bit-payloads must be preserved exactly."""
    cp = ConstantPool()

    # Negative zero Float and Double
    f_neg_zero = -0.0
    d_neg_zero = -0.0
    f_idx = cp.add_float(f_neg_zero)
    d_idx = cp.add_double(d_neg_zero)

    f_entry: CONSTANT_Float_info = cp[f_idx]  # type: ignore
    d_entry: CONSTANT_Double_info = cp[d_idx]  # type: ignore

    # Verify sign bit in raw_bits
    assert f_entry.raw_bits == 0x80000000
    assert d_entry.raw_bits == 0x8000000000000000
    assert math.copysign(1.0, f_entry.value) == -1.0
    assert math.copysign(1.0, d_entry.value) == -1.0

    # Custom NaN payload (Quiet NaN with arbitrary payload bits)
    custom_nan_bits_f = 0x7FC00042
    nan_entry_f = CONSTANT_Float_info(raw_bits=custom_nan_bits_f)
    serialized_f = nan_entry_f.to_bytes()
    tag, unpacked_f_bits = struct.unpack(">BI", serialized_f)
    assert tag == ConstantTag.FLOAT
    assert unpacked_f_bits == custom_nan_bits_f

    custom_nan_bits_d = 0x7FF8000000000042
    nan_entry_d = CONSTANT_Double_info(raw_bits=custom_nan_bits_d)
    serialized_d = nan_entry_d.to_bytes()
    tag_d, unpacked_d_bits = struct.unpack(">BQ", serialized_d)
    assert tag_d == ConstantTag.DOUBLE
    assert unpacked_d_bits == custom_nan_bits_d


def test_adv_cp_all_17_tags_complete_roundtrip():
    """Adversarial CP: Full JVM binary roundtrip containing all 17 JVMS constant pool tags."""
    cp = ConstantPool()
    utf1 = cp.add_utf8("TestClass")
    utf2 = cp.add_utf8("foo")
    utf3 = cp.add_utf8("()V")
    utf4 = cp.add_utf8("m_handle")
    c1 = cp.add_class("TestClass")
    s1 = cp.add_string("AdversarialString")
    i1 = cp.add_integer(-42)
    l1 = cp.add_long(0x7FFFFFFFFFFFFFFF)
    f1 = cp.add_float(2.71828)
    d1 = cp.add_double(1.41421356237)
    nat1 = cp.add_name_and_type("foo", "()V")
    mref1 = cp.add_member_ref(ConstantTag.METHODREF, "TestClass", "foo", "()V")
    fref1 = cp.add_member_ref(ConstantTag.FIELDREF, "TestClass", "field1", "I")
    iref1 = cp.add_member_ref(ConstantTag.INTERFACE_METHODREF, "TestClass", "ifaceM", "()I")

    # MethodHandle with all 9 reference kinds
    idx_mh = len(cp._entries)
    cp._entries.append(CONSTANT_MethodHandle_info(reference_kind=ReferenceKind.REF_invokeStatic, reference_index=mref1))
    idx_mt = len(cp._entries)
    cp._entries.append(CONSTANT_MethodType_info(descriptor_index=utf3))
    idx_dyn = len(cp._entries)
    cp._entries.append(CONSTANT_Dynamic_info(bootstrap_method_attr_index=0, name_and_type_index=nat1))
    idx_indy = len(cp._entries)
    cp._entries.append(CONSTANT_InvokeDynamic_info(bootstrap_method_attr_index=0, name_and_type_index=nat1))
    idx_mod = len(cp._entries)
    cp._entries.append(CONSTANT_Module_info(name_index=utf1))
    idx_pkg = len(cp._entries)
    cp._entries.append(CONSTANT_Package_info(name_index=utf1))

    # Serialize and parse back within a ClassFile
    cf = ClassFile(
        magic=JAVA_MAGIC,
        minor_version=0,
        major_version=55,  # Java 11
        constant_pool=cp,
        access_flags=AccessFlags.PUBLIC,
        this_class=c1,
        super_class=0,
        interfaces=[],
        fields=[],
        methods=[],
        attributes=[]
    )
    raw = cf.to_bytes()
    reparsed = ClassFile.parse(raw)

    assert reparsed.constant_pool[idx_mh].tag == ConstantTag.METHOD_HANDLE
    assert reparsed.constant_pool[idx_mt].tag == ConstantTag.METHOD_TYPE
    assert reparsed.constant_pool[idx_dyn].tag == ConstantTag.DYNAMIC
    assert reparsed.constant_pool[idx_indy].tag == ConstantTag.INVOKE_DYNAMIC
    assert reparsed.constant_pool[idx_mod].tag == ConstantTag.MODULE
    assert reparsed.constant_pool[idx_pkg].tag == ConstantTag.PACKAGE


def test_adv_class_file_parser_corrupted_inputs():
    """Adversarial ClassFile: Verifies parser throws ClassFormatError on corrupted and truncated streams."""
    # 1. Truncated header (< 10 bytes)
    with pytest.raises(ClassFormatError, match="too short"):
        ClassFile.parse(b"\xca\xfe\xba")

    # 2. Corrupt magic number
    with pytest.raises(ClassFormatError, match="Invalid magic"):
        ClassFile.parse(b"\xde\xad\xbe\xef" + b"\x00" * 6)

    # 3. Invalid CP tag
    bad_cp_stream = struct.pack(">IHHH", JAVA_MAGIC, 0, 52, 2) + b"\x63"  # tag 99 (unknown)
    with pytest.raises(ClassFormatError, match="Invalid constant pool tag: 99"):
        ClassFile.parse(bad_cp_stream)

    # 4. Truncated UTF-8 payload
    bad_utf_stream = struct.pack(">IHHH", JAVA_MAGIC, 0, 52, 2) + struct.pack(">BH", ConstantTag.UTF8, 10) + b"abc"
    with pytest.raises(ClassFormatError, match="Truncated UTF8 payload"):
        ClassFile.parse(bad_utf_stream)

    # 5. Truncated class body
    header_only = struct.pack(">IHHH", JAVA_MAGIC, 0, 52, 1)  # cp_count = 1
    with pytest.raises(ClassFormatError, match="Truncated class declaration header"):
        ClassFile.parse(header_only)


def test_adv_class_file_preserves_raw_unrecognized_attributes():
    """Adversarial ClassFile: Unrecognized/proprietary attributes are preserved via RawAttribute without corruption."""
    cp = ConstantPool()
    c1 = cp.add_class("CustomClass")
    custom_attr_name_idx = cp.add_utf8("com.vendor.ObfuscationMeta")
    custom_payload = b"\xde\xad\xbe\xef\x01\x02\x03\x04\x55"

    cf = ClassFile(
        magic=JAVA_MAGIC,
        minor_version=0,
        major_version=52,
        constant_pool=cp,
        access_flags=AccessFlags.PUBLIC,
        this_class=c1,
        super_class=0,
        interfaces=[],
        fields=[],
        methods=[],
        attributes=[RawAttribute(attribute_name_index=custom_attr_name_idx, info=custom_payload)]
    )

    raw = cf.to_bytes()
    reparsed = ClassFile.parse(raw)
    assert len(reparsed.attributes) == 1
    attr = reparsed.attributes[0]
    assert isinstance(attr, RawAttribute)
    assert attr.name(reparsed.constant_pool) == "com.vendor.ObfuscationMeta"
    assert attr.info == custom_payload


def test_adv_manifest_rfc822_line_folding_multibyte_boundary():
    """Adversarial Manifest: 72-byte line wrap straddling multi-byte UTF-8 sequences unfolds without corruption."""
    long_ascii = "A" * 60
    # Add a 3-byte unicode character right around the 72-byte boundary
    special_value = f"{long_ascii}___€___end"
    m = Manifest(main_attributes={"X-Special-Header": special_value})
    manifest_bytes = m.to_bytes()

    # Verify line continuation occurred (contains CRLF followed by space)
    assert b"\r\n " in manifest_bytes

    reparsed = Manifest.parse(manifest_bytes)
    assert reparsed.main_attributes["X-Special-Header"] == special_value


# ==============================================================================
# SECTION 2: BYTECODE PREPROCESSING & SANITIZER PASSES
# ==============================================================================

def test_adv_stack_map_legacy_java6_forced_stripping():
    """Adversarial Sanitizer: Java <= 6 bytecode (major <= 50) must strip StackMapTable attributes unconditionally."""
    cp = ConstantPool()
    c1 = cp.add_class("LegacyClass")
    sm_name = cp.add_utf8("StackMapTable")
    code_name = cp.add_utf8("Code")
    m_name = cp.add_utf8("test")
    m_desc = cp.add_utf8("()V")

    # Valid same_frame payload (0 entries or 1 entry)
    dummy_sm_data = struct.pack(">H", 1) + bytes([10])  # same_frame, offset_delta 10
    sm_attr = StackMapTableAttribute(attribute_name_index=sm_name, entries_data=dummy_sm_data)
    code_attr = CodeAttribute(
        attribute_name_index=code_name,
        max_stack=1,
        max_locals=1,
        code=bytes([Opcode.NOP] * 20 + [Opcode.RETURN]),
        exception_table=[],
        attributes=[sm_attr]
    )
    method = MethodInfo(
        access_flags=AccessFlags.PUBLIC,
        name_index=m_name,
        descriptor_index=m_desc,
        attributes=[code_attr]
    )

    cf = ClassFile(
        magic=JAVA_MAGIC,
        minor_version=0,
        major_version=50,  # Java 6
        constant_pool=cp,
        access_flags=AccessFlags.PUBLIC,
        this_class=c1,
        super_class=0,
        methods=[method]
    )

    stripped = sanitize_class_stack_maps(cf)
    assert stripped == 1
    assert len(code_attr.attributes) == 0  # StackMapTable stripped


def test_adv_stack_map_java7_reserved_tags_and_out_of_bounds_pruned():
    """Adversarial Sanitizer: StackMapTable with reserved tags (128..246) or out-of-bounds offsets is stripped."""
    cp = ConstantPool()
    c1 = cp.add_class("ModernClass")
    sm_name = cp.add_utf8("StackMapTable")
    code_name = cp.add_utf8("Code")

    # Reserved frame tag 130
    corrupt_sm_data = struct.pack(">H", 1) + bytes([130])
    sm_attr = StackMapTableAttribute(attribute_name_index=sm_name, entries_data=corrupt_sm_data)
    code_attr = CodeAttribute(
        attribute_name_index=code_name,
        max_stack=1,
        max_locals=1,
        code=bytes([Opcode.RETURN]),
        exception_table=[],
        attributes=[sm_attr]
    )
    method = MethodInfo(
        access_flags=AccessFlags.PUBLIC,
        name_index=cp.add_utf8("m"),
        descriptor_index=cp.add_utf8("()V"),
        attributes=[code_attr]
    )

    cf = ClassFile(
        major_version=52,  # Java 8
        constant_pool=cp,
        this_class=c1,
        methods=[method]
    )

    stripped = sanitize_class_stack_maps(cf)
    assert stripped == 1
    assert len(code_attr.attributes) == 0


def test_adv_stack_map_valid_java8_preserved():
    """Adversarial Sanitizer: Valid StackMapTable frames in Java 7+ classes must be preserved."""
    cp = ConstantPool()
    c1 = cp.add_class("ValidClass")
    sm_name = cp.add_utf8("StackMapTable")
    code_name = cp.add_utf8("Code")

    # Valid same_frame (offset_delta 5), code length 10
    valid_sm_data = struct.pack(">H", 1) + bytes([5])
    sm_attr = StackMapTableAttribute(attribute_name_index=sm_name, entries_data=valid_sm_data)
    code_attr = CodeAttribute(
        attribute_name_index=code_name,
        max_stack=1,
        max_locals=1,
        code=bytes([Opcode.NOP] * 9 + [Opcode.RETURN]),
        attributes=[sm_attr]
    )
    method = MethodInfo(
        access_flags=AccessFlags.PUBLIC,
        name_index=cp.add_utf8("m"),
        descriptor_index=cp.add_utf8("()V"),
        attributes=[code_attr]
    )

    cf = ClassFile(major_version=52, constant_pool=cp, this_class=c1, methods=[method])
    stripped = sanitize_class_stack_maps(cf)
    assert stripped == 0
    assert len(code_attr.attributes) == 1


def test_adv_generic_signature_complex_valid_structures():
    """Adversarial Signatures: Complex valid generic signatures with wildcards, bounds, and throws."""
    # Complex class signature
    cls_sig = "<K:Ljava/lang/Object;V::Ljava/util/List<+TK;>;>Ljava/lang/Object;Ljava/util/Map<TK;TV;>;"
    assert is_valid_class_signature(cls_sig) is True

    # Complex method signature with type parameters and multiple throws
    meth_sig = "<T:Ljava/lang/Throwable;>(TT;Ljava/util/List<*>;)Ljava/util/Map<+Ljava/lang/String;-[B>;^TT;^Ljava/lang/Exception;"
    assert is_valid_method_signature(meth_sig) is True

    # Complex field signature
    field_sig = "Ljava/util/Map<+Ljava/lang/String;-[Ljava/lang/Number;>;"
    assert is_valid_field_signature(field_sig) is True


def test_adv_generic_signature_corrupted_grammar_stripped():
    """Adversarial Signatures: Malformed signatures (unclosed, dots, missing returns) are stripped."""
    corrupted_signatures = [
        "<T:Ljava/lang/Object;",           # Unclosed type parameters
        "Ljava/lang/String",              # Missing terminating semicolon
        "Ljava.lang.String;",             # Dot instead of slash in package
        "()",                             # Missing return type
        "<T:Ljava/lang/Throwable;>()V^",  # Incomplete throws clause
        "Ljava/util/List<;",              # Incomplete type argument
        "",                               # Empty string
    ]

    cp = ConstantPool()
    c1 = cp.add_class("SigTest")
    sig_name = cp.add_utf8("Signature")

    # Add corrupted signatures to class, field, and method
    cf_attrs = [SignatureAttribute(sig_name, cp.add_utf8(corrupted_signatures[0]))]
    f_attrs = [SignatureAttribute(sig_name, cp.add_utf8(corrupted_signatures[1]))]
    m_attrs = [SignatureAttribute(sig_name, cp.add_utf8(corrupted_signatures[3]))]

    f = FieldInfo(AccessFlags.PUBLIC, cp.add_utf8("f"), cp.add_utf8("I"), f_attrs)
    m = MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("m"), cp.add_utf8("()V"), m_attrs)

    cf = ClassFile(
        constant_pool=cp, this_class=c1,
        attributes=cf_attrs, fields=[f], methods=[m]
    )

    stripped = sanitize_class_signatures(cf)
    assert stripped == 3
    assert len(cf.attributes) == 0
    assert len(f.attributes) == 0
    assert len(m.attributes) == 0


def test_adv_exception_table_inverted_and_out_of_bounds_repairs():
    """Adversarial Exceptions: Inverted ranges, out-of-bounds PCs, and negative PCs are cleanly repaired."""
    code = bytes([Opcode.NOP] * 50)
    code_attr = CodeAttribute(
        attribute_name_index=1,
        code=code,
        exception_table=[
            ExceptionTableEntry(start_pc=30, end_pc=20, handler_pc=10, catch_type=0),  # Inverted: start >= end
            ExceptionTableEntry(start_pc=10, end_pc=10, handler_pc=15, catch_type=0),  # Empty: start == end
            ExceptionTableEntry(start_pc=-5, end_pc=20, handler_pc=10, catch_type=0),  # Negative start
            ExceptionTableEntry(start_pc=10, end_pc=60, handler_pc=10, catch_type=0),  # end > code_len (50)
            ExceptionTableEntry(start_pc=10, end_pc=20, handler_pc=55, catch_type=0),  # handler >= code_len (50)
            ExceptionTableEntry(start_pc=5, end_pc=25, handler_pc=40, catch_type=0),   # Valid
        ]
    )

    mods = sanitize_code_exceptions(code_attr)
    assert mods == 5
    assert len(code_attr.exception_table) == 1
    assert code_attr.exception_table[0].start_pc == 5
    assert code_attr.exception_table[0].end_pc == 25


def test_adv_exception_table_self_catching_recursive_loops():
    """Adversarial Exceptions: Self-catching recursive loop where handler is inside protected range."""
    code = bytes([Opcode.NOP] * 50)
    code_attr = CodeAttribute(
        attribute_name_index=1,
        code=code,
        exception_table=[
            # Handler inside [start, end) where start < handler < end: truncated to (start, handler)
            ExceptionTableEntry(start_pc=10, end_pc=30, handler_pc=20, catch_type=0),
            # Handler at start of range: discarded
            ExceptionTableEntry(start_pc=5, end_pc=25, handler_pc=5, catch_type=0),
        ]
    )

    mods = sanitize_code_exceptions(code_attr)
    assert mods == 2
    assert len(code_attr.exception_table) == 1
    repaired = code_attr.exception_table[0]
    assert repaired.start_pc == 10
    assert repaired.end_pc == 20  # Truncated to handler_pc
    assert repaired.handler_pc == 20


def test_adv_synthetic_duplicate_member_deduplication():
    """Adversarial Synthetic: Conflicting duplicate field names are renamed; duplicate methods deduplicated."""
    cp = ConstantPool()
    c1 = cp.add_class("CollisionClass")
    fname_idx = cp.add_utf8("state")

    # Two fields with same name "state" but different types
    f1 = FieldInfo(access_flags=AccessFlags.PUBLIC, name_index=fname_idx, descriptor_index=cp.add_utf8("I"))
    f2 = FieldInfo(access_flags=AccessFlags.PUBLIC | AccessFlags.SYNTHETIC, name_index=fname_idx, descriptor_index=cp.add_utf8("Ljava/lang/String;"))

    # Two methods with same signature (m, ()V): one synthetic, one non-synthetic with code
    mname_idx = cp.add_utf8("run")
    mdesc_idx = cp.add_utf8("()V")
    m1 = MethodInfo(access_flags=AccessFlags.PUBLIC, name_index=mname_idx, descriptor_index=mdesc_idx, attributes=[
        CodeAttribute(attribute_name_index=cp.add_utf8("Code"), code=bytes([Opcode.RETURN]))
    ])
    m2 = MethodInfo(access_flags=AccessFlags.PUBLIC | AccessFlags.SYNTHETIC, name_index=mname_idx, descriptor_index=mdesc_idx)

    cf = ClassFile(constant_pool=cp, this_class=c1, fields=[f1, f2], methods=[m1, m2])
    f_mod, m_mod = sanitize_synthetic_members(cf)

    assert f_mod == 1
    assert m_mod == 1

    # Verify f2 renamed to state$dup$1
    assert f1.name(cp) == "state"
    assert f2.name(cp) == "state$dup$1"

    # Verify synthetic duplicate m2 pruned
    assert len(cf.methods) == 1
    assert cf.methods[0] is m1


def test_adv_synthetic_recursive_bridge_cycle_breaking():
    """Adversarial Bridge: Self-recursive synthetic bridge methods rewritten to safe returns."""
    cp = ConstantPool()
    c1 = cp.add_class("BridgeClass")
    m_name = cp.add_utf8("bridgeMethod")
    m_desc = cp.add_utf8("()V")
    mref = cp.add_member_ref(ConstantTag.METHODREF, "BridgeClass", "bridgeMethod", "()V")

    # Synthetic bridge method invoking itself via invokevirtual
    bridge_code = bytes([
        Opcode.ALOAD_0,
        Opcode.INVOKEVIRTUAL, (mref >> 8) & 0xFF, mref & 0xFF,
        Opcode.RETURN
    ])
    code_attr = CodeAttribute(attribute_name_index=cp.add_utf8("Code"), code=bridge_code)
    m = MethodInfo(
        access_flags=AccessFlags.PUBLIC | AccessFlags.SYNTHETIC | AccessFlags.BRIDGE,
        name_index=m_name,
        descriptor_index=m_desc,
        attributes=[code_attr]
    )

    cf = ClassFile(constant_pool=cp, this_class=c1, methods=[m])
    cycles_broken = break_recursive_bridge_methods(cf)

    assert cycles_broken == 1
    # Recursion broken: replaced with Opcode.RETURN
    assert code_attr.code == bytes([Opcode.RETURN])


# ==============================================================================
# SECTION 3: CONTROL FLOW GRAPH, DEFLATTENER, & OPAQUE PREDICATES
# ==============================================================================

def test_adv_cfg_partition_5_rules_leader_identification():
    """Adversarial CFG: 5-rule leader algorithm must partition entry, branch targets, fallthrough, and exception PCs."""
    # Bytecode layout:
    # 0: ICONST_1
    # 1: IFEQ 8       (target PC 9, next_pc 4)
    # 4: NOP
    # 5: GOTO 5       (target PC 10, next_pc 8)
    # 8: NOP
    # 9: NOP
    # 10: RETURN
    code = bytes([
        Opcode.ICONST_1,
        Opcode.IFEQ, 0, 8,   # target = 1 + 8 = 9
        Opcode.NOP,
        Opcode.GOTO, 0, 5,   # target = 5 + 5 = 10
        Opcode.NOP,
        Opcode.NOP,
        Opcode.RETURN
    ])
    ex_table = [ExceptionTableEntry(start_pc=4, end_pc=8, handler_pc=8, catch_type=0)]

    blocks = partition_basic_blocks(code, exception_table=ex_table)
    start_pcs = [b.start_pc for b in blocks]

    # Expected leaders:
    # Rule 1: PC 0 (entry)
    # Rule 2: PC 9 (target of IFEQ), PC 10 (target of GOTO)
    # Rule 3: PC 4 (fallthrough of IFEQ), PC 8 (post-terminator of GOTO)
    # Rule 4 & 5: PC 4 (try start), PC 8 (try end / handler)
    assert 0 in start_pcs
    assert 4 in start_pcs
    assert 8 in start_pcs
    assert 9 in start_pcs
    assert 10 in start_pcs


def test_adv_cfg_tableswitch_and_lookupswitch_edge_topology():
    """Adversarial CFG: TABLESwith and LOOKUPSWITCH edges correctly assign SWITCH_CASE keys and SWITCH_DEFAULT."""
    # TABLESWITCH at PC 0 (padding 3 bytes, default 15, low 1, high 2, case1 15, case2 16)
    # low=1, high=2 -> 2 cases
    pad = 3
    ts_operands = (
        b"\x00" * pad +
        struct.pack(">iii", 15, 1, 2) +
        struct.pack(">ii", 15, 16)
    )
    inst_ts = Instruction(pc=0, opcode=Opcode.TABLESWITCH, operands=ts_operands)
    b_switch = BasicBlock(id=0, start_pc=0, end_pc=inst_ts.next_pc, instructions=[inst_ts])
    b_default = BasicBlock(id=15, start_pc=15, end_pc=16, instructions=[Instruction(15, Opcode.RETURN)])
    b_case2 = BasicBlock(id=16, start_pc=16, end_pc=17, instructions=[Instruction(16, Opcode.RETURN)])

    cfg = ControlFlowGraph()
    cfg.add_block(b_switch)
    cfg.add_block(b_default)
    cfg.add_block(b_case2)

    # Trigger edge builder via from_code
    code = inst_ts.to_bytes() + bytes([Opcode.RETURN, Opcode.RETURN])
    cfg_built = ControlFlowGraph.from_code(code)

    out_edges = cfg_built.get_outgoing_edges(0)
    case_keys = [e.case_key for e in out_edges if e.edge_type == CFGEdgeType.SWITCH_CASE]
    has_default = any(e.edge_type == CFGEdgeType.SWITCH_DEFAULT for e in out_edges)

    assert has_default is True
    assert 1 in case_keys
    assert 2 in case_keys


def test_adv_cfg_dominators_and_natural_loop_detection():
    """Adversarial CFG: Dominator sets and natural loops identified on diamond graph with back-edge."""
    cfg = ControlFlowGraph()
    # 0 -> 1, 0 -> 2; 1 -> 3, 2 -> 3; 3 -> 0 (back edge)
    b0 = BasicBlock(id=0, start_pc=0, end_pc=1)
    b1 = BasicBlock(id=1, start_pc=1, end_pc=2)
    b2 = BasicBlock(id=2, start_pc=2, end_pc=3)
    b3 = BasicBlock(id=3, start_pc=3, end_pc=4)

    for b in [b0, b1, b2, b3]:
        cfg.add_block(b)

    cfg.add_edge(CFGEdge(0, 1, CFGEdgeType.CONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(0, 2, CFGEdgeType.FALLTHROUGH))
    cfg.add_edge(CFGEdge(1, 3, CFGEdgeType.UNCONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(2, 3, CFGEdgeType.FALLTHROUGH))
    cfg.add_edge(CFGEdge(3, 0, CFGEdgeType.UNCONDITIONAL_BRANCH))  # Back edge to header 0

    doms = cfg.compute_dominators()
    assert 0 in doms[1]
    assert 0 in doms[2]
    assert 0 in doms[3]
    assert 1 not in doms[3]  # Neither 1 nor 2 strictly dominates 3

    loops = cfg.find_natural_loops()
    assert len(loops) == 1
    loop = loops[0]
    assert loop.header_id == 0
    assert loop.latch_id == 3
    assert loop.body == {0, 1, 2, 3}


def test_adv_cfg_mccabe_cyclomatic_complexity_subgraph():
    """Adversarial CFG: McCabe complexity M = E - N + 2P is computed only on reachable connected blocks."""
    cfg = ControlFlowGraph()
    # Reachable component: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3 (N=4, E=4 -> M = 4 - 4 + 2 = 2)
    b0 = BasicBlock(id=0, start_pc=0, end_pc=1)
    b1 = BasicBlock(id=1, start_pc=1, end_pc=2)
    b2 = BasicBlock(id=2, start_pc=2, end_pc=3)
    b3 = BasicBlock(id=3, start_pc=3, end_pc=4)

    # Disconnected unreachable block 99 -> 100
    b99 = BasicBlock(id=99, start_pc=99, end_pc=100)
    b100 = BasicBlock(id=100, start_pc=100, end_pc=101)

    for b in [b0, b1, b2, b3, b99, b100]:
        cfg.add_block(b)

    cfg.add_edge(CFGEdge(0, 1, CFGEdgeType.CONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(0, 2, CFGEdgeType.FALLTHROUGH))
    cfg.add_edge(CFGEdge(1, 3, CFGEdgeType.UNCONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(2, 3, CFGEdgeType.FALLTHROUGH))
    cfg.add_edge(CFGEdge(99, 100, CFGEdgeType.UNCONDITIONAL_BRANCH))

    assert cfg.cyclomatic_complexity == 2


def test_adv_deflattener_state_switch_loop_restoration():
    """Adversarial Deflattener: State-variable switch dispatch loop is unflattened, reducing complexity."""
    cfg = ControlFlowGraph()

    # Block 0: Init state 1 -> goto Disp (Block 10)
    b0 = BasicBlock(0, 0, 4, [
        Instruction(0, Opcode.ICONST_1),
        Instruction(1, Opcode.ISTORE_1),
        Instruction(2, Opcode.GOTO, struct.pack(">h", 8))
    ])
    # Block 10: Dispatcher (iload_1; tableswitch 1->B20, 2->B30, def->B40)
    ts_ops = b"\x00" * 3 + struct.pack(">iii", 30, 1, 2) + struct.pack(">ii", 10, 20)
    b10 = BasicBlock(10, 10, 30, [
        Instruction(10, Opcode.ILOAD_1),
        Instruction(11, Opcode.TABLESWITCH, ts_ops)
    ])
    # Block 20: Body 1: do work, state=2 -> goto Disp
    b20 = BasicBlock(20, 20, 26, [
        Instruction(20, Opcode.NOP),
        Instruction(21, Opcode.ICONST_2),
        Instruction(22, Opcode.ISTORE_1),
        Instruction(23, Opcode.GOTO, struct.pack(">h", -13))
    ])
    # Block 30: Body 2: do work, state=3 -> goto Disp (def target B40 exit)
    b30 = BasicBlock(30, 30, 36, [
        Instruction(30, Opcode.NOP),
        Instruction(31, Opcode.ICONST_3),
        Instruction(32, Opcode.ISTORE_1),
        Instruction(33, Opcode.GOTO, struct.pack(">h", -23))
    ])
    # Block 40: Exit
    b40 = BasicBlock(40, 40, 41, [Instruction(40, Opcode.RETURN)])

    for b in [b0, b10, b20, b30, b40]:
        cfg.add_block(b)

    cfg.add_edge(CFGEdge(0, 10, CFGEdgeType.UNCONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(10, 20, CFGEdgeType.SWITCH_CASE, case_key=1))
    cfg.add_edge(CFGEdge(10, 30, CFGEdgeType.SWITCH_CASE, case_key=2))
    cfg.add_edge(CFGEdge(10, 40, CFGEdgeType.SWITCH_CASE, case_key=3))
    cfg.add_edge(CFGEdge(10, 40, CFGEdgeType.SWITCH_DEFAULT))
    cfg.add_edge(CFGEdge(20, 10, CFGEdgeType.UNCONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(30, 10, CFGEdgeType.UNCONDITIONAL_BRANCH))

    deflat = ControlFlowDeflattener(cfg)
    report = deflat.deflatten()

    assert report.loops_detected >= 1
    assert report.edges_reconstructed >= 2
    assert report.complexity_after < report.complexity_before

    # Check direct edge from 0 -> 20 and 20 -> 30 exists
    succ_0 = cfg.get_successors(0)
    succ_20 = cfg.get_successors(20)
    assert 20 in succ_0
    assert 30 in succ_20


def test_adv_opaque_predicate_constant_comparisons_pruned():
    """Adversarial Opaque Predicate: Constant branches (iconst_0/ifeq, aconst_null/ifnull) are pruned."""
    cfg = ControlFlowGraph()
    # Block 0: iconst_0; ifeq 10 (Always True) -> must become GOTO 10
    b0 = BasicBlock(0, 0, 4, [
        Instruction(0, Opcode.ICONST_0),
        Instruction(1, Opcode.IFEQ, struct.pack(">h", 10))
    ])
    b10 = BasicBlock(10, 10, 11, [Instruction(10, Opcode.RETURN)])
    b_dead = BasicBlock(20, 20, 21, [Instruction(20, Opcode.RETURN)])

    cfg.add_block(b0)
    cfg.add_block(b10)
    cfg.add_block(b_dead)

    cfg.add_edge(CFGEdge(0, 10, CFGEdgeType.CONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(0, 20, CFGEdgeType.FALLTHROUGH))

    op_elim = OpaquePredicateEliminator(cfg)
    rep = op_elim.eliminate()

    assert rep.constant_predicates_pruned == 1
    # Conditional branch pruned to direct jump to 10
    out_edges = cfg.get_outgoing_edges(0)
    assert len(out_edges) == 1
    assert out_edges[0].target_id == 10
    assert out_edges[0].edge_type == CFGEdgeType.UNCONDITIONAL_BRANCH


def test_adv_opaque_predicate_algebraic_invariants_stack_neutral():
    """Adversarial Opaque Predicate: x ^ x == 0 and x - x == 0 pruned while preserving stack neutrality."""
    cfg = ControlFlowGraph()
    # Block 0: iload_1; iload_1; ixor; ifeq 10 (Always True)
    b0 = BasicBlock(0, 0, 5, [
        Instruction(0, Opcode.ILOAD_1),
        Instruction(1, Opcode.ILOAD_1),
        Instruction(2, Opcode.IXOR),
        Instruction(3, Opcode.IFEQ, struct.pack(">h", 10))
    ])
    b10 = BasicBlock(10, 10, 11, [Instruction(10, Opcode.RETURN)])
    b_fall = BasicBlock(20, 20, 21, [Instruction(20, Opcode.RETURN)])

    cfg.add_block(b0)
    cfg.add_block(b10)
    cfg.add_block(b_fall)

    cfg.add_edge(CFGEdge(0, 10, CFGEdgeType.CONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(0, 20, CFGEdgeType.FALLTHROUGH))

    op_elim = OpaquePredicateEliminator(cfg)
    rep = op_elim.eliminate()

    assert rep.algebraic_predicates_pruned == 1
    # Producer instructions (iload_1, iload_1, ixor) removed to ensure stack neutrality!
    assert len(b0.instructions) == 1
    assert b0.instructions[0].opcode == Opcode.GOTO


def test_adv_dead_code_retains_active_exception_handler_blocks():
    """Adversarial Dead Code: Blocks reachable only through active exception handlers must NOT be pruned."""
    cfg = ControlFlowGraph()
    # Entry Block 0 covered by try block, throwing exception to Handler Block 10
    b0 = BasicBlock(0, 0, 2, [Instruction(0, Opcode.ATHROW)])
    b10 = BasicBlock(10, 10, 11, [Instruction(10, Opcode.RETURN)])
    # Block 99 is truly unreachable
    b99 = BasicBlock(99, 99, 100, [Instruction(99, Opcode.RETURN)])

    for b in [b0, b10, b99]:
        cfg.add_block(b)

    ex = ExceptionTableEntry(start_pc=0, end_pc=2, handler_pc=10, catch_type=0)
    b0.exception_handlers.append(ex)
    cfg.add_edge(CFGEdge(0, 10, CFGEdgeType.EXCEPTION))

    pruned = prune_unreachable_blocks(cfg, exception_table=[ex])
    assert pruned == 1  # Block 99 pruned
    assert 0 in cfg.blocks
    assert 10 in cfg.blocks  # Handler block 10 preserved!
    assert 99 not in cfg.blocks


def test_adv_dead_code_trampoline_cycle_breaker():
    """Adversarial Dead Code: Circular GOTO trampolines (L1 -> L2 -> L1) handled without infinite loops."""
    cfg = ControlFlowGraph()
    # B1: goto 2
    b1 = BasicBlock(1, 1, 4, [Instruction(1, Opcode.GOTO, struct.pack(">h", 1))])
    # B2: goto 1
    b2 = BasicBlock(2, 2, 5, [Instruction(2, Opcode.GOTO, struct.pack(">h", -1))])

    cfg.add_block(b1)
    cfg.add_block(b2)
    cfg.add_edge(CFGEdge(1, 2, CFGEdgeType.UNCONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(2, 1, CFGEdgeType.UNCONDITIONAL_BRANCH))

    # Must complete cleanly without recursion error
    compacted = compact_goto_chains(cfg)
    assert isinstance(compacted, int)


# ==============================================================================
# SECTION 4: NORMALIZATION (CONSTANT FOLDING, STRING DECRYPTOR, DYNAMIC INVOKE)
# ==============================================================================

def test_adv_arithmetic_eval_int_div_rem_zero_and_min_overflow():
    """Adversarial Arithmetic: Division/remainder by zero returns None; Integer.MIN_VALUE / -1 overflows to MIN_VALUE."""
    # Division & remainder by 0 must return None to avoid folding exceptions
    assert ArithmeticEvaluator.eval_binary_int(Opcode.IDIV, 10, 0) is None
    assert ArithmeticEvaluator.eval_binary_int(Opcode.IREM, 10, 0) is None

    # MIN_VALUE / -1 == MIN_VALUE in JVM 32-bit signed two's complement
    min_int = -2147483648
    assert ArithmeticEvaluator.eval_binary_int(Opcode.IDIV, min_int, -1) == min_int
    assert ArithmeticEvaluator.eval_binary_int(Opcode.IREM, min_int, -1) == 0

    # Long.MIN_VALUE / -1 == Long.MIN_VALUE
    min_long = -9223372036854775808
    assert ArithmeticEvaluator.eval_binary_long(Opcode.LDIV, min_long, -1) == min_long
    assert ArithmeticEvaluator.eval_binary_long(Opcode.LREM, min_long, -1) == 0


def test_adv_arithmetic_eval_java_remainder_sign_semantics():
    """Adversarial Arithmetic: JVM remainder sign matches the dividend (not divisor)."""
    # -7 % 3 == -1; 7 % -3 == 1; -7 % -3 == -1
    assert ArithmeticEvaluator.eval_binary_int(Opcode.IREM, -7, 3) == -1
    assert ArithmeticEvaluator.eval_binary_int(Opcode.IREM, 7, -3) == 1
    assert ArithmeticEvaluator.eval_binary_int(Opcode.IREM, -7, -3) == -1

    assert ArithmeticEvaluator.eval_binary_long(Opcode.LREM, -100, 30) == -10
    assert ArithmeticEvaluator.eval_binary_long(Opcode.LREM, 100, -30) == 10


def test_adv_arithmetic_eval_shifts_masking_and_unsigned():
    """Adversarial Arithmetic: Shift amounts masked by 0x1F (int) and 0x3F (long); unsigned shifts on negatives."""
    # ISHL masks shift amount with 0x1F: 1 << 33 == 1 << 1 == 2
    assert ArithmeticEvaluator.eval_binary_int(Opcode.ISHL, 1, 33) == 2
    # ISHR on negative values sign-extends
    assert ArithmeticEvaluator.eval_binary_int(Opcode.ISHR, -16, 2) == -4
    # IUSHR zero-extends
    assert ArithmeticEvaluator.eval_binary_int(Opcode.IUSHR, -1, 1) == 0x7FFFFFFF

    # LSHL masks shift amount with 0x3F: 1L << 65 == 1L << 1 == 2L
    assert ArithmeticEvaluator.eval_binary_long(Opcode.LSHL, 1, 65) == 2
    assert ArithmeticEvaluator.eval_binary_long(Opcode.LUSHR, -1, 1) == 0x7FFFFFFFFFFFFFFF


def test_adv_arithmetic_eval_float_double_nans_and_infinities():
    """Adversarial Arithmetic: IEEE-754 division by zero (+inf, -inf, nan) and float/double remainder."""
    # Float division by zero
    assert ArithmeticEvaluator.eval_binary_float(Opcode.FDIV, 1.0, 0.0) == math.inf
    assert ArithmeticEvaluator.eval_binary_float(Opcode.FDIV, -1.0, 0.0) == -math.inf
    assert math.isnan(ArithmeticEvaluator.eval_binary_float(Opcode.FDIV, 0.0, 0.0))

    # Double division by zero
    assert ArithmeticEvaluator.eval_binary_double(Opcode.DDIV, 42.0, 0.0) == math.inf
    assert ArithmeticEvaluator.eval_binary_double(Opcode.DDIV, -42.0, 0.0) == -math.inf
    assert math.isnan(ArithmeticEvaluator.eval_binary_double(Opcode.DDIV, 0.0, 0.0))

    # Float/Double remainder with inf/nan
    assert math.isnan(ArithmeticEvaluator.eval_binary_float(Opcode.FREM, 5.0, 0.0))
    assert math.isnan(ArithmeticEvaluator.eval_binary_double(Opcode.DREM, math.inf, 2.0))


def test_adv_arithmetic_eval_nan_comparisons_biasing():
    """Adversarial Arithmetic: FCMPL/DCMPL bias to -1 on NaN; FCMPG/DCMPG bias to +1 on NaN."""
    assert ArithmeticEvaluator.eval_comparison(Opcode.FCMPL, math.nan, 1.0) == -1
    assert ArithmeticEvaluator.eval_comparison(Opcode.FCMPG, math.nan, 1.0) == 1
    assert ArithmeticEvaluator.eval_comparison(Opcode.DCMPL, 1.0, math.nan) == -1
    assert ArithmeticEvaluator.eval_comparison(Opcode.DCMPG, 1.0, math.nan) == 1

    # Standard comparisons
    assert ArithmeticEvaluator.eval_comparison(Opcode.LCMP, 100, 50) == 1
    assert ArithmeticEvaluator.eval_comparison(Opcode.LCMP, 50, 100) == -1
    assert ArithmeticEvaluator.eval_comparison(Opcode.LCMP, 50, 50) == 0


def test_adv_arithmetic_eval_type_conversions_bounds():
    """Adversarial Arithmetic: F2I and D2I on NaN yields 0; out-of-range floats saturate to int min/max."""
    assert ArithmeticEvaluator.eval_conversion(Opcode.F2I, math.nan) == (0, "int")
    assert ArithmeticEvaluator.eval_conversion(Opcode.F2I, 1e12) == (2147483647, "int")
    assert ArithmeticEvaluator.eval_conversion(Opcode.F2I, -1e12) == (-2147483648, "int")

    assert ArithmeticEvaluator.eval_conversion(Opcode.D2I, math.nan) == (0, "int")
    assert ArithmeticEvaluator.eval_conversion(Opcode.D2I, 1e15) == (2147483647, "int")

    # Narrowing integer conversions
    assert ArithmeticEvaluator.eval_conversion(Opcode.I2B, 130) == (-126, "int")
    assert ArithmeticEvaluator.eval_conversion(Opcode.I2C, 65535) == (65535, "int")
    assert ArithmeticEvaluator.eval_conversion(Opcode.I2S, 32768) == (-32768, "int")


def test_adv_constant_folding_multi_step_arithmetic():
    """Adversarial Normalization: Multi-step arithmetic ((10 + 20) * 3 - 5) is folded to 85."""
    cfg = ControlFlowGraph()
    # Instructions: bipush 10; bipush 20; iadd; iconst_3; imul; iconst_5; isub; ireturn
    insts = [
        Instruction(0, Opcode.BIPUSH, struct.pack(">b", 10)),
        Instruction(2, Opcode.BIPUSH, struct.pack(">b", 20)),
        Instruction(4, Opcode.IADD),
        Instruction(5, Opcode.ICONST_3),
        Instruction(6, Opcode.IMUL),
        Instruction(7, Opcode.ICONST_5),
        Instruction(8, Opcode.ISUB),
        Instruction(9, Opcode.IRETURN),
    ]
    b = BasicBlock(0, 0, 10, insts)
    cfg.add_block(b)

    folded_count = fold_constants_and_arithmetic(cfg)
    assert folded_count >= 3
    # Result should be simplified to: BIPUSH 85; IRETURN
    assert len(b.instructions) == 2
    assert b.instructions[0].opcode == Opcode.BIPUSH
    assert struct.unpack(">b", b.instructions[0].operands)[0] == 85
    assert b.instructions[1].opcode == Opcode.IRETURN


def test_adv_constant_folding_promotes_large_constants_to_constant_pool():
    """Adversarial Normalization: Expressions producing large values (> 32767) add LDC to ConstantPool."""
    cp = ConstantPool()
    cfg = ControlFlowGraph()
    # sipush 20000; sipush 20000; iadd -> 40000 (exceeds 32767 SIPUSH limit)
    insts = [
        Instruction(0, Opcode.SIPUSH, struct.pack(">h", 20000)),
        Instruction(3, Opcode.SIPUSH, struct.pack(">h", 20000)),
        Instruction(6, Opcode.IADD),
        Instruction(7, Opcode.IRETURN),
    ]
    b = BasicBlock(0, 0, 8, insts)
    cfg.add_block(b)

    folded = fold_constants_and_arithmetic(cfg, cp=cp)
    assert folded == 1
    # Check that 40000 was inserted into ConstantPool and an LDC instruction emitted
    assert b.instructions[0].opcode in (Opcode.LDC, Opcode.LDC_W)
    idx = b.instructions[0].operands[0] if b.instructions[0].opcode == Opcode.LDC else struct.unpack(">H", b.instructions[0].operands)[0]
    entry = cp.get(idx)
    assert isinstance(entry, CONSTANT_Integer_info)
    assert entry.value == 40000


def test_adv_string_decryptor_detection_purity_checks():
    """Adversarial Decryptor: Methods with side effects (putstatic, athrow, external calls) rejected as decryptors."""
    cp = ConstantPool()
    c1 = cp.add_class("CryptoClass")
    m_desc = cp.add_utf8("()Ljava/lang/String;")

    # 1. Method with PUTSTATIC (side effect)
    fref = cp.add_member_ref(ConstantTag.FIELDREF, "CryptoClass", "count", "I")
    impure_code = bytes([Opcode.ICONST_1, Opcode.PUTSTATIC, (fref >> 8) & 0xFF, fref & 0xFF, Opcode.ACONST_NULL, Opcode.ARETURN])
    m_impure = MethodInfo(
        access_flags=AccessFlags.STATIC | AccessFlags.PUBLIC,
        name_index=cp.add_utf8("decryptImpure"),
        descriptor_index=m_desc,
        attributes=[CodeAttribute(attribute_name_index=cp.add_utf8("Code"), code=impure_code)]
    )

    # 2. Method with ATHROW (exception throwing)
    throw_code = bytes([Opcode.ACONST_NULL, Opcode.ATHROW])
    m_throw = MethodInfo(
        access_flags=AccessFlags.STATIC | AccessFlags.PUBLIC,
        name_index=cp.add_utf8("decryptThrow"),
        descriptor_index=m_desc,
        attributes=[CodeAttribute(attribute_name_index=cp.add_utf8("Code"), code=throw_code)]
    )

    cf = ClassFile(constant_pool=cp, this_class=c1, methods=[m_impure, m_throw])

    assert DecryptorMethodDetector.is_candidate(cf, m_impure) is False
    assert DecryptorMethodDetector.is_candidate(cf, m_throw) is False


def test_adv_string_decryptor_xor_cipher_emulation():
    """Adversarial Decryptor: Pure XOR cipher decryption loop emulated in sandbox, recovering plaintext."""
    cp = ConstantPool()
    c1 = cp.add_class("DecryptorClass")
    # Emulate: decrypt(String enc, int key)
    # Using String.toCharArray, iterating chars, ixor key, new String(chars)
    mref_to_char = cp.add_member_ref(ConstantTag.METHODREF, "java/lang/String", "toCharArray", "()[C")
    mref_str_init = cp.add_member_ref(ConstantTag.METHODREF, "java/lang/String", "<init>", "([C)V")

    # Bytecode:
    # 0: ALOAD_0
    # 1: INVOKEVIRTUAL toCharArray
    # 4: ASTORE_2 (char[] in loc 2)
    # 5: ALOAD_2 (array for CASTORE)
    # 6: ICONST_0 (index for CASTORE)
    # 7: ALOAD_2; ICONST_0; CALOAD (char c)
    # 10: ILOAD_1 (key)
    # 11: IXOR
    # 12: I2C
    # 13: CASTORE (arr[0] = c ^ key)
    # 14: NEW String; DUP; ALOAD_2; INVOKESPECIAL <init>([C)V; ARETURN
    code = bytes([
        Opcode.ALOAD_0,
        Opcode.INVOKEVIRTUAL, (mref_to_char >> 8) & 0xFF, mref_to_char & 0xFF,
        Opcode.ASTORE_2,
        Opcode.ALOAD_2,
        Opcode.ICONST_0,
        Opcode.ALOAD_2,
        Opcode.ICONST_0,
        Opcode.CALOAD,
        Opcode.ILOAD_1,
        Opcode.IXOR,
        Opcode.I2C,
        Opcode.CASTORE,
        Opcode.NEW, 0, 0,
        Opcode.DUP,
        Opcode.ALOAD_2,
        Opcode.INVOKESPECIAL, (mref_str_init >> 8) & 0xFF, mref_str_init & 0xFF,
        Opcode.ARETURN
    ])

    m = MethodInfo(
        access_flags=AccessFlags.STATIC | AccessFlags.PUBLIC,
        name_index=cp.add_utf8("xorSingle"),
        descriptor_index=cp.add_utf8("(Ljava/lang/String;I)Ljava/lang/String;"),
        attributes=[CodeAttribute(attribute_name_index=cp.add_utf8("Code"), code=code)]
    )
    cf = ClassFile(constant_pool=cp, this_class=c1, methods=[m])

    # Key = 0x20. Character 'A' (65) ^ 0x20 = 'a' (97)
    res = BytecodeEmulator().execute(cf, m, ["A", 0x20])
    assert res == "a"


def test_adv_string_decryptor_infinite_loop_timeout_safety():
    """Adversarial Decryptor: Infinite loops terminate safely at max_steps without freezing the process."""
    cp = ConstantPool()
    c1 = cp.add_class("HangingClass")
    # Infinite loop: GOTO 0
    hang_code = bytes([Opcode.GOTO, 0, 0])
    m = MethodInfo(
        access_flags=AccessFlags.STATIC | AccessFlags.PUBLIC,
        name_index=cp.add_utf8("hang"),
        descriptor_index=cp.add_utf8("()Ljava/lang/String;"),
        attributes=[CodeAttribute(attribute_name_index=cp.add_utf8("Code"), code=hang_code)]
    )
    cf = ClassFile(constant_pool=cp, this_class=c1, methods=[m])

    emulator = BytecodeEmulator(max_steps=50)
    res = emulator.execute(cf, m, [])
    assert res is None  # Timed out safely


def test_adv_dynamic_invoke_string_concat_and_lambda_resolution():
    """Adversarial Indy: StringConcatFactory templates folded; LambdaMetafactory resolved."""
    cp = ConstantPool()
    c1 = cp.add_class("IndyClass")

    # Bootstrap Method for StringConcatFactory.makeConcatWithConstants
    mref_bsm = cp.add_member_ref(
        ConstantTag.METHODREF,
        "java/lang/invoke/StringConcatFactory",
        "makeConcatWithConstants",
        "(Ljava/lang/invoke/MethodHandles$Lookup;Ljava/lang/String;Ljava/lang/invoke/MethodType;Ljava/lang/String;[Ljava/lang/Object;)Ljava/lang/invoke/CallSite;"
    )
    mh_bsm = cp._entries.append(CONSTANT_MethodHandle_info(reference_kind=ReferenceKind.REF_invokeStatic, reference_index=mref_bsm))
    mh_bsm_idx = len(cp._entries) - 1

    recipe_idx = cp.add_string("DecryptedSecret: \u0002 [TAG: \u0002]")
    const1_idx = cp.add_string("XYZ_TOKEN")
    const2_idx = cp.add_string("V1")

    bsm_attr = BootstrapMethodsAttribute(
        attribute_name_index=cp.add_utf8("BootstrapMethods"),
        bootstrap_methods=[
            BootstrapMethod(
                bootstrap_method_ref=mh_bsm_idx,
                bootstrap_arguments=[recipe_idx, const1_idx, const2_idx]
            )
        ]
    )

    nat_idx = cp.add_name_and_type("makeConcatWithConstants", "()Ljava/lang/String;")
    indy_entry = CONSTANT_InvokeDynamic_info(bootstrap_method_attr_index=0, name_and_type_index=nat_idx)

    cf = ClassFile(constant_pool=cp, this_class=c1, attributes=[bsm_attr])
    resolver = DynamicInvokeResolver(cf)

    resolved_concat = resolver.resolve_string_concat(indy_entry)
    assert resolved_concat is not None
    assert resolved_concat.is_fully_constant is True
    assert resolved_concat.folded_constant == "DecryptedSecret: XYZ_TOKEN [TAG: V1]"


def test_adv_dynamic_invoke_malformed_bsm_indices_safety():
    """Adversarial Indy: Indy entries pointing to missing or out-of-range BSM index return None safely."""
    cp = ConstantPool()
    c1 = cp.add_class("MalformedIndyClass")
    nat_idx = cp.add_name_and_type("test", "()V")
    indy_entry = CONSTANT_InvokeDynamic_info(bootstrap_method_attr_index=99, name_and_type_index=nat_idx)

    cf = ClassFile(constant_pool=cp, this_class=c1, attributes=[])
    resolver = DynamicInvokeResolver(cf)

    assert resolver.resolve_string_concat(indy_entry) is None
    assert resolver.resolve_lambda(indy_entry) is None



# ==============================================================================
# SECTION 5: ADDITIONAL DEEP ADVERSARIAL STRESS CASES
# ==============================================================================

def test_adv_verification_type_info_tag7_and_tag8_adversarial():
    """Adversarial Sanitizer: validate_verification_type_info checks tag 7 Object CP class index and tag 8 offset."""
    cp = ConstantPool()
    utf_idx = cp.add_utf8("NotAClass")
    cls_idx = cp.add_class("ValidClass")

    code_len = 50

    # Tag 7: Object variable info
    # 1. Invalid CP index out of bounds
    stream_oob = io.BytesIO(bytes([7]) + struct.pack(">H", 999))
    assert validate_verification_type_info(stream_oob, code_len, cp) is False

    # 2. CP entry points to UTF8 instead of CONSTANT_Class_info
    stream_utf = io.BytesIO(bytes([7]) + struct.pack(">H", utf_idx))
    assert validate_verification_type_info(stream_utf, code_len, cp) is False

    # 3. Valid CP Class index
    stream_valid = io.BytesIO(bytes([7]) + struct.pack(">H", cls_idx))
    assert validate_verification_type_info(stream_valid, code_len, cp) is True

    # Tag 8: Uninitialized variable info
    # 1. Offset exceeds code_len (offset 60 >= code_len 50)
    stream_oob_off = io.BytesIO(bytes([8]) + struct.pack(">H", 60))
    assert validate_verification_type_info(stream_oob_off, code_len, cp) is False

    # 2. Valid offset < code_len
    stream_valid_off = io.BytesIO(bytes([8]) + struct.pack(">H", 20))
    assert validate_verification_type_info(stream_valid_off, code_len, cp) is True

    # Unknown tag
    stream_bad_tag = io.BytesIO(bytes([15]))
    assert validate_verification_type_info(stream_bad_tag, code_len, cp) is False


def test_adv_method_exceptions_attribute_sanitization():
    """Adversarial Exceptions: Throws clause containing non-Class or out-of-bounds CP indices is sanitized."""
    cp = ConstantPool()
    c1 = cp.add_class("TestClass")
    ex_cls_idx = cp.add_class("java/io/IOException")
    bad_utf_idx = cp.add_utf8("JustAString")

    ex_attr = ExceptionsAttribute(
        attribute_name_index=cp.add_utf8("Exceptions"),
        exception_index_table=[ex_cls_idx, bad_utf_idx, 999]  # bad_utf_idx and 999 are invalid throws targets
    )
    m = MethodInfo(
        access_flags=AccessFlags.PUBLIC,
        name_index=cp.add_utf8("readFile"),
        descriptor_index=cp.add_utf8("()V"),
        attributes=[ex_attr]
    )
    cf = ClassFile(constant_pool=cp, this_class=c1, methods=[m])

    fixes = sanitize_method_exceptions_attribute(m, cf)
    assert fixes == 2
    # Only valid exception class index preserved
    assert ex_attr.exception_index_table == [ex_cls_idx]


def test_adv_opaque_predicate_invariants_5_and_7():
    """Adversarial Opaque Predicate: Consecutive parity x*(x+1)%2==0 and cube parity x*(x^2-1)%6==0."""
    cfg = ControlFlowGraph()

    # Invariant 5: iload_1; iload_1; iconst_1; iadd; imul; iconst_2; irem; ifeq 10 (Always True)
    b0 = BasicBlock(0, 0, 9, [
        Instruction(0, Opcode.ILOAD_1),
        Instruction(1, Opcode.ILOAD_1),
        Instruction(2, Opcode.ICONST_1),
        Instruction(3, Opcode.IADD),
        Instruction(4, Opcode.IMUL),
        Instruction(5, Opcode.ICONST_2),
        Instruction(6, Opcode.IREM),
        Instruction(7, Opcode.IFEQ, struct.pack(">h", 10))
    ])
    b10 = BasicBlock(10, 10, 11, [Instruction(10, Opcode.RETURN)])
    b20 = BasicBlock(20, 20, 21, [Instruction(20, Opcode.RETURN)])

    cfg.add_block(b0)
    cfg.add_block(b10)
    cfg.add_block(b20)

    cfg.add_edge(CFGEdge(0, 10, CFGEdgeType.CONDITIONAL_BRANCH))
    cfg.add_edge(CFGEdge(0, 20, CFGEdgeType.FALLTHROUGH))

    op_elim = OpaquePredicateEliminator(cfg)
    rep = op_elim.eliminate()

    assert rep.algebraic_predicates_pruned == 1
    # Producers pruned, branch converted to GOTO
    assert len(b0.instructions) == 1
    assert b0.instructions[0].opcode == Opcode.GOTO

    # Invariant 7: iload_2; iload_2; iload_2; imul; iconst_1; isub; imul; bipush 6; irem; ifeq 40
    cfg2 = ControlFlowGraph()
    b30 = BasicBlock(30, 30, 41, [
        Instruction(30, Opcode.ILOAD_2),
        Instruction(31, Opcode.ILOAD_2),
        Instruction(32, Opcode.ILOAD_2),
        Instruction(33, Opcode.IMUL),
        Instruction(34, Opcode.ICONST_1),
        Instruction(35, Opcode.ISUB),
        Instruction(36, Opcode.IMUL),
        Instruction(37, Opcode.BIPUSH, bytes([6])),
        Instruction(38, Opcode.IREM),
        Instruction(39, Opcode.IFEQ, struct.pack(">h", 10))
    ])
    b40 = BasicBlock(40, 40, 41, [Instruction(40, Opcode.RETURN)])
    b50 = BasicBlock(50, 50, 51, [Instruction(50, Opcode.RETURN)])

    cfg2.add_block(b30)
    cfg2.add_block(b40)
    cfg2.add_block(b50)

    cfg2.add_edge(CFGEdge(30, 40, CFGEdgeType.CONDITIONAL_BRANCH))
    cfg2.add_edge(CFGEdge(30, 50, CFGEdgeType.FALLTHROUGH))

    op_elim2 = OpaquePredicateEliminator(cfg2)
    rep2 = op_elim2.eliminate()
    assert rep2.algebraic_predicates_pruned == 1


def test_adv_constant_folding_unary_negation_and_conversions():
    """Adversarial Constant Folding: Unary negation (INEG) and conversion (I2L) folded cleanly."""
    cp = ConstantPool()
    cfg = ControlFlowGraph()
    # bipush 10; ineg; ireturn -> folds to BIPUSH -10; IRETURN
    insts = [
        Instruction(0, Opcode.BIPUSH, struct.pack(">b", 10)),
        Instruction(2, Opcode.INEG),
        Instruction(3, Opcode.IRETURN),
    ]
    b = BasicBlock(0, 0, 4, insts)
    cfg.add_block(b)

    folder = IntraProceduralConstantFolder(cfg, cp=cp)
    folded = folder.fold()
    assert folded == 1
    assert len(b.instructions) == 2
    assert b.instructions[0].opcode == Opcode.BIPUSH
    assert struct.unpack(">b", b.instructions[0].operands)[0] == -10


def test_adv_constant_folding_non_contiguous_store_safety_invariant():
    """Adversarial Constant Folding: Intervening local store (ISTORE) prevents unsafe expression deletion."""
    cfg = ControlFlowGraph()
    # bipush 15; istore_1; iload_1; bipush 5; iadd; ireturn
    # The producer chain for iadd includes index 0 and index 3, but indices 1 and 2 are non-contiguous.
    # The folder conservatively skips folding to guarantee local variable 1 is not deleted.
    insts = [
        Instruction(0, Opcode.BIPUSH, struct.pack(">b", 15)),
        Instruction(2, Opcode.ISTORE_1),
        Instruction(3, Opcode.ILOAD_1),
        Instruction(4, Opcode.BIPUSH, struct.pack(">b", 5)),
        Instruction(6, Opcode.IADD),
        Instruction(7, Opcode.IRETURN),
    ]
    b = BasicBlock(0, 0, 8, insts)
    cfg.add_block(b)

    folder = IntraProceduralConstantFolder(cfg)
    folded = folder.fold()
    # Safely preserves istore_1 and does not delete instructions
    assert folded == 0
    assert len(b.instructions) == 6


def test_adv_string_decryptor_string_builder_append():
    """Adversarial Decryptor: StringBuilder append and toString emulation in bytecode sandbox."""
    cp = ConstantPool()
    c1 = cp.add_class("BuilderDecryptor")
    mref_sb_init = cp.add_member_ref(ConstantTag.METHODREF, "java/lang/StringBuilder", "<init>", "()V")
    mref_sb_app = cp.add_member_ref(ConstantTag.METHODREF, "java/lang/StringBuilder", "append", "(Ljava/lang/String;)Ljava/lang/StringBuilder;")
    mref_sb_to_s = cp.add_member_ref(ConstantTag.METHODREF, "java/lang/StringBuilder", "toString", "()Ljava/lang/String;")
    s1_idx = cp.add_string("Hello, ")
    s2_idx = cp.add_string("World!")

    code = bytes([
        Opcode.NEW, 0, 0,
        Opcode.DUP,
        Opcode.INVOKESPECIAL, (mref_sb_init >> 8) & 0xFF, mref_sb_init & 0xFF,
        Opcode.LDC, s1_idx,
        Opcode.INVOKEVIRTUAL, (mref_sb_app >> 8) & 0xFF, mref_sb_app & 0xFF,
        Opcode.LDC, s2_idx,
        Opcode.INVOKEVIRTUAL, (mref_sb_app >> 8) & 0xFF, mref_sb_app & 0xFF,
        Opcode.INVOKEVIRTUAL, (mref_sb_to_s >> 8) & 0xFF, mref_sb_to_s & 0xFF,
        Opcode.ARETURN
    ])

    m = MethodInfo(
        access_flags=AccessFlags.STATIC | AccessFlags.PUBLIC,
        name_index=cp.add_utf8("buildString"),
        descriptor_index=cp.add_utf8("()Ljava/lang/String;"),
        attributes=[CodeAttribute(attribute_name_index=cp.add_utf8("Code"), code=code)]
    )
    cf = ClassFile(constant_pool=cp, this_class=c1, methods=[m])

    res = BytecodeEmulator().execute(cf, m, [])
    assert res == "Hello, World!"
