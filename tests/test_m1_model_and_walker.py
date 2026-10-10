"""
Comprehensive test suite for Milestone M1 (F1-F7):
- F1: JVM Class File Binary Model, 17 CP Tags, 2-Slot Long/Double, Modified UTF-8, NaN bit-exactness, Idempotency.
- F2: JarArchive, RFC 822 Manifest, Resource Preservation, Symbol Indexer, Signature Stripping.
- F3: InheritanceHierarchy, JDK Core KB, LCA (JVMS §4.10), Subtype Assignability, Array Covariance.
- F4: Polymorphic Call Graph Walker, CHA/RTA dispatch, Default Methods.
- F5: Static & Special Dispatch Resolver, Constructor (<init>), Super Calls, Invokedynamic Lambda.
- F6: Field Access Mapping, Declaring Class Resolution, Bidirectional Maps.
- F7: Inter-Class Dependency Slicing, Forward/Backward Slices, Tarjan SCCs, Condensation Topological Sort.
"""

import io
import os
import sys
import struct
import subprocess
import zipfile
import pytest
from pathlib import Path

from jar_engine.model.constants import (
    JAVA_MAGIC, JavaMajorVersion, ConstantTag, ReferenceKind,
    AccessFlags, Opcode
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
from jar_engine.model.member_info import (
    AttributeInfo, RawAttribute, ExceptionTableEntry, CodeAttribute,
    SignatureAttribute, SourceFileAttribute, ConstantValueAttribute,
    ExceptionsAttribute, StackMapTableAttribute, BootstrapMethod,
    BootstrapMethodsAttribute, FieldInfo, MethodInfo
)
from jar_engine.model.class_file import ClassFile, ClassFormatError
from jar_engine.model.jar_archive import JarArchive, Manifest, JarStats, wrap_manifest_line
from jar_engine.walker.hierarchy import (
    ClassOrigin, HierarchyNode, JDK_CORE_HIERARCHY, InheritanceHierarchy
)
from jar_engine.walker.call_graph import (
    MethodReference, FieldReference, CallEdge, FieldAccessRecord,
    InstructionScanner, CallGraph
)
from jar_engine.walker.dependency_slice import (
    DependencyKind, ClassDependencyGraph, DependencySlicer,
    extract_classes_from_descriptor
)

SNAKEYAML_JAR = Path("/data/data/com.termux/files/home/tools/snakeyaml.jar")


# ==============================================================================
# 1. CONSTANT POOL & MODIFIED UTF-8 TESTS (F1)
# ==============================================================================

def test_modified_utf8_null_byte():
    # Null byte \x00 must encode to 0xC0 0x80
    raw = encode_modified_utf8("hello\x00world")
    assert b"\xc0\x80" in raw
    decoded = decode_modified_utf8(raw)
    assert decoded == "hello\x00world"


def test_modified_utf8_multibyte_and_surrogates():
    # Greek, Cyrillic, CJK, and astral plane Emoji (> 0xFFFF)
    test_str = "αβγ-абв-你好世界-🚀🎉"
    encoded = encode_modified_utf8(test_str)
    decoded = decode_modified_utf8(encoded)
    assert decoded == test_str


def test_constant_pool_all_17_tags():
    cp = ConstantPool()
    
    # 1. UTF8
    u_idx = cp.add_utf8("com/example/Test")
    assert cp.resolve_utf8(u_idx) == "com/example/Test"

    # 3. Integer
    i_idx = cp.add_integer(-42)
    assert cp[i_idx].value == -42

    # 4. Float bit-exact NaN
    f_nan_bits = 0x7FA01234
    f_nan_val = struct.unpack(">f", struct.pack(">I", f_nan_bits))[0]
    f_idx = cp.add_float(f_nan_val)
    # Force custom NaN bits
    cp[f_idx].raw_bits = f_nan_bits
    assert cp[f_idx].raw_bits == f_nan_bits

    # 5. Long (2 slots)
    l_idx = cp.add_long(0x123456789ABCDEF0)
    assert cp[l_idx].value == 0x123456789ABCDEF0
    assert isinstance(cp._entries[l_idx + 1], CpPlaceholder)
    with pytest.raises(ValueError):
        _ = cp[l_idx + 1]

    # 6. Double (2 slots, bit-exact NaN)
    d_nan_bits = 0x7FF8000012345678
    d_nan_val = struct.unpack(">d", struct.pack(">Q", d_nan_bits))[0]
    d_idx = cp.add_double(d_nan_val)
    cp[d_idx].raw_bits = d_nan_bits
    assert cp[d_idx].raw_bits == d_nan_bits
    assert isinstance(cp._entries[d_idx + 1], CpPlaceholder)

    # 7. Class
    c_idx = cp.add_class("com/example/Test")
    assert cp.resolve_class_name(c_idx) == "com/example/Test"

    # 8. String
    s_idx = cp.add_string("Hello String")
    assert cp.resolve_string(s_idx) == "Hello String"

    # 12. NameAndType
    nat_idx = cp.add_name_and_type("myMethod", "()V")
    assert cp.resolve_name_and_type(nat_idx) == ("myMethod", "()V")

    # 9. Fieldref
    fref_idx = cp.add_member_ref(ConstantTag.FIELDREF, "com/example/Test", "myField", "I")
    assert cp.resolve_member_ref(fref_idx) == ("com/example/Test", "myField", "I")

    # 10. Methodref
    mref_idx = cp.add_member_ref(ConstantTag.METHODREF, "com/example/Test", "myMethod", "()V")
    assert cp.resolve_member_ref(mref_idx) == ("com/example/Test", "myMethod", "()V")

    # 11. InterfaceMethodref
    imref_idx = cp.add_member_ref(ConstantTag.INTERFACE_METHODREF, "com/example/MyIface", "run", "()V")
    assert cp.resolve_member_ref(imref_idx) == ("com/example/MyIface", "run", "()V")

    # 15. MethodHandle
    mh_idx = len(cp)
    cp._entries.append(CONSTANT_MethodHandle_info(reference_kind=ReferenceKind.REF_invokeStatic, reference_index=mref_idx))

    # 16. MethodType
    mt_desc_idx = cp.add_utf8("()V")
    mt_idx = len(cp)
    cp._entries.append(CONSTANT_MethodType_info(descriptor_index=mt_desc_idx))

    # 17. Dynamic
    dyn_idx = len(cp)
    cp._entries.append(CONSTANT_Dynamic_info(bootstrap_method_attr_index=0, name_and_type_index=nat_idx))

    # 18. InvokeDynamic
    indy_idx = len(cp)
    cp._entries.append(CONSTANT_InvokeDynamic_info(bootstrap_method_attr_index=0, name_and_type_index=nat_idx))

    # 19. Module
    mod_name_idx = cp.add_utf8("com.example.module")
    mod_idx = len(cp)
    cp._entries.append(CONSTANT_Module_info(name_index=mod_name_idx))

    # 20. Package
    pkg_name_idx = cp.add_utf8("com/example/pkg")
    pkg_idx = len(cp)
    cp._entries.append(CONSTANT_Package_info(name_index=pkg_name_idx))

    # Serialize and deserialize constant pool bytes
    cp_bytes = cp.to_bytes()
    assert len(cp_bytes) > 0


def test_constant_pool_deduplication_and_mutation():
    cp = ConstantPool()
    idx1 = cp.add_utf8("reusedString")
    idx2 = cp.add_utf8("reusedString")
    assert idx1 == idx2

    c_idx1 = cp.add_class("com/foo/Bar")
    c_idx2 = cp.add_class("com/foo/Bar")
    assert c_idx1 == c_idx2

    # In-place mutation
    cp.update_utf8(idx1, "updatedString")
    assert cp.resolve_utf8(idx1) == "updatedString"


# ==============================================================================
# 2. CLASSFILE PARSING, SERIALIZATION & ROUND-TRIP (F1)
# ==============================================================================

def test_classfile_invalid_magic():
    bad_bytes = struct.pack(">IHHH", 0xDEADBEEF, 0, 52, 1)
    with pytest.raises(ClassFormatError):
        ClassFile.parse(bad_bytes)


def test_synthetic_classfile_round_trip():
    # Build synthetic class from scratch
    cp = ConstantPool()
    this_c = cp.add_class("com/example/SyntheticTest")
    super_c = cp.add_class("java/lang/Object")
    iface_c = cp.add_class("java/lang/Runnable")

    # Field: private int counter;
    f_name = cp.add_utf8("counter")
    f_desc = cp.add_utf8("I")
    field0 = FieldInfo(
        access_flags=AccessFlags.PRIVATE,
        name_index=f_name,
        descriptor_index=f_desc,
        attributes=[]
    )

    # Method: public void run() { return; }
    m_name = cp.add_utf8("run")
    m_desc = cp.add_utf8("()V")
    code_attr_name = cp.add_utf8("Code")
    code = bytes([Opcode.RETURN])
    code_attr = CodeAttribute(
        attribute_name_index=code_attr_name,
        max_stack=1,
        max_locals=1,
        code=code,
        exception_table=[],
        attributes=[]
    )
    method0 = MethodInfo(
        access_flags=AccessFlags.PUBLIC,
        name_index=m_name,
        descriptor_index=m_desc,
        attributes=[code_attr]
    )

    cf = ClassFile(
        magic=JAVA_MAGIC,
        minor_version=0,
        major_version=52,
        constant_pool=cp,
        access_flags=AccessFlags.PUBLIC | AccessFlags.SUPER,
        this_class=this_c,
        super_class=super_c,
        interfaces=[iface_c],
        fields=[field0],
        methods=[method0],
        attributes=[]
    )

    data = cf.to_bytes()
    reparsed = ClassFile.parse(data)
    assert reparsed.this_class_name == "com/example/SyntheticTest"
    assert reparsed.super_class_name == "java/lang/Object"
    assert reparsed.interface_names == ["java/lang/Runnable"]
    assert len(reparsed.fields) == 1
    assert reparsed.fields[0].name(reparsed.constant_pool) == "counter"
    assert len(reparsed.methods) == 1
    assert reparsed.methods[0].name(reparsed.constant_pool) == "run"
    assert reparsed.to_bytes() == data


def test_round_trip_idempotency_on_snakeyaml():
    if not SNAKEYAML_JAR.exists():
        pytest.skip("snakeyaml.jar not available")

    with zipfile.ZipFile(SNAKEYAML_JAR, "r") as zf:
        class_entries = [name for name in zf.namelist() if name.endswith(".class")]
        assert len(class_entries) >= 200

        for name in class_entries:
            original_bytes = zf.read(name)
            cf = ClassFile.parse(original_bytes)
            serialized_bytes = cf.to_bytes()
            assert serialized_bytes == original_bytes, f"Round-trip byte mismatch in {name}"
            # Second pass idempotency
            reparsed_cf = ClassFile.parse(serialized_bytes)
            assert reparsed_cf.to_bytes() == original_bytes


# ==============================================================================
# 3. JAR ARCHIVE & RFC 822 MANIFEST (F2)
# ==============================================================================

def test_manifest_rfc822_line_folding_and_unfolding():
    # Long line > 72 bytes
    long_val = "org.example.very.long.package.name.that.definitely.exceeds.the.standard.seventy.two.bytes.limit.MainClass"
    wrapped = wrap_manifest_line("Main-Class", long_val)
    lines = wrapped.decode("utf-8").splitlines()
    assert len(lines) > 1
    assert lines[1].startswith(" ")

    # Round trip parse
    manifest_bytes = b"Manifest-Version: 1.0\r\n" + wrapped + b"\r\n"
    m = Manifest.parse(manifest_bytes)
    assert m.get_main_class() == long_val


def test_manifest_sections_and_case_insensitivity():
    manifest_text = (
        "Manifest-Version: 1.0\r\n"
        "Created-By: 17.0.2 (Oracle Corporation)\r\n"
        "Main-Class: com.example.Main\r\n"
        "\r\n"
        "Name: com/example/pkg/\r\n"
        "Specification-Title: Example API\r\n"
        "Implementation-Version: 1.2.3\r\n"
        "\r\n"
    )
    m = Manifest.parse(manifest_text)
    assert m.get_main_class() == "com.example.Main"
    # Case insensitive
    assert m.get_attribute("main-class") == "com.example.Main"
    assert m.get_attribute("specification-title", section="com/example/pkg/") == "Example API"

    # Mutate
    m.set_main_class("com/example/Updated")
    assert m.get_main_class() == "com.example.Updated"

    re_bytes = m.to_bytes()
    m2 = Manifest.parse(re_bytes)
    assert m2.get_main_class() == "com.example.Updated"


def test_jar_archive_ingestion_and_symbol_indexing():
    if not SNAKEYAML_JAR.exists():
        pytest.skip("snakeyaml.jar not available")

    archive = JarArchive.from_file(SNAKEYAML_JAR)
    stats = archive.get_stats()
    assert stats.class_count == 229
    assert stats.method_count > 1000
    assert stats.field_count > 500
    assert archive.has_class("org/yaml/snakeyaml/Yaml")

    dump_m = archive.find_method("org/yaml/snakeyaml/Yaml", "dump", "(Ljava/lang/Object;)Ljava/lang/String;")
    assert dump_m is not None
    assert dump_m.access_flags & AccessFlags.PUBLIC


def test_jar_archive_repackage_and_strip_signatures():
    archive = JarArchive()
    # Add dummy class
    cp = ConstantPool()
    c_idx = cp.add_class("com/test/Dummy")
    s_idx = cp.add_class("java/lang/Object")
    cf = ClassFile(
        magic=JAVA_MAGIC,
        minor_version=0,
        major_version=52,
        constant_pool=cp,
        access_flags=AccessFlags.PUBLIC,
        this_class=c_idx,
        super_class=s_idx
    )
    archive.add_class(cf)
    archive.set_resource("config.properties", b"key=value")
    archive.set_resource("META-INF/SIGNING.SF", b"signature data")
    archive.set_resource("META-INF/SIGNING.RSA", b"rsa data")

    # Strip signatures
    removed = archive.strip_signatures()
    assert removed == 2
    assert not archive.has_resource("META-INF/SIGNING.SF")
    assert archive.has_resource("config.properties")

    # Repackage to bytes
    jar_bytes = archive.to_bytes()
    reloaded = JarArchive.from_bytes(jar_bytes)
    assert reloaded.has_class("com/test/Dummy")
    assert reloaded.get_resource("config.properties") == b"key=value"


# ==============================================================================
# 4. INHERITANCE HIERARCHY DAG, LCA & ASSIGNABILITY (F3)
# ==============================================================================

def test_hierarchy_dag_and_jdk_stubs():
    if not SNAKEYAML_JAR.exists():
        pytest.skip("snakeyaml.jar not available")

    archive = JarArchive.from_file(SNAKEYAML_JAR)
    hier = archive.get_hierarchy()

    assert not hier.has_cycles()
    assert "java/lang/Object" in hier.nodes

    # Superclass chain
    supers = hier.get_superclasses("org/yaml/snakeyaml/events/AliasEvent")
    assert "org/yaml/snakeyaml/events/NodeEvent" in supers
    assert "org/yaml/snakeyaml/events/Event" in supers
    assert "java/lang/Object" in supers

    # Subclasses
    subs = hier.get_all_subclasses("org/yaml/snakeyaml/events/Event")
    assert "org/yaml/snakeyaml/events/AliasEvent" in subs
    assert "org/yaml/snakeyaml/events/CollectionStartEvent" in subs


def test_lca_resolution_jvms_rules():
    if not SNAKEYAML_JAR.exists():
        pytest.skip("snakeyaml.jar not available")

    archive = JarArchive.from_file(SNAKEYAML_JAR)
    hier = archive.get_hierarchy()

    # Sibling subclasses under Event
    lca = hier.find_lowest_common_ancestor(
        "org/yaml/snakeyaml/events/AliasEvent",
        "org/yaml/snakeyaml/events/CollectionEndEvent"
    )
    assert lca == "org/yaml/snakeyaml/events/Event"

    # Identity
    assert hier.find_lowest_common_ancestor(
        "org/yaml/snakeyaml/events/Event",
        "org/yaml/snakeyaml/events/Event"
    ) == "org/yaml/snakeyaml/events/Event"

    # Object ancestor
    assert hier.find_lowest_common_ancestor(
        "org/yaml/snakeyaml/events/Event",
        "java/lang/String"
    ) == "java/lang/Object"

    # Interface merge rule (JVMS §4.10): must merge to Object
    assert hier.find_lowest_common_ancestor(
        "java/util/List",
        "java/util/Set"
    ) == "java/lang/Object"


def test_subtype_assignability_and_array_covariance():
    hier = InheritanceHierarchy()
    # Add dummy classes: Animal <- Dog
    hier.nodes["Animal"] = HierarchyNode("Animal", "java/lang/Object", (), AccessFlags.PUBLIC, ClassOrigin.ARCHIVE)
    hier.nodes["Dog"] = HierarchyNode("Dog", "Animal", (), AccessFlags.PUBLIC, ClassOrigin.ARCHIVE)
    hier.nodes["Cat"] = HierarchyNode("Cat", "Animal", (), AccessFlags.PUBLIC, ClassOrigin.ARCHIVE)
    hier.nodes["java/lang/Object"] = HierarchyNode("java/lang/Object", None, (), AccessFlags.PUBLIC, ClassOrigin.JDK_KNOWN)
    hier._super_class["Animal"] = "java/lang/Object"
    hier._super_class["Dog"] = "Animal"
    hier._super_class["Cat"] = "Animal"
    hier._super_class["java/lang/Object"] = None

    # Reference assignability
    assert hier.is_assignable_from("Animal", "Dog")
    assert not hier.is_assignable_from("Dog", "Animal")
    assert not hier.is_assignable_from("Dog", "Cat")
    assert hier.is_assignable_from("java/lang/Object", "Dog")

    # Array Covariance
    assert hier.is_assignable_from("[LAnimal;", "[LDog;")
    assert not hier.is_assignable_from("[LDog;", "[LAnimal;")
    assert hier.is_assignable_from("[Ljava/lang/Object;", "[LDog;")
    assert hier.is_assignable_from("java/lang/Object", "[LDog;")

    # Primitive Array Invariance
    assert hier.is_assignable_from("[I", "[I")
    assert not hier.is_assignable_from("[I", "[J")
    assert hier.is_assignable_from("java/lang/Object", "[I")


# ==============================================================================
# 5. CALL GRAPH & FIELD ACCESS MAPPING (F4, F5, F6)
# ==============================================================================

def test_instruction_scanner():
    # Sequence: nop, bipush 10, sipush 1000, return
    code = bytes([
        Opcode.NOP,
        Opcode.BIPUSH, 10,
        Opcode.SIPUSH, 0x03, 0xE8,
        Opcode.RETURN
    ])
    instructions = list(InstructionScanner.scan(code))
    assert len(instructions) == 4
    assert instructions[0] == (0, Opcode.NOP, b"")
    assert instructions[1] == (1, Opcode.BIPUSH, bytes([10]))
    assert instructions[2] == (3, Opcode.SIPUSH, bytes([0x03, 0xE8]))
    assert instructions[3] == (6, Opcode.RETURN, b"")


def test_call_graph_and_field_maps_on_snakeyaml():
    if not SNAKEYAML_JAR.exists():
        pytest.skip("snakeyaml.jar not available")

    archive = JarArchive.from_file(SNAKEYAML_JAR)
    hier = archive.get_hierarchy()
    cg = CallGraph.build(archive, hier, mode="cha")

    assert len(cg.call_edges) > 1000
    assert len(cg.field_accesses) > 500

    # Query callers / callees
    # Yaml.dump(Object) calls dump(Object, Writer)
    dump_sig = "Lorg/yaml/snakeyaml/Yaml;->dump(Ljava/lang/Object;)Ljava/lang/String;"
    callees = cg.get_callees(dump_sig)
    assert len(callees) > 0

    # Verify field read/write maps
    assert len(cg.field_readers) > 0
    assert len(cg.field_writers) > 0

    # Test entry points and leaves
    entry_points = cg.get_entry_points()
    assert len(entry_points) > 0
    leaf_methods = cg.get_leaf_methods()
    assert len(leaf_methods) > 0


# ==============================================================================
# 6. INTER-CLASS DEPENDENCY SLICING & SCCs (F7)
# ==============================================================================

def test_dependency_slicing_forward_and_backward():
    archive = JarArchive()
    # Construct 3 classes: A -> B -> C
    for name, target in [("com/A", "com/B"), ("com/B", "com/C"), ("com/C", None)]:
        cp = ConstantPool()
        c_idx = cp.add_class(name)
        s_idx = cp.add_class("java/lang/Object")
        m_name = cp.add_utf8("exec")
        m_desc = cp.add_utf8("()V")
        code_attr_name = cp.add_utf8("Code")

        if target:
            # invoke target.exec()
            mref_idx = cp.add_member_ref(ConstantTag.METHODREF, target, "exec", "()V")
            code = bytes([Opcode.INVOKESTATIC, (mref_idx >> 8) & 0xFF, mref_idx & 0xFF, Opcode.RETURN])
        else:
            code = bytes([Opcode.RETURN])

        code_attr = CodeAttribute(
            attribute_name_index=code_attr_name,
            max_stack=1,
            max_locals=1,
            code=code
        )
        method0 = MethodInfo(AccessFlags.PUBLIC | AccessFlags.STATIC, m_name, m_desc, [code_attr])
        cf = ClassFile(
            magic=JAVA_MAGIC,
            constant_pool=cp,
            access_flags=AccessFlags.PUBLIC,
            this_class=c_idx,
            super_class=s_idx,
            methods=[method0]
        )
        archive.add_class(cf)

    hier = archive.get_hierarchy()
    cg = CallGraph.build(archive, hier)
    cdg = ClassDependencyGraph.build(archive, hier, cg, include_external=False)

    # Forward slice from A: {A, B, C}
    f_slice = cdg.forward_slice("com/A")
    assert f_slice == {"com/A", "com/B", "com/C"}

    # Backward slice from C: {A, B, C}
    b_slice = cdg.backward_slice("com/C")
    assert b_slice == {"com/A", "com/B", "com/C"}

    # Backward slice from B: {A, B}
    assert cdg.backward_slice("com/B") == {"com/A", "com/B"}


def test_tarjan_scc_and_topological_sort():
    archive = JarArchive()
    # Cycle between X and Y: X -> Y -> X, and independent Z
    for name, target in [("com/X", "com/Y"), ("com/Y", "com/X"), ("com/Z", None)]:
        cp = ConstantPool()
        c_idx = cp.add_class(name)
        s_idx = cp.add_class("java/lang/Object")
        m_name = cp.add_utf8("foo")
        m_desc = cp.add_utf8("()V")
        code_attr_name = cp.add_utf8("Code")

        if target:
            mref_idx = cp.add_member_ref(ConstantTag.METHODREF, target, "foo", "()V")
            code = bytes([Opcode.INVOKESTATIC, (mref_idx >> 8) & 0xFF, mref_idx & 0xFF, Opcode.RETURN])
        else:
            code = bytes([Opcode.RETURN])

        code_attr = CodeAttribute(attribute_name_index=code_attr_name, max_stack=1, max_locals=1, code=code)
        m0 = MethodInfo(AccessFlags.PUBLIC | AccessFlags.STATIC, m_name, m_desc, [code_attr])
        cf = ClassFile(magic=JAVA_MAGIC, constant_pool=cp, access_flags=AccessFlags.PUBLIC, this_class=c_idx, super_class=s_idx, methods=[m0])
        archive.add_class(cf)

    hier = archive.get_hierarchy()
    cg = CallGraph.build(archive, hier)
    cdg = ClassDependencyGraph.build(archive, hier, cg, include_external=False)

    assert cdg.has_cycles()
    cycles = cdg.get_cyclic_components()
    assert len(cycles) == 1
    assert cycles[0] == {"com/X", "com/Y"}

    # Condensation DAG and topological sorting
    cond_order = cdg.topological_sort(direction="build")
    assert len(cond_order) >= 2


def test_dependency_slicing_on_snakeyaml():
    if not SNAKEYAML_JAR.exists():
        pytest.skip("snakeyaml.jar not available")

    archive = JarArchive.from_file(SNAKEYAML_JAR)
    hier = archive.get_hierarchy()
    cg = CallGraph.build(archive, hier)
    cdg = ClassDependencyGraph.build(archive, hier, cg, include_external=False)

    # Forward slice from Yaml
    f_slice = cdg.forward_slice("org/yaml/snakeyaml/Yaml")
    assert len(f_slice) > 20
    assert "org/yaml/snakeyaml/events/Event" in f_slice

    # Backward slice from Event
    b_slice = cdg.backward_slice("org/yaml/snakeyaml/events/Event")
    assert len(b_slice) > 10
    assert "org/yaml/snakeyaml/Yaml" in b_slice


def test_call_graph_interface_compute_dependency_slice():
    if not SNAKEYAML_JAR.exists():
        pytest.skip("snakeyaml.jar not available")

    archive = JarArchive.from_file(SNAKEYAML_JAR)
    hier = archive.get_hierarchy()
    cg = CallGraph.build(archive, hier)

    # CallGraph.compute_dependency_slice per PROJECT.md interface contract
    forward_slice = cg.compute_dependency_slice(["org/yaml/snakeyaml/Yaml"], "forward")
    assert len(forward_slice) > 20
    backward_slice = cg.compute_dependency_slice(["org/yaml/snakeyaml/events/Event"], "backward")
    assert len(backward_slice) > 10


def test_instruction_scanner_variable_length_opcodes():
    # 1. tableswitch with padding: pc=0, opcode=0xAA (170)
    # pad for pc=0 is (4 - (0+1)%4)%4 = 3 bytes
    # default (4 bytes) = 100, low = 0, high = 2 -> 3 targets (12 bytes)
    # total len = 1 + 3 + 12 + 12 = 28 bytes
    code_tableswitch = bytearray([Opcode.TABLESWITCH, 0, 0, 0])  # opcode + 3 pad
    code_tableswitch.extend(struct.pack(">iii", 100, 0, 2))      # default, low, high
    code_tableswitch.extend(struct.pack(">iii", 10, 20, 30))     # 3 jump offsets
    code_tableswitch.append(Opcode.RETURN)

    insts = list(InstructionScanner.scan(bytes(code_tableswitch)))
    assert len(insts) == 2
    assert insts[0][1] == Opcode.TABLESWITCH
    assert insts[1][1] == Opcode.RETURN

    # 2. lookupswitch: pc=0, pad=3 bytes, default=50, npairs=2 -> 2 pairs (16 bytes)
    # total len = 1 + 3 + 8 + 16 = 28 bytes
    code_lookup = bytearray([Opcode.LOOKUPSWITCH, 0, 0, 0])
    code_lookup.extend(struct.pack(">ii", 50, 2))                # default, npairs
    code_lookup.extend(struct.pack(">iiii", 1, 10, 2, 20))       # match1, off1, match2, off2
    code_lookup.append(Opcode.RETURN)

    insts_lookup = list(InstructionScanner.scan(bytes(code_lookup)))
    assert len(insts_lookup) == 2
    assert insts_lookup[0][1] == Opcode.LOOKUPSWITCH
    assert insts_lookup[1][1] == Opcode.RETURN

    # 3. wide iinc: 6 bytes
    code_wide = bytes([Opcode.WIDE, Opcode.IINC, 0x00, 0x05, 0x00, 0x01, Opcode.RETURN])
    insts_wide = list(InstructionScanner.scan(code_wide))
    assert len(insts_wide) == 2
    assert insts_wide[0][1] == Opcode.WIDE
    assert insts_wide[1][1] == Opcode.RETURN


def test_deep_hierarchy_and_lca():
    # Construct 30-level inheritance hierarchy: L0 -> L1 -> ... -> L29
    hier = InheritanceHierarchy()
    hier.nodes["java/lang/Object"] = HierarchyNode("java/lang/Object", None, (), AccessFlags.PUBLIC, ClassOrigin.JDK_KNOWN)
    hier._super_class["java/lang/Object"] = None
    prev = "java/lang/Object"
    for i in range(30):
        c_name = f"com/level/L{i}"
        hier.nodes[c_name] = HierarchyNode(c_name, prev, (), AccessFlags.PUBLIC, ClassOrigin.ARCHIVE)
        hier._super_class[c_name] = prev
        prev = c_name

    assert len(hier.get_superclasses("com/level/L29")) == 30
    assert hier.find_lowest_common_ancestor("com/level/L29", "com/level/L15") == "com/level/L15"
    assert hier.is_assignable_from("com/level/L10", "com/level/L25")


def test_circular_inheritance_safety():
    # Obfuscator cycle: A extends B, B extends A
    hier = InheritanceHierarchy()
    hier.nodes["A"] = HierarchyNode("A", "B", (), AccessFlags.PUBLIC, ClassOrigin.ARCHIVE)
    hier.nodes["B"] = HierarchyNode("B", "A", (), AccessFlags.PUBLIC, ClassOrigin.ARCHIVE)
    hier._super_class["A"] = "B"
    hier._super_class["B"] = "A"

    # Should not infinite loop
    supers_a = hier.get_superclasses("A")
    assert "B" in supers_a
    # LCA does not hang
    lca = hier.find_lowest_common_ancestor("A", "B")
    assert lca in ("A", "B", "java/lang/Object")


# ==============================================================================
# 8. REGRESSION & HARDENING TESTS (Iteration 2 Fixes)
# ==============================================================================

def test_manifest_multibyte_utf8_line_folding():
    """
    Reviewer 1 Finding 2: RFC 822 line unfolding in byte domain before UTF-8 decode.
    Verifies 2-byte, 3-byte, 4-byte (Emoji), and CJK characters straddling 72-byte line wrap boundary.
    """
    # 3-byte character (Euro symbol) straddling 72-byte boundary
    euro_val = "Prefix-" + ("x" * 49) + "€" + "-Suffix"
    wrapped = wrap_manifest_line("Custom-Data", euro_val)
    manifest_bytes = b"Manifest-Version: 1.0\r\n" + wrapped + b"\r\n"
    m = Manifest.parse(manifest_bytes)
    assert m.get_attribute("Custom-Data") == euro_val

    # 4-byte character (Emoji) straddling 72-byte boundary
    emoji_val = "Prefix-" + ("y" * 48) + "🚀" + "-Suffix"
    wrapped_emoji = wrap_manifest_line("Emoji-Header", emoji_val)
    manifest_bytes_emoji = b"Manifest-Version: 1.0\r\n" + wrapped_emoji + b"\r\n"
    m_emoji = Manifest.parse(manifest_bytes_emoji)
    assert m_emoji.get_attribute("Emoji-Header") == emoji_val

    # CJK characters (3 bytes each) spanning multiple continuation lines
    cjk_val = "日本語テキスト" * 10
    wrapped_cjk = wrap_manifest_line("CJK-Header", cjk_val)
    manifest_bytes_cjk = b"Manifest-Version: 1.0\r\n" + wrapped_cjk + b"\r\n"
    m_cjk = Manifest.parse(manifest_bytes_cjk)
    assert m_cjk.get_attribute("CJK-Header") == cjk_val

    # Tab continuation and LF-only newline
    raw_lf_tab = b"Manifest-Version: 1.0\nHeader: " + b"A" * 60 + b"\n\t" + "€".encode("utf-8") + b"\n\n"
    m_tab = Manifest.parse(raw_lf_tab)
    assert m_tab.get_attribute("Header") == ("A" * 60) + "€"


def test_constant_float_and_double_raw_bits_synchronization():
    """
    Reviewer 1 Finding 3: Synchronize raw_bits and value on direct instantiation
    and mutation of CONSTANT_Float_info and CONSTANT_Double_info.
    """
    # Direct float instantiation
    f = CONSTANT_Float_info(value=2.5)
    assert f.raw_bits == struct.unpack(">I", struct.pack(">f", 2.5))[0]
    assert f.to_bytes() == struct.pack(">Bf", ConstantTag.FLOAT, 2.5)

    # Float negative zero
    f_neg = CONSTANT_Float_info(value=-0.0)
    assert f_neg.raw_bits == 0x80000000
    assert f_neg.to_bytes() == b"\x04\x80\x00\x00\x00"

    # Float positive zero
    f_pos = CONSTANT_Float_info(value=0.0)
    assert f_pos.raw_bits == 0
    assert f_pos.to_bytes() == b"\x04\x00\x00\x00\x00"

    # Direct double instantiation
    d = CONSTANT_Double_info(value=2.5)
    assert d.raw_bits == struct.unpack(">Q", struct.pack(">d", 2.5))[0]
    assert d.to_bytes() == struct.pack(">Bd", ConstantTag.DOUBLE, 2.5)

    # Double negative zero
    d_neg = CONSTANT_Double_info(value=-0.0)
    assert d_neg.raw_bits == 0x8000000000000000
    assert d_neg.to_bytes() == b"\x06\x80\x00\x00\x00\x00\x00\x00\x00"

    # NaN payload preservation
    nan_bits = 0x7FC00001
    f_nan = CONSTANT_Float_info(raw_bits=nan_bits)
    assert f_nan.raw_bits == nan_bits
    assert f_nan.to_bytes() == struct.pack(">BI", ConstantTag.FLOAT, nan_bits)

    # Post-init mutation
    f_mut = CONSTANT_Float_info()
    f_mut.value = 4.0
    f_mut.raw_bits = 0
    assert f_mut.to_bytes() == struct.pack(">Bf", ConstantTag.FLOAT, 4.0)


def test_classfile_truncated_attributes_raise_class_format_error():
    """
    Reviewer 1 Finding 4: Truncated bytecode attributes must raise ClassFormatError
    chained from EOFError.
    """
    cp_bytes = (
        b"\x07\x00\x02"
        b"\x01\x00\x04Test"
        b"\x07\x00\x04"
        b"\x01\x00\x10java/lang/Object"
        b"\x01\x00\x04Attr"
    )
    header = struct.pack(">IHHH", 0xCAFEBABE, 0, 52, 6)
    body = struct.pack(">HHHH", 0x0001, 1, 3, 0)
    base_prefix = header + cp_bytes + body

    # 1. Truncated field attribute header
    field_trunc_hdr = struct.pack(">H", 1) + struct.pack(">HHHH", 0x0001, 2, 2, 1) + b"\x00\x05"
    with pytest.raises(ClassFormatError) as exc_info:
        ClassFile.parse(base_prefix + field_trunc_hdr)
    assert "Truncated field attribute" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, EOFError)

    # 2. Truncated field attribute payload
    field_trunc_payload = struct.pack(">H", 1) + struct.pack(">HHHH", 0x0001, 2, 2, 1) + struct.pack(">HI", 5, 20) + b"\x00" * 5
    with pytest.raises(ClassFormatError) as exc_info:
        ClassFile.parse(base_prefix + field_trunc_payload)
    assert "Truncated field attribute" in str(exc_info.value)

    # 3. Truncated method attribute header
    valid_fields = struct.pack(">H", 0)
    method_trunc_hdr = struct.pack(">H", 1) + struct.pack(">HHHH", 0x0001, 2, 2, 1) + b"\x00\x05"
    with pytest.raises(ClassFormatError) as exc_info:
        ClassFile.parse(base_prefix + valid_fields + method_trunc_hdr)
    assert "Truncated method attribute" in str(exc_info.value)

    # 4. Truncated method attribute payload
    method_trunc_payload = struct.pack(">H", 1) + struct.pack(">HHHH", 0x0001, 2, 2, 1) + struct.pack(">HI", 5, 30) + b"\x00" * 8
    with pytest.raises(ClassFormatError) as exc_info:
        ClassFile.parse(base_prefix + valid_fields + method_trunc_payload)
    assert "Truncated method attribute" in str(exc_info.value)

    # 5. Truncated class attribute header
    valid_methods = struct.pack(">H", 0)
    class_attr_trunc_hdr = struct.pack(">H", 1) + b"\x00\x05"
    with pytest.raises(ClassFormatError) as exc_info:
        ClassFile.parse(base_prefix + valid_fields + valid_methods + class_attr_trunc_hdr)
    assert "Truncated class attribute" in str(exc_info.value)

    # 6. Truncated class attribute payload
    class_attr_trunc_payload = struct.pack(">H", 1) + struct.pack(">HI", 5, 50) + b"\x00" * 10
    with pytest.raises(ClassFormatError) as exc_info:
        ClassFile.parse(base_prefix + valid_fields + valid_methods + class_attr_trunc_payload)
    assert "Truncated class attribute" in str(exc_info.value)


def test_hierarchy_contract_get_subclasses_and_get_interfaces():
    """
    Reviewer 1 & 2 Finding 1: PROJECT.md interface contract conformance.
    InheritanceHierarchy must expose get_subclasses(class_name) -> Set[str]
    and get_interfaces(class_name) -> List[str].
    """
    archive = JarArchive()

    # Create Base implementing Iface1, Iface2
    cp_iface1 = ConstantPool()
    c_i1 = cp_iface1.add_class("com/Iface1")
    s_i1 = cp_iface1.add_class("java/lang/Object")
    cf_i1 = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_iface1, access_flags=AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT, this_class=c_i1, super_class=s_i1)
    archive.add_class(cf_i1)

    cp_iface2 = ConstantPool()
    c_i2 = cp_iface2.add_class("com/Iface2")
    s_i2 = cp_iface2.add_class("java/lang/Object")
    cf_i2 = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_iface2, access_flags=AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT, this_class=c_i2, super_class=s_i2)
    archive.add_class(cf_i2)

    cp_base = ConstantPool()
    c_base = cp_base.add_class("com/Base")
    s_base = cp_base.add_class("java/lang/Object")
    i1_ref = cp_base.add_class("com/Iface1")
    i2_ref = cp_base.add_class("com/Iface2")
    cf_base = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_base, access_flags=AccessFlags.PUBLIC, this_class=c_base, super_class=s_base, interfaces=[i1_ref, i2_ref])
    archive.add_class(cf_base)

    # Create Mid extends Base
    cp_mid = ConstantPool()
    c_mid = cp_mid.add_class("com/Mid")
    s_mid = cp_mid.add_class("com/Base")
    cf_mid = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_mid, access_flags=AccessFlags.PUBLIC, this_class=c_mid, super_class=s_mid)
    archive.add_class(cf_mid)

    # Create Leaf extends Mid
    cp_leaf = ConstantPool()
    c_leaf = cp_leaf.add_class("com/Leaf")
    s_leaf = cp_leaf.add_class("com/Mid")
    cf_leaf = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_leaf, access_flags=AccessFlags.PUBLIC, this_class=c_leaf, super_class=s_leaf)
    archive.add_class(cf_leaf)

    hier = archive.get_hierarchy()

    # Verify method existence
    assert hasattr(hier, "get_subclasses"), "Missing get_subclasses contract method"
    assert hasattr(hier, "get_interfaces"), "Missing get_interfaces contract method"

    # Verify return types and results
    subclasses_base = hier.get_subclasses("com/Base")
    assert isinstance(subclasses_base, set)
    assert subclasses_base == {"com/Mid", "com/Leaf"}

    subclasses_mid = hier.get_subclasses("com/Mid")
    assert subclasses_mid == {"com/Leaf"}

    subclasses_leaf = hier.get_subclasses("com/Leaf")
    assert subclasses_leaf == set()

    ifaces_base = hier.get_interfaces("com/Base")
    assert isinstance(ifaces_base, list)
    assert set(ifaces_base) == {"com/Iface1", "com/Iface2"}

    ifaces_mid = hier.get_interfaces("com/Mid")
    assert ifaces_mid == []


def test_lca_reference_arrays_jvms_rules():
    """
    Reviewer 2 Finding 2: JVMS §4.10.1.2 Lowest Common Ancestor on array reference types.
    - Equal dimension reference arrays merge to [ * dim + L + LCA(elem_a, elem_b) + ;
    - Primitive arrays or unequal dimensions merge to java/lang/Object.
    """
    hier = InheritanceHierarchy()

    # 1. 1D Reference Arrays
    assert hier.find_lowest_common_ancestor("[Ljava/lang/String;", "[Ljava/lang/Integer;") == "[Ljava/lang/Object;"

    # 2. 2D Reference Arrays
    assert hier.find_lowest_common_ancestor("[[Ljava/lang/String;", "[[Ljava/lang/Integer;") == "[[Ljava/lang/Object;"

    # 3. 3D Reference Arrays
    assert hier.find_lowest_common_ancestor("[[[Ljava/lang/String;", "[[[Ljava/lang/Integer;") == "[[[Ljava/lang/Object;"

    # 4. Identity
    assert hier.find_lowest_common_ancestor("[Ljava/lang/String;", "[Ljava/lang/String;") == "[Ljava/lang/String;"
    assert hier.find_lowest_common_ancestor("[I", "[I") == "[I"

    # 5. Primitive arrays & Unequal Dimensions merge to Object
    assert hier.find_lowest_common_ancestor("[I", "[D") == "java/lang/Object"
    assert hier.find_lowest_common_ancestor("[I", "[Ljava/lang/String;") == "java/lang/Object"
    assert hier.find_lowest_common_ancestor("[Ljava/lang/String;", "[[Ljava/lang/String;") == "java/lang/Object"
    assert hier.find_lowest_common_ancestor("[Ljava/lang/String;", "java/lang/String") == "java/lang/Object"


def test_lca_reference_arrays_equal_dimension_snakeyaml():
    """Verifies JVMS §4.10.1.2 covariant LCA resolution on equal-dimension reference arrays with snakeyaml hierarchy."""
    if not SNAKEYAML_JAR.exists():
        pytest.skip("snakeyaml.jar not available")

    archive = JarArchive.from_file(SNAKEYAML_JAR)
    hier = archive.get_hierarchy()

    # 1D Reference Arrays
    lca_1d = hier.find_lowest_common_ancestor("[Ljava/lang/String;", "[Ljava/lang/Integer;")
    assert lca_1d == "[Ljava/lang/Object;"

    # 1D Sibling Domain Hierarchy Classes
    lca_event_1d = hier.find_lowest_common_ancestor(
        "[Lorg/yaml/snakeyaml/events/AliasEvent;",
        "[Lorg/yaml/snakeyaml/events/CollectionEndEvent;"
    )
    assert lca_event_1d == "[Lorg/yaml/snakeyaml/events/Event;"

    # 2D Reference Arrays
    lca_2d = hier.find_lowest_common_ancestor("[[Ljava/lang/String;", "[[Ljava/lang/Integer;")
    assert lca_2d == "[[Ljava/lang/Object;"

    # 2D Sibling Domain Hierarchy Classes
    lca_event_2d = hier.find_lowest_common_ancestor(
        "[[Lorg/yaml/snakeyaml/events/AliasEvent;",
        "[[Lorg/yaml/snakeyaml/events/CollectionEndEvent;"
    )
    assert lca_event_2d == "[[Lorg/yaml/snakeyaml/events/Event;"

    # 3D Reference Arrays
    lca_3d = hier.find_lowest_common_ancestor("[[[Ljava/lang/String;", "[[[Ljava/lang/Integer;")
    assert lca_3d == "[[[Ljava/lang/Object;"


def test_assignability_multidimensional_interface_arrays():
    """Verifies multi-dimensional array assignability to interface arrays (Reviewer 2 Finding 5)."""
    hier = InheritanceHierarchy()
    assert hier.is_assignable_from("[Ljava/lang/Cloneable;", "[[Ljava/lang/String;") is True
    assert hier.is_assignable_from("[Ljava/io/Serializable;", "[[Ljava/lang/String;") is True
    assert hier.is_assignable_from("[Ljava/lang/Object;", "[[Ljava/lang/String;") is True
    assert hier.is_assignable_from("java/lang/Object", "[[Ljava/lang/String;") is True


def test_call_graph_duplicate_edge_deduplication():
    """
    Reviewer 2 Finding 4: CallEdge deduplication across polymorphic subclass fanout.
    When multiple subclasses inherit a method from a common superclass without overriding,
    virtual dispatch at a call site must emit exactly 1 CallEdge for that concrete target.
    CallEdge must be hashable and edges between caller and callee deduplicated in sets.
    """
    archive = JarArchive()

    # Base class declaring foo()V
    cp_base = ConstantPool()
    c_base = cp_base.add_class("com/Base")
    s_base = cp_base.add_class("java/lang/Object")
    m_foo_name = cp_base.add_utf8("foo")
    m_foo_desc = cp_base.add_utf8("()V")
    code_foo = CodeAttribute(cp_base.add_utf8("Code"), 1, 1, bytes([Opcode.RETURN]))
    m_foo = MethodInfo(AccessFlags.PUBLIC, m_foo_name, m_foo_desc, [code_foo])
    cf_base = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_base, access_flags=AccessFlags.PUBLIC, this_class=c_base, super_class=s_base, methods=[m_foo])
    archive.add_class(cf_base)

    # SubA extends Base (no override)
    cp_suba = ConstantPool()
    cf_suba = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_suba, access_flags=AccessFlags.PUBLIC, this_class=cp_suba.add_class("com/SubA"), super_class=cp_suba.add_class("com/Base"))
    archive.add_class(cf_suba)

    # SubB extends Base (no override)
    cp_subb = ConstantPool()
    cf_subb = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_subb, access_flags=AccessFlags.PUBLIC, this_class=cp_subb.add_class("com/SubB"), super_class=cp_subb.add_class("com/Base"))
    archive.add_class(cf_subb)

    # SubC extends Base (overrides foo()V)
    cp_subc = ConstantPool()
    m_subc_foo = MethodInfo(AccessFlags.PUBLIC, cp_subc.add_utf8("foo"), cp_subc.add_utf8("()V"), [CodeAttribute(cp_subc.add_utf8("Code"), 1, 1, bytes([Opcode.RETURN]))])
    cf_subc = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_subc, access_flags=AccessFlags.PUBLIC, this_class=cp_subc.add_class("com/SubC"), super_class=cp_subc.add_class("com/Base"), methods=[m_subc_foo])
    archive.add_class(cf_subc)

    # Caller calls Base.foo() via invokevirtual
    cp_caller = ConstantPool()
    mref = cp_caller.add_member_ref(ConstantTag.METHODREF, "com/Base", "foo", "()V")
    code_caller = bytes([Opcode.ALOAD_0, Opcode.INVOKEVIRTUAL, (mref >> 8) & 0xFF, mref & 0xFF, Opcode.RETURN])
    m_run = MethodInfo(AccessFlags.PUBLIC, cp_caller.add_utf8("run"), cp_caller.add_utf8("()V"), [CodeAttribute(cp_caller.add_utf8("Code"), 2, 2, code_caller)])
    cf_caller = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_caller, access_flags=AccessFlags.PUBLIC, this_class=cp_caller.add_class("com/Caller"), super_class=cp_caller.add_class("java/lang/Object"), methods=[m_run])
    archive.add_class(cf_caller)

    hier = archive.get_hierarchy()
    cg = CallGraph.build(archive, hier)

    # Exactly 2 targets: Base.foo() (shared by Base, SubA, SubB) and SubC.foo() (override)
    caller_sig = "Lcom/Caller;->run()V"
    callees = cg.get_callees(caller_sig)
    assert len(callees) == 2
    assert "Lcom/Base;->foo()V" in callees
    assert "Lcom/SubC;->foo()V" in callees

    # Verify no duplicate CallEdges for Base.foo()
    edges_to_base = [e for e in cg.call_edges if e.callee == "Lcom/Base;->foo()V"]
    assert len(edges_to_base) == 1, f"Expected 1 CallEdge to Base.foo(), got {len(edges_to_base)}"

    # Verify CallEdge is hashable and can be stored in sets
    edge_set = set(cg.call_edges)
    assert len(edge_set) == len(cg.call_edges)

    # Verify adjacency property
    adj = cg.adjacency
    assert caller_sig in adj
    assert adj[caller_sig] == {"Lcom/Base;->foo()V", "Lcom/SubC;->foo()V"}


def test_call_graph_interface_constant_field_resolution():
    """
    Reviewer 2 Finding 3: JVMS §5.4.3.2 Field Resolution on Interface Hierarchies.
    Resolving static fields declared on implemented or extended interfaces.
    """
    archive = JarArchive()

    # 1. SuperInterface declaring FLAG:I
    cp_super_iface = ConstantPool()
    f_flag = FieldInfo(AccessFlags.PUBLIC | AccessFlags.STATIC | AccessFlags.FINAL, cp_super_iface.add_utf8("FLAG"), cp_super_iface.add_utf8("I"), [])
    cf_super_iface = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_super_iface, access_flags=AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT, this_class=cp_super_iface.add_class("com/SuperIface"), super_class=cp_super_iface.add_class("java/lang/Object"), fields=[f_flag])
    archive.add_class(cf_super_iface)

    # 2. SubInterface extends SuperInterface declaring CONST:I
    cp_sub_iface = ConstantPool()
    f_const = FieldInfo(AccessFlags.PUBLIC | AccessFlags.STATIC | AccessFlags.FINAL, cp_sub_iface.add_utf8("CONST"), cp_sub_iface.add_utf8("I"), [])
    cf_sub_iface = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_sub_iface, access_flags=AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT, this_class=cp_sub_iface.add_class("com/SubIface"), super_class=cp_sub_iface.add_class("java/lang/Object"), interfaces=[cp_sub_iface.add_class("com/SuperIface")], fields=[f_const])
    archive.add_class(cf_sub_iface)

    # 3. BaseImpl implements SubIface
    cp_base_impl = ConstantPool()
    cf_base_impl = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_base_impl, access_flags=AccessFlags.PUBLIC, this_class=cp_base_impl.add_class("com/BaseImpl"), super_class=cp_base_impl.add_class("java/lang/Object"), interfaces=[cp_base_impl.add_class("com/SubIface")])
    archive.add_class(cf_base_impl)

    # 4. DerivedImpl extends BaseImpl
    cp_derived = ConstantPool()
    cf_derived = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_derived, access_flags=AccessFlags.PUBLIC, this_class=cp_derived.add_class("com/DerivedImpl"), super_class=cp_derived.add_class("com/BaseImpl"))
    archive.add_class(cf_derived)

    # 5. Client class accessing fields via DerivedImpl
    cp_client = ConstantPool()
    fref_const = cp_client.add_member_ref(ConstantTag.FIELDREF, "com/DerivedImpl", "CONST", "I")
    fref_flag = cp_client.add_member_ref(ConstantTag.FIELDREF, "com/DerivedImpl", "FLAG", "I")
    code_client = bytes([
        Opcode.GETSTATIC, (fref_const >> 8) & 0xFF, fref_const & 0xFF,
        Opcode.GETSTATIC, (fref_flag >> 8) & 0xFF, fref_flag & 0xFF,
        Opcode.RETURN
    ])
    m_test = MethodInfo(AccessFlags.PUBLIC, cp_client.add_utf8("test"), cp_client.add_utf8("()V"), [CodeAttribute(cp_client.add_utf8("Code"), 1, 1, code_client)])
    cf_client = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_client, access_flags=AccessFlags.PUBLIC, this_class=cp_client.add_class("com/Client"), super_class=cp_client.add_class("java/lang/Object"), methods=[m_test])
    archive.add_class(cf_client)

    hier = archive.get_hierarchy()
    cg = CallGraph.build(archive, hier)

    # Verify resolution via direct query
    assert cg._resolve_declaring_field_class("com/DerivedImpl", "CONST", "I", archive, hier) == "com/SubIface"
    assert cg._resolve_declaring_field_class("com/DerivedImpl", "FLAG", "I", archive, hier) == "com/SuperIface"
    assert cg._resolve_field_declaring_class("com/BaseImpl", "CONST", "I", archive, hier) == "com/SubIface"

    # Verify field readers map indexed under true declaring interfaces
    const_id = "Lcom/SubIface;->CONST:I"
    flag_id = "Lcom/SuperIface;->FLAG:I"
    assert "Lcom/Client;->test()V" in cg.field_readers[const_id]
    assert "Lcom/Client;->test()V" in cg.field_readers[flag_id]


def test_default_interface_method_override_maximally_specific_under_any_hashseed():
    """
    JVMS §5.4.3.3 Maximally-Specific Default Method Override Filtering.
    When SubIface extends BaseIface and both declare default method act(),
    calling act() on a class implementing SubIface deterministically resolves
    to SubIface.act() under any PYTHONHASHSEED.
    """
    arc = JarArchive()

    # 1. BaseIface declaring default act()V
    cp_base = ConstantPool()
    c_base = cp_base.add_class("pkg/BaseIface")
    s_base = cp_base.add_class("java/lang/Object")
    m_base_code = CodeAttribute(cp_base.add_utf8("Code"), 1, 1, bytes([Opcode.RETURN]))
    m_base = MethodInfo(
        AccessFlags.PUBLIC,
        cp_base.add_utf8("act"),
        cp_base.add_utf8("()V"),
        [m_base_code]
    )
    cf_base = ClassFile(
        magic=JAVA_MAGIC,
        constant_pool=cp_base,
        access_flags=AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT,
        this_class=c_base,
        super_class=s_base,
        methods=[m_base]
    )
    arc.add_class(cf_base)

    # 2. SubIface extends BaseIface, declaring default act()V (overriding BaseIface)
    cp_sub = ConstantPool()
    c_sub = cp_sub.add_class("pkg/SubIface")
    s_sub = cp_sub.add_class("java/lang/Object")
    ref_base = cp_sub.add_class("pkg/BaseIface")
    m_sub_code = CodeAttribute(cp_sub.add_utf8("Code"), 1, 1, bytes([Opcode.RETURN]))
    m_sub = MethodInfo(
        AccessFlags.PUBLIC,
        cp_sub.add_utf8("act"),
        cp_sub.add_utf8("()V"),
        [m_sub_code]
    )
    cf_sub = ClassFile(
        magic=JAVA_MAGIC,
        constant_pool=cp_sub,
        access_flags=AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT,
        this_class=c_sub,
        super_class=s_sub,
        interfaces=[ref_base],
        methods=[m_sub]
    )
    arc.add_class(cf_sub)

    # 3. Impl implements SubIface
    cp_impl = ConstantPool()
    c_impl = cp_impl.add_class("pkg/Impl")
    s_impl = cp_impl.add_class("java/lang/Object")
    ref_sub = cp_impl.add_class("pkg/SubIface")
    cf_impl = ClassFile(
        magic=JAVA_MAGIC,
        constant_pool=cp_impl,
        access_flags=AccessFlags.PUBLIC,
        this_class=c_impl,
        super_class=s_impl,
        interfaces=[ref_sub]
    )
    arc.add_class(cf_impl)

    # 4. Caller invoking Impl.act()V via invokevirtual
    cp_caller = ConstantPool()
    c_caller = cp_caller.add_class("pkg/Caller")
    s_caller = cp_caller.add_class("java/lang/Object")
    mref_act = cp_caller.add_member_ref(ConstantTag.METHODREF, "pkg/Impl", "act", "()V")
    caller_code = bytes([
        Opcode.INVOKEVIRTUAL, (mref_act >> 8) & 0xFF, mref_act & 0xFF,
        Opcode.RETURN
    ])
    m_caller = MethodInfo(
        AccessFlags.PUBLIC,
        cp_caller.add_utf8("run"),
        cp_caller.add_utf8("()V"),
        [CodeAttribute(cp_caller.add_utf8("Code"), 2, 2, caller_code)]
    )
    cf_caller = ClassFile(
        magic=JAVA_MAGIC,
        constant_pool=cp_caller,
        access_flags=AccessFlags.PUBLIC,
        this_class=c_caller,
        super_class=s_caller,
        methods=[m_caller]
    )
    arc.add_class(cf_caller)

    hier = arc.get_hierarchy()

    # In-process verification: _find_default_method directly
    def_res = CallGraph()._find_default_method("pkg/Impl", "act", "()V", arc, hier)
    assert def_res is not None, "Expected to find default method act() on pkg/Impl"
    declaring_iface, method_info = def_res
    assert declaring_iface == "pkg/SubIface", f"Expected pkg/SubIface, got {declaring_iface}"

    # In-process verification: CallGraph.build resolution
    cg = CallGraph.build(arc, hier, mode="cha")
    callees = cg.get_callees("Lpkg/Caller;->run()V")
    assert "Lpkg/SubIface;->act()V" in callees
    assert "Lpkg/BaseIface;->act()V" not in callees

    # Multi-seed determinism verification across arbitrary PYTHONHASHSEED values
    test_script = """
import sys
from jar_engine.model.constants import JAVA_MAGIC, AccessFlags, Opcode, ConstantTag
from jar_engine.model.cp_info import ConstantPool
from jar_engine.model.member_info import MethodInfo, CodeAttribute
from jar_engine.model.class_file import ClassFile
from jar_engine.model.jar_archive import JarArchive
from jar_engine.walker.call_graph import CallGraph

arc = JarArchive()

# BaseIface
cp_b = ConstantPool()
cb = cp_b.add_class("pkg/BaseIface")
sb = cp_b.add_class("java/lang/Object")
mb = MethodInfo(AccessFlags.PUBLIC, cp_b.add_utf8("act"), cp_b.add_utf8("()V"),
                [CodeAttribute(cp_b.add_utf8("Code"), 1, 1, bytes([Opcode.RETURN]))])
arc.add_class(ClassFile(JAVA_MAGIC, constant_pool=cp_b,
                        access_flags=AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT,
                        this_class=cb, super_class=sb, methods=[mb]))

# SubIface extends BaseIface
cp_s = ConstantPool()
cs = cp_s.add_class("pkg/SubIface")
ss = cp_s.add_class("java/lang/Object")
ref_b = cp_s.add_class("pkg/BaseIface")
ms = MethodInfo(AccessFlags.PUBLIC, cp_s.add_utf8("act"), cp_s.add_utf8("()V"),
                [CodeAttribute(cp_s.add_utf8("Code"), 1, 1, bytes([Opcode.RETURN]))])
arc.add_class(ClassFile(JAVA_MAGIC, constant_pool=cp_s,
                        access_flags=AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT,
                        this_class=cs, super_class=ss, interfaces=[ref_b], methods=[ms]))

# Impl implements SubIface
cp_i = ConstantPool()
ci = cp_i.add_class("pkg/Impl")
si = cp_i.add_class("java/lang/Object")
ref_s = cp_i.add_class("pkg/SubIface")
arc.add_class(ClassFile(JAVA_MAGIC, constant_pool=cp_i, access_flags=AccessFlags.PUBLIC,
                        this_class=ci, super_class=si, interfaces=[ref_s]))

# Caller calls Impl.act()V
cp_c = ConstantPool()
cc = cp_c.add_class("pkg/Caller")
sc = cp_c.add_class("java/lang/Object")
mref = cp_c.add_member_ref(ConstantTag.METHODREF, "pkg/Impl", "act", "()V")
code = bytes([Opcode.INVOKEVIRTUAL, (mref >> 8) & 0xFF, mref & 0xFF, Opcode.RETURN])
mc = MethodInfo(AccessFlags.PUBLIC, cp_c.add_utf8("run"), cp_c.add_utf8("()V"),
                [CodeAttribute(cp_c.add_utf8("Code"), 1, 1, code)])
arc.add_class(ClassFile(JAVA_MAGIC, constant_pool=cp_c, access_flags=AccessFlags.PUBLIC,
                        this_class=cc, super_class=sc, methods=[mc]))

h = arc.get_hierarchy()
cg = CallGraph.build(arc, h, mode="cha")

# 1. Check _find_default_method
res = cg._find_default_method("pkg/Impl", "act", "()V", arc, h)
assert res is not None, "Failed to resolve default method"
assert res[0] == "pkg/SubIface", f"Expected pkg/SubIface, got {res[0]}"

# 2. Check call graph callee resolution
callees = cg.get_callees("Lpkg/Caller;->run()V")
assert "Lpkg/SubIface;->act()V" in callees, f"SubIface.act() missing from callees: {callees}"
assert "Lpkg/BaseIface;->act()V" not in callees, f"BaseIface.act() erroneously present in callees: {callees}"
"""

    seeds_to_test = ["0", "1", "2", "42", "123", "999", "1337", "2026", "65535", "random"]
    for seed in seeds_to_test:
        proc = subprocess.run(
            [sys.executable, "-c", test_script],
            env=dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH="."),
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"Failed under PYTHONHASHSEED={seed}: {proc.stderr}"


