"""
Tier 1: Feature Coverage for F1 - F7 (Archive Model & Whole-Archive Walker).
Tests 35 isolated features (5 per feature F1 through F7) per explorer_e2e_3 specification.
"""

from __future__ import annotations
import struct
import io
import zipfile
from pathlib import Path
from typing import Dict, List, Set, Optional

import pytest

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
    FieldInfo, MethodInfo, CodeAttribute, BootstrapMethod,
    BootstrapMethodsAttribute
)
from jar_engine.model.class_file import ClassFile, ClassFormatError
from jar_engine.model.jar_archive import JarArchive, Manifest
from jar_engine.walker.hierarchy import (
    ClassOrigin, HierarchyNode, InheritanceHierarchy
)
from jar_engine.walker.call_graph import (
    MethodReference, FieldReference, CallEdge, FieldAccessRecord, CallGraph
)
from jar_engine.walker.dependency_slice import ClassDependencyGraph


# ============================================================================
# F1: Class Binary Parser & Serializer (JVMS §4)
# ============================================================================

def test_f01_magic_and_major_minor_versions(make_test_class):
    """F1.1: Parse valid header 0xCAFEBABE, versions (Java 5 through 21), verify version mapping."""
    for major in [49, 50, 51, 52, 53, 55, 61, 65]:
        cf = make_test_class("com/example/VersionTest")
        cf.major_version = major
        cf.minor_version = 0
        raw = cf.to_bytes()
        assert raw[:4] == struct.pack(">I", JAVA_MAGIC)
        parsed = ClassFile.parse(raw)
        assert parsed.magic == JAVA_MAGIC
        assert parsed.major_version == major
        assert parsed.minor_version == 0
        assert JavaMajorVersion(parsed.major_version).value == major


def test_f01_constant_pool_17_tags():
    """F1.2: Verify parsing and serialization of all 17 constant pool tag types."""
    cp = ConstantPool()
    u_idx = cp.add_utf8("Sample")
    i_idx = cp.add_integer(12345)
    f_idx = cp.add_float(3.14)
    l_idx = cp.add_long(9876543210)
    d_idx = cp.add_double(2.71828)
    c_idx = cp.add_class("SampleClass")
    s_idx = cp.add_string("SampleString")
    fr_idx = cp.add_member_ref(ConstantTag.FIELDREF, "SampleClass", "fld", "I")
    mr_idx = cp.add_member_ref(ConstantTag.METHODREF, "SampleClass", "mth", "()V")
    imr_idx = cp.add_member_ref(ConstantTag.INTERFACE_METHODREF, "SampleClass", "imth", "()V")
    nat_idx = cp.add_name_and_type("foo", "()I")

    mh_idx = len(cp._entries)
    cp._entries.append(CONSTANT_MethodHandle_info(reference_kind=ReferenceKind.REF_invokeStatic, reference_index=mr_idx))

    mt_idx = len(cp._entries)
    cp._entries.append(CONSTANT_MethodType_info(descriptor_index=u_idx))

    dyn_idx = len(cp._entries)
    cp._entries.append(CONSTANT_Dynamic_info(bootstrap_method_attr_index=0, name_and_type_index=nat_idx))

    indy_idx = len(cp._entries)
    cp._entries.append(CONSTANT_InvokeDynamic_info(bootstrap_method_attr_index=0, name_and_type_index=nat_idx))

    mod_name_idx = cp.add_utf8("mod")
    mod_idx = len(cp._entries)
    cp._entries.append(CONSTANT_Module_info(name_index=mod_name_idx))

    pkg_name_idx = cp.add_utf8("pkg")
    pkg_idx = len(cp._entries)
    cp._entries.append(CONSTANT_Package_info(name_index=pkg_name_idx))

    # Build ClassFile and roundtrip
    cf = ClassFile(
        magic=JAVA_MAGIC,
        minor_version=0,
        major_version=53,
        constant_pool=cp,
        access_flags=AccessFlags.PUBLIC,
        this_class=c_idx,
        super_class=cp.add_class("java/lang/Object")
    )
    serialized = cf.to_bytes()
    deserialized = ClassFile.parse(serialized)

    assert deserialized.constant_pool[u_idx].value == "Sample"
    assert deserialized.constant_pool[i_idx].value == 12345
    assert deserialized.constant_pool[l_idx].value == 9876543210
    assert deserialized.constant_pool.resolve_utf8(deserialized.constant_pool[c_idx].name_index) == "SampleClass"
    assert isinstance(deserialized.constant_pool[mh_idx], CONSTANT_MethodHandle_info)
    assert isinstance(deserialized.constant_pool[indy_idx], CONSTANT_InvokeDynamic_info)
    assert isinstance(deserialized.constant_pool[mod_idx], CONSTANT_Module_info)
    assert isinstance(deserialized.constant_pool[pkg_idx], CONSTANT_Package_info)


def test_f01_twoslot_long_double_handling():
    """F1.3: Verify 8-byte Long and Double occupy two constant pool slots with placeholder entries."""
    cp = ConstantPool()
    l_idx = cp.add_long(100)
    assert isinstance(cp._entries[l_idx + 1], CpPlaceholder)
    with pytest.raises(ValueError, match="unusable 2nd slot"):
        _ = cp[l_idx + 1]

    d_idx = cp.add_double(200.5)
    assert isinstance(cp._entries[d_idx + 1], CpPlaceholder)
    with pytest.raises(ValueError, match="unusable 2nd slot"):
        _ = cp[d_idx + 1]

    cf = ClassFile(
        magic=JAVA_MAGIC,
        constant_pool=cp,
        this_class=cp.add_class("LongDoubleTest"),
        super_class=cp.add_class("java/lang/Object")
    )
    raw = cf.to_bytes()
    parsed = ClassFile.parse(raw)
    assert parsed.constant_pool[l_idx].value == 100
    assert isinstance(parsed.constant_pool._entries[l_idx + 1], CpPlaceholder)
    with pytest.raises(ValueError, match="unusable 2nd slot"):
        _ = parsed.constant_pool[l_idx + 1]

    assert parsed.constant_pool[d_idx].value == 200.5
    assert isinstance(parsed.constant_pool._entries[d_idx + 1], CpPlaceholder)
    with pytest.raises(ValueError, match="unusable 2nd slot"):
        _ = parsed.constant_pool[d_idx + 1]


def test_f01_modified_utf8_roundtrip():
    """F1.4: Encode/decode null bytes \\x00 -> \\xc0\\x80, multi-byte code points, verify bit-exact string preservation."""
    cases = [
        "plain text 123",
        "embedded\x00null\x00bytes",
        "multibyte: 你好世界 αβγ €100",
        "astral plane: 🚀✨🎉💻"
    ]
    for s in cases:
        encoded = encode_modified_utf8(s)
        if "\x00" in s:
            assert b"\xc0\x80" in encoded
        decoded = decode_modified_utf8(encoded)
        assert decoded == s


def test_f01_idempotent_serialization(make_test_class):
    """F1.5: Verify ClassFile.parse(cf.to_bytes()).to_bytes() == cf.to_bytes()."""
    cf = make_test_class("com/example/Idempotent")
    cp = cf.constant_pool
    # Add a method and field
    f_name = cp.add_utf8("val")
    f_desc = cp.add_utf8("I")
    cf.fields.append(FieldInfo(AccessFlags.PRIVATE, f_name, f_desc, []))

    m_name = cp.add_utf8("getVal")
    m_desc = cp.add_utf8("()I")
    code = bytes([Opcode.ICONST_1, Opcode.IRETURN])
    cf.methods.append(MethodInfo(
        AccessFlags.PUBLIC, m_name, m_desc,
        [CodeAttribute(cp.add_utf8("Code"), 1, 1, code)]
    ))

    raw1 = cf.to_bytes()
    cf2 = ClassFile.parse(raw1)
    raw2 = cf2.to_bytes()
    cf3 = ClassFile.parse(raw2)
    raw3 = cf3.to_bytes()

    assert raw1 == raw2
    assert raw2 == raw3


# ============================================================================
# F2: Archive Symbol Indexer
# ============================================================================

def test_f02_jar_archive_loading(make_test_archive, make_test_class, temp_workspace):
    """F2.1: Ingest multi-class JAR, verify all classes loaded into archive.classes."""
    jar_path = temp_workspace / "multi.jar"
    classes = [
        make_test_class("pkg/A"),
        make_test_class("pkg/B"),
        make_test_class("pkg/C")
    ]
    arc = make_test_archive(classes)
    arc.save(jar_path)

    loaded = JarArchive.from_file(jar_path)
    assert len(loaded.classes) == 3
    assert set(loaded.classes.keys()) == {"pkg/A", "pkg/B", "pkg/C"}


def test_f02_manifest_rfc822_parsing(temp_workspace):
    """F2.2: Extract Main-Class and manifest attributes per RFC 822 line specifications."""
    buf = io.BytesIO()
    manifest_bytes = (
        b"Manifest-Version: 1.0\r\n"
        b"Main-Class: com.example.MainApp\r\n"
        b"X-Custom-Attribute: VeryLongValueThatWillBeTestedForRFC822LineLengthRules\r\n"
        b"\r\n"
    )
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("META-INF/MANIFEST.MF", manifest_bytes)
    
    jar_path = temp_workspace / "manifest_test.jar"
    jar_path.write_bytes(buf.getvalue())

    loaded = JarArchive.from_file(jar_path)
    assert loaded.manifest is not None
    assert loaded.manifest.get_main_class() == "com.example.MainApp"
    assert loaded.manifest.main_attributes.get("X-Custom-Attribute") == "VeryLongValueThatWillBeTestedForRFC822LineLengthRules"


def test_f02_resource_preservation(make_test_class, temp_workspace):
    """F2.3: Verify non-class files in JAR (.properties, .xml, binary assets) preserved verbatim."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("config.properties", b"key=value\nnumber=42")
        zf.writestr("assets/data.bin", bytes(range(256)))
        zf.writestr("pkg/Foo.class", make_test_class("pkg/Foo").to_bytes())
    
    in_jar = temp_workspace / "in_res.jar"
    in_jar.write_bytes(buf.getvalue())

    arc = JarArchive.from_file(in_jar)
    assert arc.get_resource("config.properties") == b"key=value\nnumber=42"
    assert arc.get_resource("assets/data.bin") == bytes(range(256))

    out_jar = temp_workspace / "out_res.jar"
    arc.save(out_jar)
    reloaded = JarArchive.from_file(out_jar)
    assert reloaded.get_resource("config.properties") == b"key=value\nnumber=42"
    assert reloaded.get_resource("assets/data.bin") == bytes(range(256))


def test_f02_symbol_indexing_members(make_test_class):
    """F2.4: Verify member signatures and access flags indexed across all archive classes."""
    arc = JarArchive()
    cf = make_test_class("com/example/Indexed")
    cp = cf.constant_pool
    cf.fields.append(FieldInfo(
        AccessFlags.PUBLIC | AccessFlags.STATIC,
        cp.add_utf8("COUNTER"), cp.add_utf8("I"), []
    ))
    cf.methods.append(MethodInfo(
        AccessFlags.PUBLIC,
        cp.add_utf8("compute"), cp.add_utf8("(I)I"), []
    ))
    arc.add_class(cf)

    indexed_cf = arc.get_class("com/example/Indexed")
    assert indexed_cf is not None
    field = indexed_cf.get_field("COUNTER")
    assert field is not None
    assert field.access_flags & AccessFlags.STATIC
    assert field.descriptor(indexed_cf.constant_pool) == "I"

    method = indexed_cf.get_method("compute", "(I)I")
    assert method is not None
    assert method.access_flags & AccessFlags.PUBLIC
    assert method.descriptor(indexed_cf.constant_pool) == "(I)I"


def test_f02_signature_stripping(make_test_class, temp_workspace):
    """F2.5: Verify cryptographic signature files (META-INF/*.SF, *.RSA) stripped."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("pkg/A.class", make_test_class("pkg/A").to_bytes())
        zf.writestr("META-INF/SIGNER.SF", b"Signature-Version: 1.0\r\n")
        zf.writestr("META-INF/SIGNER.RSA", b"FAKE-RSA-BYTES")
        zf.writestr("META-INF/SIGNER.DSA", b"FAKE-DSA-BYTES")
    
    in_jar = temp_workspace / "signed.jar"
    in_jar.write_bytes(buf.getvalue())

    arc = JarArchive.from_file(in_jar)
    assert arc.has_resource("META-INF/SIGNER.SF")
    stripped_count = arc.strip_signatures()
    assert stripped_count >= 3
    assert not arc.has_resource("META-INF/SIGNER.SF")

    out_jar = temp_workspace / "stripped.jar"
    arc.save(out_jar)

    with zipfile.ZipFile(out_jar, "r") as zf:
        names = zf.namelist()
        assert not any(n.startswith("META-INF/") and n.endswith((".SF", ".RSA", ".DSA")) for n in names)


# ============================================================================
# F3: Inheritance Hierarchy DAG
# ============================================================================

def test_f03_superclasses_chain(make_test_class):
    """F3.1: Trace class hierarchy up to java/lang/Object."""
    arc = JarArchive()
    arc.add_class(make_test_class("pkg/A", super_name="java/lang/Object"))
    arc.add_class(make_test_class("pkg/B", super_name="pkg/A"))
    arc.add_class(make_test_class("pkg/C", super_name="pkg/B"))

    hier = arc.get_hierarchy()
    supers = hier.get_superclasses("pkg/C")
    assert supers == ["pkg/B", "pkg/A", "java/lang/Object"]


def test_f03_subclasses_discovery(make_test_class):
    """F3.2: Query all direct and transitive subclasses of base classes."""
    arc = JarArchive()
    arc.add_class(make_test_class("pkg/Base", super_name="java/lang/Object"))
    arc.add_class(make_test_class("pkg/Sub1", super_name="pkg/Base"))
    arc.add_class(make_test_class("pkg/Sub2", super_name="pkg/Base"))
    arc.add_class(make_test_class("pkg/SubSub1", super_name="pkg/Sub1"))

    hier = arc.get_hierarchy()
    subs = hier.get_subclasses("pkg/Base")
    assert "pkg/Sub1" in subs
    assert "pkg/Sub2" in subs
    assert "pkg/SubSub1" in subs


def test_f03_interfaces_transitive(make_test_class):
    """F3.3: Query direct and inherited interface implementations."""
    arc = JarArchive()
    arc.add_class(make_test_class("pkg/I1", access_flags=AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT))
    arc.add_class(make_test_class("pkg/I2", interfaces=["pkg/I1"], access_flags=AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT))
    arc.add_class(make_test_class("pkg/Base", interfaces=["pkg/I2"]))
    arc.add_class(make_test_class("pkg/Impl", super_name="pkg/Base"))

    hier = arc.get_hierarchy()
    direct_base_ifaces = hier.get_interfaces("pkg/Base")
    assert direct_base_ifaces == ["pkg/I2"]

    all_impl_ifaces = hier.get_all_interfaces("pkg/Impl")
    assert "pkg/I1" in all_impl_ifaces
    assert "pkg/I2" in all_impl_ifaces


def test_f03_lca_class_resolution(make_test_class):
    """F3.4: Lowest Common Ancestor calculation conforming to JVMS §4.10."""
    arc = JarArchive()
    arc.add_class(make_test_class("pkg/Root", super_name="java/lang/Object"))
    arc.add_class(make_test_class("pkg/Left", super_name="pkg/Root"))
    arc.add_class(make_test_class("pkg/Right", super_name="pkg/Root"))
    arc.add_class(make_test_class("pkg/LeftLeaf", super_name="pkg/Left"))

    hier = arc.get_hierarchy()
    assert hier.find_lowest_common_ancestor("pkg/LeftLeaf", "pkg/Right") == "pkg/Root"
    assert hier.find_lowest_common_ancestor("pkg/LeftLeaf", "pkg/Left") == "pkg/Left"


def test_f03_lca_interface_object_merge(make_test_class):
    """F3.5: Interface-to-interface or class-to-interface LCA correctly resolves to java/lang/Object."""
    arc = JarArchive()
    arc.add_class(make_test_class("pkg/IfaceA", access_flags=AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT))
    arc.add_class(make_test_class("pkg/IfaceB", access_flags=AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT))
    arc.add_class(make_test_class("pkg/Concrete", interfaces=["pkg/IfaceA"]))

    hier = arc.get_hierarchy()
    # Interface to interface merges to Object per JVMS §4.10
    assert hier.find_lowest_common_ancestor("pkg/IfaceA", "pkg/IfaceB") == "java/lang/Object"
    # Class to unrelated interface merges to Object
    assert hier.find_lowest_common_ancestor("pkg/Concrete", "pkg/IfaceB") == "java/lang/Object"


# ============================================================================
# F4: Polymorphic Call Graph Walker
# ============================================================================

def test_f04_invokevirtual_cha_resolution():
    """F4.1: Dispatch virtual call to all possible overriding implementations."""
    arc = JarArchive()
    # Base with m()V
    cp_base = ConstantPool()
    c_base = cp_base.add_class("pkg/Base")
    s_base = cp_base.add_class("java/lang/Object")
    m_base = MethodInfo(AccessFlags.PUBLIC, cp_base.add_utf8("m"), cp_base.add_utf8("()V"), [
        CodeAttribute(cp_base.add_utf8("Code"), 1, 1, bytes([Opcode.RETURN]))
    ])
    cf_base = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_base, access_flags=AccessFlags.PUBLIC, this_class=c_base, super_class=s_base, methods=[m_base])
    arc.add_class(cf_base)

    # Sub with override m()V
    cp_sub = ConstantPool()
    c_sub = cp_sub.add_class("pkg/Sub")
    s_sub = cp_sub.add_class("pkg/Base")
    m_sub = MethodInfo(AccessFlags.PUBLIC, cp_sub.add_utf8("m"), cp_sub.add_utf8("()V"), [
        CodeAttribute(cp_sub.add_utf8("Code"), 1, 1, bytes([Opcode.RETURN]))
    ])
    cf_sub = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_sub, access_flags=AccessFlags.PUBLIC, this_class=c_sub, super_class=s_sub, methods=[m_sub])
    arc.add_class(cf_sub)

    # Caller invoking Base.m()V
    cp_call = ConstantPool()
    c_call = cp_call.add_class("pkg/Caller")
    mref = cp_call.add_member_ref(ConstantTag.METHODREF, "pkg/Base", "m", "()V")
    code = bytes([Opcode.ALOAD_0, Opcode.INVOKEVIRTUAL, (mref >> 8) & 0xFF, mref & 0xFF, Opcode.RETURN])
    m_call = MethodInfo(AccessFlags.PUBLIC, cp_call.add_utf8("call"), cp_call.add_utf8("()V"), [
        CodeAttribute(cp_call.add_utf8("Code"), 1, 1, code)
    ])
    cf_call = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_call, access_flags=AccessFlags.PUBLIC, this_class=c_call, super_class=cp_call.add_class("java/lang/Object"), methods=[m_call])
    arc.add_class(cf_call)

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier, mode="cha")
    callees = cg.get_callees("Lpkg/Caller;->call()V")
    assert "Lpkg/Base;->m()V" in callees
    assert "Lpkg/Sub;->m()V" in callees


def test_f04_invokeinterface_implementors():
    """F4.2: Resolve interface calls across all implementors in archive."""
    arc = JarArchive()
    # Interface
    cp_i = ConstantPool()
    cf_i = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_i, access_flags=AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT,
                     this_class=cp_i.add_class("pkg/IFoo"), super_class=cp_i.add_class("java/lang/Object"),
                     methods=[MethodInfo(AccessFlags.PUBLIC | AccessFlags.ABSTRACT, cp_i.add_utf8("run"), cp_i.add_utf8("()V"), [])])
    arc.add_class(cf_i)

    # ImplA
    cp_a = ConstantPool()
    m_a = MethodInfo(AccessFlags.PUBLIC, cp_a.add_utf8("run"), cp_a.add_utf8("()V"), [CodeAttribute(cp_a.add_utf8("Code"), 1, 1, bytes([Opcode.RETURN]))])
    cf_a = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_a, access_flags=AccessFlags.PUBLIC, this_class=cp_a.add_class("pkg/ImplA"), super_class=cp_a.add_class("java/lang/Object"), interfaces=[cp_a.add_class("pkg/IFoo")], methods=[m_a])
    arc.add_class(cf_a)

    # ImplB
    cp_b = ConstantPool()
    m_b = MethodInfo(AccessFlags.PUBLIC, cp_b.add_utf8("run"), cp_b.add_utf8("()V"), [CodeAttribute(cp_b.add_utf8("Code"), 1, 1, bytes([Opcode.RETURN]))])
    cf_b = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_b, access_flags=AccessFlags.PUBLIC, this_class=cp_b.add_class("pkg/ImplB"), super_class=cp_b.add_class("java/lang/Object"), interfaces=[cp_b.add_class("pkg/IFoo")], methods=[m_b])
    arc.add_class(cf_b)

    # Caller invoking IFoo.run()
    cp_c = ConstantPool()
    imref = cp_c.add_member_ref(ConstantTag.INTERFACE_METHODREF, "pkg/IFoo", "run", "()V")
    code_c = bytes([Opcode.ALOAD_0, Opcode.INVOKEINTERFACE, (imref >> 8) & 0xFF, imref & 0xFF, 1, 0, Opcode.RETURN])
    m_c = MethodInfo(AccessFlags.PUBLIC, cp_c.add_utf8("doRun"), cp_c.add_utf8("()V"), [CodeAttribute(cp_c.add_utf8("Code"), 1, 1, code_c)])
    cf_c = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_c, access_flags=AccessFlags.PUBLIC, this_class=cp_c.add_class("pkg/Caller"), super_class=cp_c.add_class("java/lang/Object"), methods=[m_c])
    arc.add_class(cf_c)

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier, mode="cha")
    callees = cg.get_callees("Lpkg/Caller;->doRun()V")
    assert "Lpkg/ImplA;->run()V" in callees
    assert "Lpkg/ImplB;->run()V" in callees


def test_f04_default_methods_maximally_specific():
    """F4.3: Resolve Java 8+ default methods choosing maximally specific interface per JVMS §5.4.3.3."""
    arc = JarArchive()
    # BaseIface declaring default act()V
    cp_base = ConstantPool()
    m_base = MethodInfo(AccessFlags.PUBLIC, cp_base.add_utf8("act"), cp_base.add_utf8("()V"), [CodeAttribute(cp_base.add_utf8("Code"), 1, 1, bytes([Opcode.RETURN]))])
    cf_base = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_base, access_flags=AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT,
                        this_class=cp_base.add_class("pkg/BaseIface"), super_class=cp_base.add_class("java/lang/Object"), methods=[m_base])
    arc.add_class(cf_base)

    # SubIface extends BaseIface overriding act()V
    cp_sub = ConstantPool()
    m_sub = MethodInfo(AccessFlags.PUBLIC, cp_sub.add_utf8("act"), cp_sub.add_utf8("()V"), [CodeAttribute(cp_sub.add_utf8("Code"), 1, 1, bytes([Opcode.RETURN]))])
    cf_sub = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_sub, access_flags=AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT,
                       this_class=cp_sub.add_class("pkg/SubIface"), super_class=cp_sub.add_class("java/lang/Object"),
                       interfaces=[cp_sub.add_class("pkg/BaseIface")], methods=[m_sub])
    arc.add_class(cf_sub)

    # Impl implements SubIface without overriding act()V
    cp_impl = ConstantPool()
    cf_impl = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_impl, access_flags=AccessFlags.PUBLIC,
                        this_class=cp_impl.add_class("pkg/Impl"), super_class=cp_impl.add_class("java/lang/Object"),
                        interfaces=[cp_impl.add_class("pkg/SubIface")])
    arc.add_class(cf_impl)

    hier = arc.get_hierarchy()
    def_res = CallGraph()._find_default_method("pkg/Impl", "act", "()V", arc, hier)
    assert def_res is not None
    declaring_iface, _ = def_res
    assert declaring_iface == "pkg/SubIface"


def test_f04_rta_mode_filtering():
    """F4.4: Verify RTA mode restricts dispatch targets to instantiated classes."""
    arc = JarArchive()
    # Base class
    cp_b = ConstantPool()
    m_b = MethodInfo(AccessFlags.PUBLIC, cp_b.add_utf8("m"), cp_b.add_utf8("()V"), [CodeAttribute(cp_b.add_utf8("Code"), 1, 1, bytes([Opcode.RETURN]))])
    cf_b = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_b, access_flags=AccessFlags.PUBLIC, this_class=cp_b.add_class("pkg/Base"), super_class=cp_b.add_class("java/lang/Object"), methods=[m_b])
    arc.add_class(cf_b)

    # Sub1 (instantiated)
    cp_s1 = ConstantPool()
    m_s1 = MethodInfo(AccessFlags.PUBLIC, cp_s1.add_utf8("m"), cp_s1.add_utf8("()V"), [CodeAttribute(cp_s1.add_utf8("Code"), 1, 1, bytes([Opcode.RETURN]))])
    cf_s1 = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_s1, access_flags=AccessFlags.PUBLIC, this_class=cp_s1.add_class("pkg/Sub1"), super_class=cp_s1.add_class("pkg/Base"), methods=[m_s1])
    arc.add_class(cf_s1)

    # Sub2 (never instantiated with new)
    cp_s2 = ConstantPool()
    m_s2 = MethodInfo(AccessFlags.PUBLIC, cp_s2.add_utf8("m"), cp_s2.add_utf8("()V"), [CodeAttribute(cp_s2.add_utf8("Code"), 1, 1, bytes([Opcode.RETURN]))])
    cf_s2 = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_s2, access_flags=AccessFlags.PUBLIC, this_class=cp_s2.add_class("pkg/Sub2"), super_class=cp_s2.add_class("pkg/Base"), methods=[m_s2])
    arc.add_class(cf_s2)

    # Main: news Sub1, then calls Base.m()
    cp_m = ConstantPool()
    c_sub1 = cp_m.add_class("pkg/Sub1")
    init_ref = cp_m.add_member_ref(ConstantTag.METHODREF, "pkg/Sub1", "<init>", "()V")
    mref = cp_m.add_member_ref(ConstantTag.METHODREF, "pkg/Base", "m", "()V")
    code = bytes([
        Opcode.NEW, (c_sub1 >> 8) & 0xFF, c_sub1 & 0xFF,
        Opcode.DUP, Opcode.INVOKESPECIAL, (init_ref >> 8) & 0xFF, init_ref & 0xFF,
        Opcode.INVOKEVIRTUAL, (mref >> 8) & 0xFF, mref & 0xFF,
        Opcode.RETURN
    ])
    m_main = MethodInfo(AccessFlags.PUBLIC | AccessFlags.STATIC, cp_m.add_utf8("main"), cp_m.add_utf8("()V"), [
        CodeAttribute(cp_m.add_utf8("Code"), 2, 1, code)
    ])
    cf_m = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_m, access_flags=AccessFlags.PUBLIC, this_class=cp_m.add_class("pkg/Main"), super_class=cp_m.add_class("java/lang/Object"), methods=[m_main])
    arc.add_class(cf_m)

    hier = arc.get_hierarchy()
    cg_rta = CallGraph.build(arc, hier, mode="rta")
    callees = cg_rta.get_callees("Lpkg/Main;->main()V")
    # Sub1 was instantiated with NEW, so Sub1.m is in callees, but Sub2.m is NOT
    assert "Lpkg/Sub1;->m()V" in callees
    assert "Lpkg/Sub2;->m()V" not in callees


def test_f04_call_graph_callers_callees():
    """F4.5: Verify bidirectional indexing (get_callers, get_callees)."""
    arc = JarArchive()
    # Method A calls Method B
    cp_a = ConstantPool()
    mref_b = cp_a.add_member_ref(ConstantTag.METHODREF, "pkg/Target", "targetMethod", "()V")
    code_a = bytes([Opcode.INVOKESTATIC, (mref_b >> 8) & 0xFF, mref_b & 0xFF, Opcode.RETURN])
    m_a = MethodInfo(AccessFlags.PUBLIC | AccessFlags.STATIC, cp_a.add_utf8("callA"), cp_a.add_utf8("()V"), [
        CodeAttribute(cp_a.add_utf8("Code"), 1, 1, code_a)
    ])
    cf_a = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_a, access_flags=AccessFlags.PUBLIC, this_class=cp_a.add_class("pkg/CallerClass"), super_class=cp_a.add_class("java/lang/Object"), methods=[m_a])
    arc.add_class(cf_a)

    cp_t = ConstantPool()
    m_t = MethodInfo(AccessFlags.PUBLIC | AccessFlags.STATIC, cp_t.add_utf8("targetMethod"), cp_t.add_utf8("()V"), [
        CodeAttribute(cp_t.add_utf8("Code"), 1, 1, bytes([Opcode.RETURN]))
    ])
    cf_t = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_t, access_flags=AccessFlags.PUBLIC, this_class=cp_t.add_class("pkg/Target"), super_class=cp_t.add_class("java/lang/Object"), methods=[m_t])
    arc.add_class(cf_t)

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)
    assert "Lpkg/Target;->targetMethod()V" in cg.get_callees("Lpkg/CallerClass;->callA()V")
    assert "Lpkg/CallerClass;->callA()V" in cg.get_callers("Lpkg/Target;->targetMethod()V")


# ============================================================================
# F5: Static & Special Dispatch Resolver
# ============================================================================

def test_f05_invokestatic_direct_target():
    """F5.1: Direct resolution of static methods to declaring classes."""
    arc = JarArchive()
    cp_util = ConstantPool()
    m_util = MethodInfo(AccessFlags.PUBLIC | AccessFlags.STATIC, cp_util.add_utf8("helper"), cp_util.add_utf8("(I)I"), [
        CodeAttribute(cp_util.add_utf8("Code"), 1, 1, bytes([Opcode.ILOAD_0, Opcode.IRETURN]))
    ])
    cf_util = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_util, access_flags=AccessFlags.PUBLIC, this_class=cp_util.add_class("pkg/Util"), super_class=cp_util.add_class("java/lang/Object"), methods=[m_util])
    arc.add_class(cf_util)

    cp_client = ConstantPool()
    mref = cp_client.add_member_ref(ConstantTag.METHODREF, "pkg/Util", "helper", "(I)I")
    code = bytes([Opcode.ICONST_5, Opcode.INVOKESTATIC, (mref >> 8) & 0xFF, mref & 0xFF, Opcode.IRETURN])
    m_client = MethodInfo(AccessFlags.PUBLIC | AccessFlags.STATIC, cp_client.add_utf8("run"), cp_client.add_utf8("()I"), [
        CodeAttribute(cp_client.add_utf8("Code"), 2, 1, code)
    ])
    cf_client = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_client, access_flags=AccessFlags.PUBLIC, this_class=cp_client.add_class("pkg/Client"), super_class=cp_client.add_class("java/lang/Object"), methods=[m_client])
    arc.add_class(cf_client)

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)
    callees = cg.get_callees("Lpkg/Client;->run()I")
    assert "Lpkg/Util;->helper(I)I" in callees


def test_f05_invokespecial_init_constructor():
    """F5.2: Resolution of <init> constructor calls."""
    arc = JarArchive()
    cp_item = ConstantPool()
    m_init = MethodInfo(AccessFlags.PUBLIC, cp_item.add_utf8("<init>"), cp_item.add_utf8("()V"), [
        CodeAttribute(cp_item.add_utf8("Code"), 1, 1, bytes([Opcode.ALOAD_0, Opcode.RETURN]))
    ])
    cf_item = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_item, access_flags=AccessFlags.PUBLIC, this_class=cp_item.add_class("pkg/Item"), super_class=cp_item.add_class("java/lang/Object"), methods=[m_init])
    arc.add_class(cf_item)

    cp_f = ConstantPool()
    c_item = cp_f.add_class("pkg/Item")
    init_ref = cp_f.add_member_ref(ConstantTag.METHODREF, "pkg/Item", "<init>", "()V")
    code = bytes([
        Opcode.NEW, (c_item >> 8) & 0xFF, c_item & 0xFF,
        Opcode.DUP, Opcode.INVOKESPECIAL, (init_ref >> 8) & 0xFF, init_ref & 0xFF,
        Opcode.POP, Opcode.RETURN
    ])
    m_create = MethodInfo(AccessFlags.PUBLIC | AccessFlags.STATIC, cp_f.add_utf8("create"), cp_f.add_utf8("()V"), [
        CodeAttribute(cp_f.add_utf8("Code"), 2, 1, code)
    ])
    cf_f = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_f, access_flags=AccessFlags.PUBLIC, this_class=cp_f.add_class("pkg/Factory"), super_class=cp_f.add_class("java/lang/Object"), methods=[m_create])
    arc.add_class(cf_f)

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)
    callees = cg.get_callees("Lpkg/Factory;->create()V")
    assert "Lpkg/Item;-><init>()V" in callees


def test_f05_invokespecial_super_call():
    """F5.3: Resolution of super.method() starting from direct superclass."""
    arc = JarArchive()
    # Parent
    cp_p = ConstantPool()
    m_p = MethodInfo(AccessFlags.PUBLIC, cp_p.add_utf8("doWork"), cp_p.add_utf8("()V"), [
        CodeAttribute(cp_p.add_utf8("Code"), 1, 1, bytes([Opcode.RETURN]))
    ])
    cf_p = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_p, access_flags=AccessFlags.PUBLIC, this_class=cp_p.add_class("pkg/Parent"), super_class=cp_p.add_class("java/lang/Object"), methods=[m_p])
    arc.add_class(cf_p)

    # Child invokesuper Parent.doWork()
    cp_c = ConstantPool()
    mref_super = cp_c.add_member_ref(ConstantTag.METHODREF, "pkg/Parent", "doWork", "()V")
    code_c = bytes([Opcode.ALOAD_0, Opcode.INVOKESPECIAL, (mref_super >> 8) & 0xFF, mref_super & 0xFF, Opcode.RETURN])
    m_c = MethodInfo(AccessFlags.PUBLIC, cp_c.add_utf8("doWork"), cp_c.add_utf8("()V"), [
        CodeAttribute(cp_c.add_utf8("Code"), 1, 1, code_c)
    ])
    cf_c = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_c, access_flags=AccessFlags.PUBLIC, this_class=cp_c.add_class("pkg/Child"), super_class=cp_c.add_class("pkg/Parent"), methods=[m_c])
    arc.add_class(cf_c)

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)
    callees = cg.get_callees("Lpkg/Child;->doWork()V")
    assert "Lpkg/Parent;->doWork()V" in callees


def test_f05_invokespecial_private_method():
    """F5.4: Resolution of private instance method invocations."""
    arc = JarArchive()
    cp = ConstantPool()
    m_priv = MethodInfo(AccessFlags.PRIVATE, cp.add_utf8("secret"), cp.add_utf8("()V"), [
        CodeAttribute(cp.add_utf8("Code"), 1, 1, bytes([Opcode.RETURN]))
    ])
    mref_priv = cp.add_member_ref(ConstantTag.METHODREF, "pkg/Vault", "secret", "()V")
    code_pub = bytes([Opcode.ALOAD_0, Opcode.INVOKESPECIAL, (mref_priv >> 8) & 0xFF, mref_priv & 0xFF, Opcode.RETURN])
    m_pub = MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("unlock"), cp.add_utf8("()V"), [
        CodeAttribute(cp.add_utf8("Code"), 1, 1, code_pub)
    ])
    cf = ClassFile(magic=JAVA_MAGIC, constant_pool=cp, access_flags=AccessFlags.PUBLIC, this_class=cp.add_class("pkg/Vault"), super_class=cp.add_class("java/lang/Object"), methods=[m_priv, m_pub])
    arc.add_class(cf)

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)
    callees = cg.get_callees("Lpkg/Vault;->unlock()V")
    assert "Lpkg/Vault;->secret()V" in callees


def test_f05_invokedynamic_lambda_target(fixture_jars):
    """F5.5: Extract lambda implementation target from BootstrapMethods attribute."""
    # Use synthetic constant_obfuscated_jar which contains DynamicInvoker with LambdaMetafactory
    arc = JarArchive.from_file(fixture_jars["constant_obfuscated"])
    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)

    # In DynamicInvoker, getWorker() invokes LambdaMetafactory pointing to workerAction()V
    callees = cg.get_callees("Lcom/synthetic/constants/DynamicInvoker;->getWorker()Ljava/lang/Runnable;")
    assert "Lcom/synthetic/constants/DynamicInvoker;->workerAction()V" in callees


# ============================================================================
# F6: Field Access Mapping
# ============================================================================

def test_f06_getstatic_putstatic_records():
    """F6.1: Record static field reads and writes with bytecode offsets."""
    arc = JarArchive()
    cp = ConstantPool()
    fref = cp.add_member_ref(ConstantTag.FIELDREF, "pkg/Data", "COUNT", "I")
    code = bytes([
        Opcode.ICONST_5,
        Opcode.PUTSTATIC, (fref >> 8) & 0xFF, fref & 0xFF,  # PC 1
        Opcode.GETSTATIC, (fref >> 8) & 0xFF, fref & 0xFF,  # PC 4
        Opcode.POP,
        Opcode.RETURN
    ])
    m = MethodInfo(AccessFlags.PUBLIC | AccessFlags.STATIC, cp.add_utf8("test"), cp.add_utf8("()V"), [
        CodeAttribute(cp.add_utf8("Code"), 1, 1, code)
    ])
    f = FieldInfo(AccessFlags.PUBLIC | AccessFlags.STATIC, cp.add_utf8("COUNT"), cp.add_utf8("I"), [])
    cf = ClassFile(magic=JAVA_MAGIC, constant_pool=cp, access_flags=AccessFlags.PUBLIC, this_class=cp.add_class("pkg/Data"), super_class=cp.add_class("java/lang/Object"), fields=[f], methods=[m])
    arc.add_class(cf)

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)
    records = cg.field_accesses

    put_recs = [r for r in records if r.access_type == "write"]
    get_recs = [r for r in records if r.access_type == "read"]

    assert len(put_recs) == 1
    assert put_recs[0].pc == 1
    assert put_recs[0].field == "Lpkg/Data;->COUNT:I"

    assert len(get_recs) == 1
    assert get_recs[0].pc == 4
    assert get_recs[0].field == "Lpkg/Data;->COUNT:I"


def test_f06_getfield_putfield_records():
    """F6.2: Record instance field reads and writes."""
    arc = JarArchive()
    cp = ConstantPool()
    fref = cp.add_member_ref(ConstantTag.FIELDREF, "pkg/Entity", "id", "I")
    code = bytes([
        Opcode.ALOAD_0,
        Opcode.BIPUSH, 42,
        Opcode.PUTFIELD, (fref >> 8) & 0xFF, fref & 0xFF,
        Opcode.ALOAD_0,
        Opcode.GETFIELD, (fref >> 8) & 0xFF, fref & 0xFF,
        Opcode.IRETURN
    ])
    m = MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("update"), cp.add_utf8("()I"), [
        CodeAttribute(cp.add_utf8("Code"), 2, 1, code)
    ])
    f = FieldInfo(AccessFlags.PRIVATE, cp.add_utf8("id"), cp.add_utf8("I"), [])
    cf = ClassFile(magic=JAVA_MAGIC, constant_pool=cp, access_flags=AccessFlags.PUBLIC, this_class=cp.add_class("pkg/Entity"), super_class=cp.add_class("java/lang/Object"), fields=[f], methods=[m])
    arc.add_class(cf)

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)
    fid = "Lpkg/Entity;->id:I"
    assert "Lpkg/Entity;->update()I" in cg.field_writers[fid]
    assert "Lpkg/Entity;->update()I" in cg.field_readers[fid]


def test_f06_declaring_class_resolution():
    """F6.3: Walk superclass/interface hierarchy to find declaring class of inherited field."""
    arc = JarArchive()
    # Super declaring field
    cp_p = ConstantPool()
    f_p = FieldInfo(AccessFlags.PUBLIC, cp_p.add_utf8("name"), cp_p.add_utf8("Ljava/lang/String;"), [])
    cf_p = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_p, access_flags=AccessFlags.PUBLIC, this_class=cp_p.add_class("pkg/ParentClass"), super_class=cp_p.add_class("java/lang/Object"), fields=[f_p])
    arc.add_class(cf_p)

    # Subclass does not declare field
    cp_c = ConstantPool()
    cf_c = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_c, access_flags=AccessFlags.PUBLIC, this_class=cp_c.add_class("pkg/ChildClass"), super_class=cp_c.add_class("pkg/ParentClass"))
    arc.add_class(cf_c)

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)
    declaring = cg._resolve_field_declaring_class("pkg/ChildClass", "name", "Ljava/lang/String;", arc, hier)
    assert declaring == "pkg/ParentClass"


def test_f06_field_readers_index():
    """F6.4: Query all methods reading a specific field."""
    arc = JarArchive()
    cp = ConstantPool()
    fref = cp.add_member_ref(ConstantTag.FIELDREF, "pkg/State", "flag", "Z")
    code1 = bytes([Opcode.GETSTATIC, (fref >> 8) & 0xFF, fref & 0xFF, Opcode.IRETURN])
    code2 = bytes([Opcode.GETSTATIC, (fref >> 8) & 0xFF, fref & 0xFF, Opcode.IRETURN])
    m1 = MethodInfo(AccessFlags.PUBLIC | AccessFlags.STATIC, cp.add_utf8("readerOne"), cp.add_utf8("()Z"), [CodeAttribute(cp.add_utf8("Code"), 1, 1, code1)])
    m2 = MethodInfo(AccessFlags.PUBLIC | AccessFlags.STATIC, cp.add_utf8("readerTwo"), cp.add_utf8("()Z"), [CodeAttribute(cp.add_utf8("Code"), 1, 1, code2)])
    f = FieldInfo(AccessFlags.PUBLIC | AccessFlags.STATIC, cp.add_utf8("flag"), cp.add_utf8("Z"), [])
    cf = ClassFile(magic=JAVA_MAGIC, constant_pool=cp, access_flags=AccessFlags.PUBLIC, this_class=cp.add_class("pkg/State"), super_class=cp.add_class("java/lang/Object"), fields=[f], methods=[m1, m2])
    arc.add_class(cf)

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)
    readers = cg.get_field_readers("Lpkg/State;->flag:Z")
    assert readers == {"Lpkg/State;->readerOne()Z", "Lpkg/State;->readerTwo()Z"}


def test_f06_field_writers_index():
    """F6.5: Query all methods modifying a specific field."""
    arc = JarArchive()
    cp = ConstantPool()
    fref = cp.add_member_ref(ConstantTag.FIELDREF, "pkg/Counter", "val", "I")
    code1 = bytes([Opcode.ICONST_1, Opcode.PUTSTATIC, (fref >> 8) & 0xFF, fref & 0xFF, Opcode.RETURN])
    code2 = bytes([Opcode.ICONST_2, Opcode.PUTSTATIC, (fref >> 8) & 0xFF, fref & 0xFF, Opcode.RETURN])
    m1 = MethodInfo(AccessFlags.PUBLIC | AccessFlags.STATIC, cp.add_utf8("initVal"), cp.add_utf8("()V"), [CodeAttribute(cp.add_utf8("Code"), 1, 1, code1)])
    m2 = MethodInfo(AccessFlags.PUBLIC | AccessFlags.STATIC, cp.add_utf8("resetVal"), cp.add_utf8("()V"), [CodeAttribute(cp.add_utf8("Code"), 1, 1, code2)])
    f = FieldInfo(AccessFlags.PUBLIC | AccessFlags.STATIC, cp.add_utf8("val"), cp.add_utf8("I"), [])
    cf = ClassFile(magic=JAVA_MAGIC, constant_pool=cp, access_flags=AccessFlags.PUBLIC, this_class=cp.add_class("pkg/Counter"), super_class=cp.add_class("java/lang/Object"), fields=[f], methods=[m1, m2])
    arc.add_class(cf)

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)
    writers = cg.get_field_writers("Lpkg/Counter;->val:I")
    assert writers == {"Lpkg/Counter;->initVal()V", "Lpkg/Counter;->resetVal()V"}


# ============================================================================
# F7: Inter-Class Dependency Slicing
# ============================================================================

def test_f07_forward_slice_reachability():
    """F7.1: Forward reachability slice from root class."""
    arc = JarArchive()
    for name, target in [("com/A", "com/B"), ("com/B", "com/C"), ("com/C", None)]:
        cp = ConstantPool()
        c_idx = cp.add_class(name)
        s_idx = cp.add_class("java/lang/Object")
        if target:
            mref = cp.add_member_ref(ConstantTag.METHODREF, target, "run", "()V")
            code = bytes([Opcode.INVOKESTATIC, (mref >> 8) & 0xFF, mref & 0xFF, Opcode.RETURN])
        else:
            code = bytes([Opcode.RETURN])
        m = MethodInfo(AccessFlags.PUBLIC | AccessFlags.STATIC, cp.add_utf8("run"), cp.add_utf8("()V"), [
            CodeAttribute(cp.add_utf8("Code"), 1, 1, code)
        ])
        arc.add_class(ClassFile(magic=JAVA_MAGIC, constant_pool=cp, access_flags=AccessFlags.PUBLIC, this_class=c_idx, super_class=s_idx, methods=[m]))

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)
    cdg = ClassDependencyGraph.build(arc, hier, cg, include_external=False)
    assert cdg.forward_slice("com/A") == {"com/A", "com/B", "com/C"}
    assert cdg.forward_slice("com/B") == {"com/B", "com/C"}


def test_f07_backward_slice_dependents():
    """F7.2: Backward slice computing all classes depending on target class."""
    arc = JarArchive()
    for name, target in [("com/A", "com/B"), ("com/B", "com/C"), ("com/C", None)]:
        cp = ConstantPool()
        c_idx = cp.add_class(name)
        s_idx = cp.add_class("java/lang/Object")
        if target:
            mref = cp.add_member_ref(ConstantTag.METHODREF, target, "run", "()V")
            code = bytes([Opcode.INVOKESTATIC, (mref >> 8) & 0xFF, mref & 0xFF, Opcode.RETURN])
        else:
            code = bytes([Opcode.RETURN])
        m = MethodInfo(AccessFlags.PUBLIC | AccessFlags.STATIC, cp.add_utf8("run"), cp.add_utf8("()V"), [
            CodeAttribute(cp.add_utf8("Code"), 1, 1, code)
        ])
        arc.add_class(ClassFile(magic=JAVA_MAGIC, constant_pool=cp, access_flags=AccessFlags.PUBLIC, this_class=c_idx, super_class=s_idx, methods=[m]))

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)
    cdg = ClassDependencyGraph.build(arc, hier, cg, include_external=False)
    assert cdg.backward_slice("com/C") == {"com/A", "com/B", "com/C"}
    assert cdg.backward_slice("com/B") == {"com/A", "com/B"}


def test_f07_tarjan_scc_detection():
    """F7.3: Tarjan's algorithm detecting strongly connected components."""
    arc = JarArchive()
    # Cycle A -> B -> A, plus independent D
    for name, target in [("com/A", "com/B"), ("com/B", "com/A"), ("com/D", None)]:
        cp = ConstantPool()
        c_idx = cp.add_class(name)
        s_idx = cp.add_class("java/lang/Object")
        if target:
            mref = cp.add_member_ref(ConstantTag.METHODREF, target, "run", "()V")
            code = bytes([Opcode.INVOKESTATIC, (mref >> 8) & 0xFF, mref & 0xFF, Opcode.RETURN])
        else:
            code = bytes([Opcode.RETURN])
        m = MethodInfo(AccessFlags.PUBLIC | AccessFlags.STATIC, cp.add_utf8("run"), cp.add_utf8("()V"), [
            CodeAttribute(cp.add_utf8("Code"), 1, 1, code)
        ])
        arc.add_class(ClassFile(magic=JAVA_MAGIC, constant_pool=cp, access_flags=AccessFlags.PUBLIC, this_class=c_idx, super_class=s_idx, methods=[m]))

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)
    cdg = ClassDependencyGraph.build(arc, hier, cg, include_external=False)
    sccs = cdg.get_sccs()
    cycle_sccs = [s for s in sccs if len(s) > 1]
    assert len(cycle_sccs) == 1
    assert set(cycle_sccs[0]) == {"com/A", "com/B"}


def test_f07_cyclic_components():
    """F7.4: Isolation of cyclic dependency components."""
    arc = JarArchive()
    # Cyclic pair X <-> Y
    for name, target in [("com/X", "com/Y"), ("com/Y", "com/X"), ("com/Z", None)]:
        cp = ConstantPool()
        c_idx = cp.add_class(name)
        s_idx = cp.add_class("java/lang/Object")
        if target:
            mref = cp.add_member_ref(ConstantTag.METHODREF, target, "run", "()V")
            code = bytes([Opcode.INVOKESTATIC, (mref >> 8) & 0xFF, mref & 0xFF, Opcode.RETURN])
        else:
            code = bytes([Opcode.RETURN])
        m = MethodInfo(AccessFlags.PUBLIC | AccessFlags.STATIC, cp.add_utf8("run"), cp.add_utf8("()V"), [
            CodeAttribute(cp.add_utf8("Code"), 1, 1, code)
        ])
        arc.add_class(ClassFile(magic=JAVA_MAGIC, constant_pool=cp, access_flags=AccessFlags.PUBLIC, this_class=c_idx, super_class=s_idx, methods=[m]))

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)
    cdg = ClassDependencyGraph.build(arc, hier, cg, include_external=False)
    assert cdg.has_cycles()
    cyclic_comps = cdg.get_cyclic_components()
    assert len(cyclic_comps) == 1
    assert cyclic_comps[0] == {"com/X", "com/Y"}


def test_f07_condensation_topological_sort():
    """F7.5: Condensation DAG with topological sorting for compilation order."""
    arc = JarArchive()
    # DAG: A -> B -> C
    for name, target in [("com/A", "com/B"), ("com/B", "com/C"), ("com/C", None)]:
        cp = ConstantPool()
        c_idx = cp.add_class(name)
        s_idx = cp.add_class("java/lang/Object")
        if target:
            mref = cp.add_member_ref(ConstantTag.METHODREF, target, "run", "()V")
            code = bytes([Opcode.INVOKESTATIC, (mref >> 8) & 0xFF, mref & 0xFF, Opcode.RETURN])
        else:
            code = bytes([Opcode.RETURN])
        m = MethodInfo(AccessFlags.PUBLIC | AccessFlags.STATIC, cp.add_utf8("run"), cp.add_utf8("()V"), [
            CodeAttribute(cp.add_utf8("Code"), 1, 1, code)
        ])
        arc.add_class(ClassFile(magic=JAVA_MAGIC, constant_pool=cp, access_flags=AccessFlags.PUBLIC, this_class=c_idx, super_class=s_idx, methods=[m]))

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)
    cdg = ClassDependencyGraph.build(arc, hier, cg, include_external=False)

    order = cdg.topological_sort(direction="build")
    # For build order, dependencies must come first: C before B, B before A
    pos_c = min(i for i, comp in enumerate(order) if "com/C" in comp)
    pos_b = min(i for i, comp in enumerate(order) if "com/B" in comp)
    pos_a = min(i for i, comp in enumerate(order) if "com/A" in comp)
    assert pos_c < pos_b < pos_a
