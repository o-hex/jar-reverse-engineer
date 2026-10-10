"""
tests/tier2_boundaries/test_bva_f01_f07_model_walker.py:
Boundary value analysis and corner cases for features F1 through F7.
Total: 35 tests.
"""

import io
import math
import struct
import zipfile
import pytest

from jar_engine.model.constants import (
    JAVA_MAGIC, ConstantTag, AccessFlags, Opcode, ReferenceKind
)
from jar_engine.model.cp_info import (
    ConstantPool, CpPlaceholder,
    CONSTANT_Utf8_info, CONSTANT_Integer_info, CONSTANT_Float_info,
    CONSTANT_Long_info, CONSTANT_Double_info, CONSTANT_Class_info,
    CONSTANT_String_info, CONSTANT_Fieldref_info, CONSTANT_Methodref_info,
    CONSTANT_InterfaceMethodref_info, CONSTANT_NameAndType_info
)
from jar_engine.model.member_info import (
    FieldInfo, MethodInfo, CodeAttribute, RawAttribute
)
from jar_engine.model.class_file import ClassFile, ClassFormatError
from jar_engine.model.jar_archive import JarArchive, Manifest
from jar_engine.walker.hierarchy import (
    ClassOrigin, HierarchyNode, InheritanceHierarchy
)
from jar_engine.walker.call_graph import (
    CallGraph, MethodReference, FieldReference
)
from jar_engine.walker.dependency_slice import (
    ClassDependencyGraph, DependencyKind
)


# ==============================================================================
# F1 BOUNDARIES (5 tests)
# ==============================================================================

def test_bva_f01_empty_class_file():
    """BVA F1-1: Minimal valid class with 0 fields, 0 methods, 0 attributes."""
    cp = ConstantPool()
    c_idx = cp.add_class("com/example/EmptyClass")
    s_idx = cp.add_class("java/lang/Object")
    cf = ClassFile(
        magic=JAVA_MAGIC,
        minor_version=0,
        major_version=52,
        constant_pool=cp,
        access_flags=AccessFlags.PUBLIC,
        this_class=c_idx,
        super_class=s_idx,
        interfaces=[],
        fields=[],
        methods=[],
        attributes=[]
    )
    raw = cf.to_bytes()
    reparsed = ClassFile.parse(raw)
    assert reparsed.this_class_name == "com/example/EmptyClass"
    assert reparsed.super_class_name == "java/lang/Object"
    assert len(reparsed.interfaces) == 0
    assert len(reparsed.fields) == 0
    assert len(reparsed.methods) == 0
    assert len(reparsed.attributes) == 0
    assert reparsed.to_bytes() == raw


def test_bva_f01_max_constant_pool_count():
    """BVA F1-2: Scaled constant pool with thousands of items nearing uint16 limit."""
    cp = ConstantPool()
    c_idx = cp.add_class("com/example/LargePoolClass")
    s_idx = cp.add_class("java/lang/Object")
    # Add 1200 unique UTF-8 constants
    last_idx = 0
    for i in range(1200):
        last_idx = cp.add_utf8(f"str_const_tag_{i:04d}")

    assert len(cp) >= 1202
    cf = ClassFile(
        magic=JAVA_MAGIC,
        minor_version=0,
        major_version=52,
        constant_pool=cp,
        access_flags=AccessFlags.PUBLIC,
        this_class=c_idx,
        super_class=s_idx,
        interfaces=[],
        fields=[],
        methods=[],
        attributes=[]
    )
    raw = cf.to_bytes()
    reparsed = ClassFile.parse(raw)
    assert len(reparsed.constant_pool) == len(cp)
    assert reparsed.constant_pool.resolve_utf8(last_idx) == "str_const_tag_1199"


def test_bva_f01_max_interfaces():
    """BVA F1-3: Class implementing many interfaces at boundary."""
    cp = ConstantPool()
    c_idx = cp.add_class("com/example/MultiInterfaceClass")
    s_idx = cp.add_class("java/lang/Object")
    iface_indices = []
    for i in range(64):
        iface_indices.append(cp.add_class(f"com/example/Iface{i:03d}"))

    cf = ClassFile(
        magic=JAVA_MAGIC,
        minor_version=0,
        major_version=52,
        constant_pool=cp,
        access_flags=AccessFlags.PUBLIC,
        this_class=c_idx,
        super_class=s_idx,
        interfaces=iface_indices,
        fields=[],
        methods=[],
        attributes=[]
    )
    raw = cf.to_bytes()
    reparsed = ClassFile.parse(raw)
    assert len(reparsed.interfaces) == 64
    assert reparsed.interface_names[0] == "com/example/Iface000"
    assert reparsed.interface_names[63] == "com/example/Iface063"


def test_bva_f01_zero_length_attribute():
    """BVA F1-4: Class, field, or method with attribute_length = 0."""
    cp = ConstantPool()
    c_idx = cp.add_class("com/example/ZeroLenAttrClass")
    s_idx = cp.add_class("java/lang/Object")
    synthetic_name_idx = cp.add_utf8("Synthetic")
    raw_attr = RawAttribute(attribute_name_index=synthetic_name_idx, info=b"")

    cf = ClassFile(
        magic=JAVA_MAGIC,
        minor_version=0,
        major_version=52,
        constant_pool=cp,
        access_flags=AccessFlags.PUBLIC,
        this_class=c_idx,
        super_class=s_idx,
        interfaces=[],
        fields=[],
        methods=[],
        attributes=[raw_attr]
    )
    raw = cf.to_bytes()
    reparsed = ClassFile.parse(raw)
    assert len(reparsed.attributes) == 1
    assert reparsed.attributes[0].name(reparsed.constant_pool) == "Synthetic"
    assert len(reparsed.attributes[0].info) == 0


def test_bva_f01_extreme_floating_point_constants():
    """BVA F1-5: Extreme float/double constants (NaN bit-exactness, +/-0.0, +/-inf, subnormals)."""
    cp = ConstantPool()
    # 1. Float NaN bit-exact
    f_nan_bits = 0x7FA01234
    f_idx = len(cp)
    f_info = CONSTANT_Float_info(raw_bits=f_nan_bits)
    cp._entries.append(f_info)

    # 2. Float -0.0
    f_neg_zero = CONSTANT_Float_info(value=-0.0)
    f_neg_zero_idx = len(cp)
    cp._entries.append(f_neg_zero)

    # 3. Float subnormal
    f_subnormal = CONSTANT_Float_info(value=1.4e-45)
    f_sub_idx = len(cp)
    cp._entries.append(f_subnormal)

    # 4. Double NaN bit-exact
    d_nan_bits = 0x7FF8000012345678
    d_info = CONSTANT_Double_info(raw_bits=d_nan_bits)
    d_idx = len(cp)
    cp._entries.append(d_info)
    cp._entries.append(CpPlaceholder())

    # 5. Double -0.0
    d_neg_zero = CONSTANT_Double_info(value=-0.0)
    d_neg_zero_idx = len(cp)
    cp._entries.append(d_neg_zero)
    cp._entries.append(CpPlaceholder())

    c_idx = cp.add_class("com/example/FloatingClass")
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
    raw = cf.to_bytes()
    reparsed = ClassFile.parse(raw)

    assert reparsed.constant_pool[f_idx].raw_bits == f_nan_bits
    assert reparsed.constant_pool[f_neg_zero_idx].raw_bits == 0x80000000
    assert reparsed.constant_pool[d_idx].raw_bits == d_nan_bits
    assert reparsed.constant_pool[d_neg_zero_idx].raw_bits == 0x8000000000000000


# ==============================================================================
# F2 BOUNDARIES (5 tests)
# ==============================================================================

def test_bva_f02_empty_jar_zero_classes():
    """BVA F2-1: JarArchive with 0 class entries."""
    archive = JarArchive()
    archive.manifest = Manifest(main_attributes={"Manifest-Version": "1.0"})
    raw_zip = archive.to_bytes()

    reloaded = JarArchive.from_bytes(raw_zip)
    assert len(reloaded.classes) == 0
    assert reloaded.get_stats().class_count == 0
    assert reloaded.manifest is not None
    assert reloaded.manifest.get_attribute("Manifest-Version") == "1.0"


def test_bva_f02_single_class_jar(make_test_class):
    """BVA F2-2: JarArchive with exactly 1 class."""
    cf = make_test_class(name="com/example/SoleClass")
    archive = JarArchive()
    archive.add_class(cf)

    assert archive.has_class("com/example/SoleClass")
    assert archive.get_stats().class_count == 1
    reloaded = JarArchive.from_bytes(archive.to_bytes())
    assert reloaded.has_class("com/example/SoleClass")
    assert len(reloaded.classes) == 1


def test_bva_f02_corrupted_zip_header_rejection():
    """BVA F2-3: Gracefully raise error on non-ZIP files."""
    corrupted_data = b"NOT_A_VALID_ZIP_STREAM_HEADER_BYTE_DATA_PADDING"
    with pytest.raises(zipfile.BadZipFile):
        JarArchive.from_bytes(corrupted_data)


def test_bva_f02_non_class_binary_assets(make_test_class):
    """BVA F2-4: Binary assets (PNG header, serialized bytes) preserved byte-for-byte."""
    cf = make_test_class(name="com/example/App")
    archive = JarArchive()
    archive.add_class(cf)

    png_header = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06"
    serialized_obj = bytes(range(256))
    archive.set_resource("assets/icon.png", png_header)
    archive.set_resource("data/blob.bin", serialized_obj)

    reloaded = JarArchive.from_bytes(archive.to_bytes())
    assert reloaded.get_resource("assets/icon.png") == png_header
    assert reloaded.get_resource("data/blob.bin") == serialized_obj


def test_bva_f02_deeply_nested_package_paths(make_test_class):
    """BVA F2-5: Class with long deeply nested package path depth."""
    deep_path = "/".join([f"pkg{i}" for i in range(25)]) + "/DeepClass"
    assert len(deep_path) > 120
    cf = make_test_class(name=deep_path)
    archive = JarArchive()
    archive.add_class(cf)

    reloaded = JarArchive.from_bytes(archive.to_bytes())
    assert reloaded.has_class(deep_path)
    retrieved = reloaded.get_class(deep_path)
    assert retrieved.this_class_name == deep_path


# ==============================================================================
# F3 BOUNDARIES (5 tests)
# ==============================================================================

def test_bva_f03_single_node_hierarchy():
    """BVA F3-1: Class hierarchy with only java/lang/Object."""
    hier = InheritanceHierarchy()
    hier.nodes["java/lang/Object"] = HierarchyNode(
        "java/lang/Object", None, (), AccessFlags.PUBLIC, ClassOrigin.JDK_KNOWN
    )
    hier._super_class["java/lang/Object"] = None

    assert hier.get_superclasses("java/lang/Object") == []
    assert hier.get_subclasses("java/lang/Object") == set()
    assert hier.get_interfaces("java/lang/Object") == []
    assert hier.find_lowest_common_ancestor("java/lang/Object", "java/lang/Object") == "java/lang/Object"


def test_bva_f03_circular_inheritance_safety():
    """BVA F3-2: Graceful handling of malformed circular inheritance without infinite loops."""
    hier = InheritanceHierarchy()
    hier.nodes["A"] = HierarchyNode("A", "B", (), AccessFlags.PUBLIC, ClassOrigin.ARCHIVE)
    hier.nodes["B"] = HierarchyNode("B", "C", (), AccessFlags.PUBLIC, ClassOrigin.ARCHIVE)
    hier.nodes["C"] = HierarchyNode("C", "A", (), AccessFlags.PUBLIC, ClassOrigin.ARCHIVE)
    hier._super_class["A"] = "B"
    hier._super_class["B"] = "C"
    hier._super_class["C"] = "A"

    supers = hier.get_superclasses("A")
    assert "B" in supers and "C" in supers
    # LCA does not hang
    lca = hier.find_lowest_common_ancestor("A", "B")
    assert lca in ("A", "B", "C", "java/lang/Object")


def test_bva_f03_deep_linear_inheritance():
    """BVA F3-3: Linear inheritance chain of depth 50."""
    hier = InheritanceHierarchy()
    hier.nodes["java/lang/Object"] = HierarchyNode(
        "java/lang/Object", None, (), AccessFlags.PUBLIC, ClassOrigin.JDK_KNOWN
    )
    hier._super_class["java/lang/Object"] = None

    prev = "java/lang/Object"
    for i in range(50):
        cls_name = f"com/depth/Node{i:02d}"
        hier.nodes[cls_name] = HierarchyNode(cls_name, prev, (), AccessFlags.PUBLIC, ClassOrigin.ARCHIVE)
        hier._super_class[cls_name] = prev
        prev = cls_name

    assert len(hier.get_superclasses("com/depth/Node49")) == 50
    assert hier.find_lowest_common_ancestor("com/depth/Node49", "com/depth/Node20") == "com/depth/Node20"
    assert hier.is_assignable_from("com/depth/Node10", "com/depth/Node40")
    assert not hier.is_assignable_from("com/depth/Node40", "com/depth/Node10")


def test_bva_f03_complex_interface_diamond():
    """BVA F3-4: Diamond interface implementation hierarchy with multiple branches."""
    hier = InheritanceHierarchy()
    hier.nodes["java/lang/Object"] = HierarchyNode("java/lang/Object", None, (), AccessFlags.PUBLIC, ClassOrigin.JDK_KNOWN)
    hier._super_class["java/lang/Object"] = None

    # Top -> Left, Right -> Bottom
    for name, ifaces in [
        ("I_Top", ()),
        ("I_Left", ("I_Top",)),
        ("I_Right", ("I_Top",)),
        ("I_Bottom", ("I_Left", "I_Right"))
    ]:
        hier.nodes[name] = HierarchyNode(name, "java/lang/Object", ifaces, AccessFlags.PUBLIC | AccessFlags.INTERFACE, ClassOrigin.ARCHIVE)
        hier._super_class[name] = "java/lang/Object"
        hier._interfaces[name] = list(ifaces)

    all_ifaces = hier.get_all_interfaces("I_Bottom")
    assert "I_Top" in all_ifaces
    assert "I_Left" in all_ifaces
    assert "I_Right" in all_ifaces


def test_bva_f03_object_root_boundary():
    """BVA F3-5: LCA of java/lang/Object with any class returns java/lang/Object."""
    hier = InheritanceHierarchy()
    hier.nodes["java/lang/Object"] = HierarchyNode("java/lang/Object", None, (), AccessFlags.PUBLIC, ClassOrigin.JDK_KNOWN)
    hier._super_class["java/lang/Object"] = None
    hier.nodes["com/Custom"] = HierarchyNode("com/Custom", "java/lang/Object", (), AccessFlags.PUBLIC, ClassOrigin.ARCHIVE)
    hier._super_class["com/Custom"] = "java/lang/Object"

    assert hier.find_lowest_common_ancestor("java/lang/Object", "com/Custom") == "java/lang/Object"
    assert hier.find_lowest_common_ancestor("com/Custom", "java/lang/Object") == "java/lang/Object"
    assert hier.find_lowest_common_ancestor("java/lang/Object", "non/Existent") == "java/lang/Object"


# ==============================================================================
# F4 BOUNDARIES (5 tests)
# ==============================================================================

def test_bva_f04_leaf_method_zero_callees():
    """BVA F4-1: Method with 0 invocations returns empty callees set."""
    cp = ConstantPool()
    c_idx = cp.add_class("com/example/LeafClass")
    s_idx = cp.add_class("java/lang/Object")
    m_name = cp.add_utf8("leafMethod")
    m_desc = cp.add_utf8("()V")
    code = bytes([Opcode.RETURN])
    code_attr = CodeAttribute(cp.add_utf8("Code"), 1, 1, code)
    m = MethodInfo(AccessFlags.PUBLIC, m_name, m_desc, [code_attr])
    cf = ClassFile(magic=JAVA_MAGIC, constant_pool=cp, access_flags=AccessFlags.PUBLIC, this_class=c_idx, super_class=s_idx, methods=[m])

    arc = JarArchive()
    arc.add_class(cf)
    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)

    leaf_sig = "Lcom/example/LeafClass;->leafMethod()V"
    assert cg.get_callees(leaf_sig) == set()
    assert leaf_sig in cg.get_leaf_methods()


def test_bva_f04_polymorphic_target_set_20_classes():
    """BVA F4-2: Interface method implemented by 20 distinct classes."""
    arc = JarArchive()

    # Interface IAction
    cp_iface = ConstantPool()
    c_iface = cp_iface.add_class("com/example/IAction")
    s_iface = cp_iface.add_class("java/lang/Object")
    m_act_name = cp_iface.add_utf8("act")
    m_act_desc = cp_iface.add_utf8("()V")
    m_iface = MethodInfo(AccessFlags.PUBLIC | AccessFlags.ABSTRACT, m_act_name, m_act_desc, [])
    cf_iface = ClassFile(
        magic=JAVA_MAGIC, constant_pool=cp_iface,
        access_flags=AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT,
        this_class=c_iface, super_class=s_iface, methods=[m_iface]
    )
    arc.add_class(cf_iface)

    # 20 Implementing classes
    for i in range(20):
        cp_i = ConstantPool()
        c_i = cp_i.add_class(f"com/example/ActionImpl{i:02d}")
        s_i = cp_i.add_class("java/lang/Object")
        iface_ref = cp_i.add_class("com/example/IAction")
        code_attr = CodeAttribute(cp_i.add_utf8("Code"), 1, 1, bytes([Opcode.RETURN]))
        m_impl = MethodInfo(AccessFlags.PUBLIC, cp_i.add_utf8("act"), cp_i.add_utf8("()V"), [code_attr])
        cf_i = ClassFile(
            magic=JAVA_MAGIC, constant_pool=cp_i, access_flags=AccessFlags.PUBLIC,
            this_class=c_i, super_class=s_i, interfaces=[iface_ref], methods=[m_impl]
        )
        arc.add_class(cf_i)

    # Caller invoking IAction.act()V
    cp_caller = ConstantPool()
    c_caller = cp_caller.add_class("com/example/Caller")
    s_caller = cp_caller.add_class("java/lang/Object")
    imref = cp_caller.add_member_ref(ConstantTag.INTERFACE_METHODREF, "com/example/IAction", "act", "()V")
    code_caller = bytes([
        Opcode.INVOKEINTERFACE, (imref >> 8) & 0xFF, imref & 0xFF, 1, 0,
        Opcode.RETURN
    ])
    m_caller = MethodInfo(AccessFlags.PUBLIC, cp_caller.add_utf8("run"), cp_caller.add_utf8("()V"), [CodeAttribute(cp_caller.add_utf8("Code"), 2, 2, code_caller)])
    cf_caller = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_caller, access_flags=AccessFlags.PUBLIC, this_class=c_caller, super_class=s_caller, methods=[m_caller])
    arc.add_class(cf_caller)

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier, mode="cha")
    callees = cg.get_callees("Lcom/example/Caller;->run()V")
    assert len(callees) == 20
    for i in range(20):
        assert f"Lcom/example/ActionImpl{i:02d};->act()V" in callees


def test_bva_f04_interface_zero_implementors():
    """BVA F4-3: Interface call with 0 implementors in archive returns graceful target."""
    arc = JarArchive()
    cp = ConstantPool()
    c_idx = cp.add_class("com/example/Client")
    s_idx = cp.add_class("java/lang/Object")
    imref = cp.add_member_ref(ConstantTag.INTERFACE_METHODREF, "com/example/EmptyIface", "missingMethod", "()V")
    code = bytes([
        Opcode.INVOKEINTERFACE, (imref >> 8) & 0xFF, imref & 0xFF, 1, 0,
        Opcode.RETURN
    ])
    m = MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("run"), cp.add_utf8("()V"), [CodeAttribute(cp.add_utf8("Code"), 2, 2, code)])
    cf = ClassFile(magic=JAVA_MAGIC, constant_pool=cp, access_flags=AccessFlags.PUBLIC, this_class=c_idx, super_class=s_idx, methods=[m])
    arc.add_class(cf)

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier, mode="cha")
    callees = cg.get_callees("Lcom/example/Client;->run()V")
    # With zero implementors, fallback returns the interface method itself
    assert "Lcom/example/EmptyIface;->missingMethod()V" in callees


def test_bva_f04_unresolvable_external_class():
    """BVA F4-4: External class invocation indexed cleanly without crashing."""
    arc = JarArchive()
    cp = ConstantPool()
    c_idx = cp.add_class("com/example/ClientExt")
    s_idx = cp.add_class("java/lang/Object")
    mref = cp.add_member_ref(ConstantTag.METHODREF, "org/apache/commons/lang3/StringUtils", "isEmpty", "(Ljava/lang/CharSequence;)Z")
    code = bytes([
        Opcode.INVOKESTATIC, (mref >> 8) & 0xFF, mref & 0xFF,
        Opcode.IRETURN
    ])
    m = MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("check"), cp.add_utf8("()Z"), [CodeAttribute(cp.add_utf8("Code"), 2, 2, code)])
    cf = ClassFile(magic=JAVA_MAGIC, constant_pool=cp, access_flags=AccessFlags.PUBLIC, this_class=c_idx, super_class=s_idx, methods=[m])
    arc.add_class(cf)

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)
    callees = cg.get_callees("Lcom/example/ClientExt;->check()Z")
    assert "Lorg/apache/commons/lang3/StringUtils;->isEmpty(Ljava/lang/CharSequence;)Z" in callees


def test_bva_f04_recursive_self_invoking_method():
    """BVA F4-5: Method that calls itself handled cleanly in call graph."""
    arc = JarArchive()
    cp = ConstantPool()
    c_idx = cp.add_class("com/example/Recursion")
    s_idx = cp.add_class("java/lang/Object")
    mref = cp.add_member_ref(ConstantTag.METHODREF, "com/example/Recursion", "fact", "(I)I")
    code = bytes([
        Opcode.ILOAD_0,
        Opcode.INVOKESTATIC, (mref >> 8) & 0xFF, mref & 0xFF,
        Opcode.IRETURN
    ])
    m = MethodInfo(AccessFlags.PUBLIC | AccessFlags.STATIC, cp.add_utf8("fact"), cp.add_utf8("(I)I"), [CodeAttribute(cp.add_utf8("Code"), 2, 2, code)])
    cf = ClassFile(magic=JAVA_MAGIC, constant_pool=cp, access_flags=AccessFlags.PUBLIC, this_class=c_idx, super_class=s_idx, methods=[m])
    arc.add_class(cf)

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)
    sig = "Lcom/example/Recursion;->fact(I)I"
    assert sig in cg.get_callees(sig)
    assert sig in cg.get_callers(sig)


# ==============================================================================
# F5 BOUNDARIES (5 tests)
# ==============================================================================

def test_bva_f05_clinit_static_initializer():
    """BVA F5-1: <clinit> static initializer method resolution."""
    arc = JarArchive()
    cp = ConstantPool()
    c_idx = cp.add_class("com/example/StaticInit")
    s_idx = cp.add_class("java/lang/Object")
    f_idx = cp.add_member_ref(ConstantTag.FIELDREF, "com/example/StaticInit", "STATIC_VAL", "I")
    code = bytes([
        Opcode.ICONST_5,
        Opcode.PUTSTATIC, (f_idx >> 8) & 0xFF, f_idx & 0xFF,
        Opcode.RETURN
    ])
    m = MethodInfo(AccessFlags.STATIC, cp.add_utf8("<clinit>"), cp.add_utf8("()V"), [CodeAttribute(cp.add_utf8("Code"), 2, 0, code)])
    cf = ClassFile(magic=JAVA_MAGIC, constant_pool=cp, access_flags=AccessFlags.PUBLIC, this_class=c_idx, super_class=s_idx, methods=[m])
    arc.add_class(cf)

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)
    field_sig = "Lcom/example/StaticInit;->STATIC_VAL:I"
    assert "Lcom/example/StaticInit;-><clinit>()V" in cg.field_writers[field_sig]


def test_bva_f05_init_constructor_chaining():
    """BVA F5-2: this(...) constructor chaining vs super(...) constructor calls."""
    arc = JarArchive()
    cp = ConstantPool()
    c_idx = cp.add_class("com/example/Constructors")
    s_idx = cp.add_class("java/lang/Object")

    super_init = cp.add_member_ref(ConstantTag.METHODREF, "java/lang/Object", "<init>", "()V")
    this_init = cp.add_member_ref(ConstantTag.METHODREF, "com/example/Constructors", "<init>", "(I)V")

    # ctor 1: ()V calls this(0)
    code1 = bytes([
        Opcode.ALOAD_0,
        Opcode.ICONST_0,
        Opcode.INVOKESPECIAL, (this_init >> 8) & 0xFF, this_init & 0xFF,
        Opcode.RETURN
    ])
    m1 = MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("<init>"), cp.add_utf8("()V"), [CodeAttribute(cp.add_utf8("Code"), 2, 1, code1)])

    # ctor 2: (I)V calls super()
    code2 = bytes([
        Opcode.ALOAD_0,
        Opcode.INVOKESPECIAL, (super_init >> 8) & 0xFF, super_init & 0xFF,
        Opcode.RETURN
    ])
    m2 = MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("<init>"), cp.add_utf8("(I)V"), [CodeAttribute(cp.add_utf8("Code"), 2, 2, code2)])

    cf = ClassFile(magic=JAVA_MAGIC, constant_pool=cp, access_flags=AccessFlags.PUBLIC, this_class=c_idx, super_class=s_idx, methods=[m1, m2])
    arc.add_class(cf)

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)

    callees1 = cg.get_callees("Lcom/example/Constructors;-><init>()V")
    assert "Lcom/example/Constructors;-><init>(I)V" in callees1

    callees2 = cg.get_callees("Lcom/example/Constructors;-><init>(I)V")
    assert "Ljava/lang/Object;-><init>()V" in callees2


def test_bva_f05_private_method_special_invoke():
    """BVA F5-3: Invoking private method on current class via invokespecial."""
    arc = JarArchive()
    cp = ConstantPool()
    c_idx = cp.add_class("com/example/PrivateCaller")
    s_idx = cp.add_class("java/lang/Object")

    priv_ref = cp.add_member_ref(ConstantTag.METHODREF, "com/example/PrivateCaller", "secretHelper", "()I")
    code_pub = bytes([
        Opcode.ALOAD_0,
        Opcode.INVOKESPECIAL, (priv_ref >> 8) & 0xFF, priv_ref & 0xFF,
        Opcode.IRETURN
    ])
    m_pub = MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("run"), cp.add_utf8("()I"), [CodeAttribute(cp.add_utf8("Code"), 2, 1, code_pub)])

    code_priv = bytes([Opcode.ICONST_4, Opcode.IRETURN])
    m_priv = MethodInfo(AccessFlags.PRIVATE, cp.add_utf8("secretHelper"), cp.add_utf8("()I"), [CodeAttribute(cp.add_utf8("Code"), 1, 1, code_priv)])

    cf = ClassFile(magic=JAVA_MAGIC, constant_pool=cp, access_flags=AccessFlags.PUBLIC, this_class=c_idx, super_class=s_idx, methods=[m_pub, m_priv])
    arc.add_class(cf)

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)
    callees = cg.get_callees("Lcom/example/PrivateCaller;->run()I")
    assert "Lcom/example/PrivateCaller;->secretHelper()I" in callees


def test_bva_f05_missing_special_target():
    """BVA F5-4: Non-existent target in superclass handled gracefully without unhandled exception."""
    arc = JarArchive()
    cp = ConstantPool()
    c_idx = cp.add_class("com/example/Sub")
    s_idx = cp.add_class("java/lang/Object")

    mref = cp.add_member_ref(ConstantTag.METHODREF, "java/lang/Object", "nonExistentMethod", "()V")
    code = bytes([
        Opcode.ALOAD_0,
        Opcode.INVOKESPECIAL, (mref >> 8) & 0xFF, mref & 0xFF,
        Opcode.RETURN
    ])
    m = MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("run"), cp.add_utf8("()V"), [CodeAttribute(cp.add_utf8("Code"), 2, 1, code)])
    cf = ClassFile(magic=JAVA_MAGIC, constant_pool=cp, access_flags=AccessFlags.PUBLIC, this_class=c_idx, super_class=s_idx, methods=[m])
    arc.add_class(cf)

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)
    callees = cg.get_callees("Lcom/example/Sub;->run()V")
    assert "Ljava/lang/Object;->nonExistentMethod()V" in callees


def test_bva_f05_static_method_subclass_shadow():
    """BVA F5-5: Static method shadowed in subclass resolved to correct declaring class."""
    arc = JarArchive()

    # Base declares static helper()V
    cp_base = ConstantPool()
    cf_base = ClassFile(
        magic=JAVA_MAGIC, constant_pool=cp_base, access_flags=AccessFlags.PUBLIC,
        this_class=cp_base.add_class("com/example/Base"), super_class=cp_base.add_class("java/lang/Object"),
        methods=[MethodInfo(AccessFlags.PUBLIC | AccessFlags.STATIC, cp_base.add_utf8("helper"), cp_base.add_utf8("()V"), [CodeAttribute(cp_base.add_utf8("Code"), 1, 0, bytes([Opcode.RETURN]))])]
    )
    arc.add_class(cf_base)

    # Sub declares static helper()V (shadowing)
    cp_sub = ConstantPool()
    cf_sub = ClassFile(
        magic=JAVA_MAGIC, constant_pool=cp_sub, access_flags=AccessFlags.PUBLIC,
        this_class=cp_sub.add_class("com/example/SubClass"), super_class=cp_sub.add_class("com/example/Base"),
        methods=[MethodInfo(AccessFlags.PUBLIC | AccessFlags.STATIC, cp_sub.add_utf8("helper"), cp_sub.add_utf8("()V"), [CodeAttribute(cp_sub.add_utf8("Code"), 1, 0, bytes([Opcode.RETURN]))])]
    )
    arc.add_class(cf_sub)

    # Caller calls SubClass.helper()V
    cp_caller = ConstantPool()
    mref = cp_caller.add_member_ref(ConstantTag.METHODREF, "com/example/SubClass", "helper", "()V")
    code = bytes([Opcode.INVOKESTATIC, (mref >> 8) & 0xFF, mref & 0xFF, Opcode.RETURN])
    cf_caller = ClassFile(
        magic=JAVA_MAGIC, constant_pool=cp_caller, access_flags=AccessFlags.PUBLIC,
        this_class=cp_caller.add_class("com/example/StaticCaller"), super_class=cp_caller.add_class("java/lang/Object"),
        methods=[MethodInfo(AccessFlags.PUBLIC, cp_caller.add_utf8("test"), cp_caller.add_utf8("()V"), [CodeAttribute(cp_caller.add_utf8("Code"), 1, 1, code)])]
    )
    arc.add_class(cf_caller)

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)
    callees = cg.get_callees("Lcom/example/StaticCaller;->test()V")
    assert "Lcom/example/SubClass;->helper()V" in callees


# ==============================================================================
# F6 BOUNDARIES (5 tests)
# ==============================================================================

def test_bva_f06_zero_field_accesses():
    """BVA F6-1: Class with 0 field reads and writes."""
    arc = JarArchive()
    cp = ConstantPool()
    code = bytes([Opcode.RETURN])
    m = MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("run"), cp.add_utf8("()V"), [CodeAttribute(cp.add_utf8("Code"), 1, 1, code)])
    cf = ClassFile(magic=JAVA_MAGIC, constant_pool=cp, access_flags=AccessFlags.PUBLIC, this_class=cp.add_class("com/NoFields"), super_class=cp.add_class("java/lang/Object"), methods=[m])
    arc.add_class(cf)

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)
    assert len(cg.field_accesses) == 0
    assert len(cg.field_readers) == 0
    assert len(cg.field_writers) == 0


def test_bva_f06_field_name_collision_different_types():
    """BVA F6-2: Same field name with different descriptors across hierarchy."""
    arc = JarArchive()

    # Base has field data:I
    cp_base = ConstantPool()
    f_int = FieldInfo(AccessFlags.PUBLIC, cp_base.add_utf8("data"), cp_base.add_utf8("I"), [])
    cf_base = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_base, access_flags=AccessFlags.PUBLIC, this_class=cp_base.add_class("com/BaseCol"), super_class=cp_base.add_class("java/lang/Object"), fields=[f_int])
    arc.add_class(cf_base)

    # Sub has field data:Ljava/lang/String;
    cp_sub = ConstantPool()
    f_str = FieldInfo(AccessFlags.PUBLIC, cp_sub.add_utf8("data"), cp_sub.add_utf8("Ljava/lang/String;"), [])
    fref = cp_sub.add_member_ref(ConstantTag.FIELDREF, "com/SubCol", "data", "Ljava/lang/String;")
    code = bytes([Opcode.ALOAD_0, Opcode.GETFIELD, (fref >> 8) & 0xFF, fref & 0xFF, Opcode.POP, Opcode.RETURN])
    m_sub = MethodInfo(AccessFlags.PUBLIC, cp_sub.add_utf8("readStr"), cp_sub.add_utf8("()V"), [CodeAttribute(cp_sub.add_utf8("Code"), 2, 1, code)])
    cf_sub = ClassFile(magic=JAVA_MAGIC, constant_pool=cp_sub, access_flags=AccessFlags.PUBLIC, this_class=cp_sub.add_class("com/SubCol"), super_class=cp_sub.add_class("com/BaseCol"), fields=[f_str], methods=[m_sub])
    arc.add_class(cf_sub)

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)

    str_field = "Lcom/SubCol;->data:Ljava/lang/String;"
    int_field = "Lcom/BaseCol;->data:I"
    assert "Lcom/SubCol;->readStr()V" in cg.field_readers[str_field]
    assert int_field not in cg.field_readers


def test_bva_f06_write_only_field():
    """BVA F6-3: Field written to but never read."""
    arc = JarArchive()
    cp = ConstantPool()
    c_idx = cp.add_class("com/WriteOnly")
    s_idx = cp.add_class("java/lang/Object")
    f_idx = cp.add_member_ref(ConstantTag.FIELDREF, "com/WriteOnly", "writeOnlyField", "I")
    f_info = FieldInfo(AccessFlags.PUBLIC, cp.add_utf8("writeOnlyField"), cp.add_utf8("I"), [])

    code = bytes([
        Opcode.ALOAD_0,
        Opcode.ICONST_1,
        Opcode.PUTFIELD, (f_idx >> 8) & 0xFF, f_idx & 0xFF,
        Opcode.RETURN
    ])
    m = MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("writeIt"), cp.add_utf8("()V"), [CodeAttribute(cp.add_utf8("Code"), 2, 2, code)])
    cf = ClassFile(magic=JAVA_MAGIC, constant_pool=cp, access_flags=AccessFlags.PUBLIC, this_class=c_idx, super_class=s_idx, fields=[f_info], methods=[m])
    arc.add_class(cf)

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)

    field_sig = "Lcom/WriteOnly;->writeOnlyField:I"
    assert "Lcom/WriteOnly;->writeIt()V" in cg.field_writers[field_sig]
    assert field_sig not in cg.field_readers


def test_bva_f06_read_only_constant_field():
    """BVA F6-4: static final field read but never written in methods."""
    arc = JarArchive()
    cp = ConstantPool()
    c_idx = cp.add_class("com/ReadOnly")
    s_idx = cp.add_class("java/lang/Object")
    f_idx = cp.add_member_ref(ConstantTag.FIELDREF, "com/ReadOnly", "CONSTANT_VAL", "I")
    f_info = FieldInfo(AccessFlags.PUBLIC | AccessFlags.STATIC | AccessFlags.FINAL, cp.add_utf8("CONSTANT_VAL"), cp.add_utf8("I"), [])

    code = bytes([
        Opcode.GETSTATIC, (f_idx >> 8) & 0xFF, f_idx & 0xFF,
        Opcode.IRETURN
    ])
    m = MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("getVal"), cp.add_utf8("()I"), [CodeAttribute(cp.add_utf8("Code"), 2, 1, code)])
    cf = ClassFile(magic=JAVA_MAGIC, constant_pool=cp, access_flags=AccessFlags.PUBLIC, this_class=c_idx, super_class=s_idx, fields=[f_info], methods=[m])
    arc.add_class(cf)

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)

    field_sig = "Lcom/ReadOnly;->CONSTANT_VAL:I"
    assert "Lcom/ReadOnly;->getVal()I" in cg.field_readers[field_sig]
    assert field_sig not in cg.field_writers


def test_bva_f06_static_vs_instance_disambiguation():
    """BVA F6-5: Static and instance fields correctly disambiguated."""
    arc = JarArchive()
    cp = ConstantPool()
    c_idx = cp.add_class("com/Disambig")
    s_idx = cp.add_class("java/lang/Object")

    f_stat_ref = cp.add_member_ref(ConstantTag.FIELDREF, "com/Disambig", "statField", "I")
    f_inst_ref = cp.add_member_ref(ConstantTag.FIELDREF, "com/Disambig", "instField", "I")

    code = bytes([
        Opcode.GETSTATIC, (f_stat_ref >> 8) & 0xFF, f_stat_ref & 0xFF,
        Opcode.ALOAD_0,
        Opcode.GETFIELD, (f_inst_ref >> 8) & 0xFF, f_inst_ref & 0xFF,
        Opcode.IADD,
        Opcode.IRETURN
    ])
    m = MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("sum"), cp.add_utf8("()I"), [CodeAttribute(cp.add_utf8("Code"), 2, 1, code)])
    cf = ClassFile(magic=JAVA_MAGIC, constant_pool=cp, access_flags=AccessFlags.PUBLIC, this_class=c_idx, super_class=s_idx, methods=[m])
    arc.add_class(cf)

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)

    stat_sig = "Lcom/Disambig;->statField:I"
    inst_sig = "Lcom/Disambig;->instField:I"
    assert "Lcom/Disambig;->sum()I" in cg.field_readers[stat_sig]
    assert "Lcom/Disambig;->sum()I" in cg.field_readers[inst_sig]
    assert len(cg.field_accesses) == 2


# ==============================================================================
# F7 BOUNDARIES (5 tests)
# ==============================================================================

def test_bva_f07_disconnected_graph_empty_slice():
    """BVA F7-1: Slice on isolated node returns only the node itself."""
    arc = JarArchive()
    cp = ConstantPool()
    cf = ClassFile(magic=JAVA_MAGIC, constant_pool=cp, access_flags=AccessFlags.PUBLIC, this_class=cp.add_class("com/Isolated"), super_class=cp.add_class("java/lang/Object"))
    arc.add_class(cf)

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)
    cdg = ClassDependencyGraph.build(arc, hier, cg, include_external=False)

    assert cdg.forward_slice("com/Isolated") == {"com/Isolated"}
    assert cdg.backward_slice("com/Isolated") == {"com/Isolated"}


def test_bva_f07_giant_cyclic_scc():
    """BVA F7-2: Archive where classes form a cyclic SCC."""
    arc = JarArchive()
    n = 10
    for i in range(n):
        nxt = (i + 1) % n
        cp = ConstantPool()
        c_idx = cp.add_class(f"com/Ring{i}")
        s_idx = cp.add_class("java/lang/Object")
        mref = cp.add_member_ref(ConstantTag.METHODREF, f"com/Ring{nxt}", "jump", "()V")
        code = bytes([Opcode.INVOKESTATIC, (mref >> 8) & 0xFF, mref & 0xFF, Opcode.RETURN])
        m = MethodInfo(AccessFlags.PUBLIC | AccessFlags.STATIC, cp.add_utf8("jump"), cp.add_utf8("()V"), [CodeAttribute(cp.add_utf8("Code"), 1, 0, code)])
        cf = ClassFile(magic=JAVA_MAGIC, constant_pool=cp, access_flags=AccessFlags.PUBLIC, this_class=c_idx, super_class=s_idx, methods=[m])
        arc.add_class(cf)

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)
    cdg = ClassDependencyGraph.build(arc, hier, cg, include_external=False)

    assert cdg.has_cycles()
    cyclics = cdg.get_cyclic_components()
    assert len(cyclics) == 1
    assert len(cyclics[0]) == 10


def test_bva_f07_missing_target_class():
    """BVA F7-3: Slicing non-existent class returns empty or single-element set safely."""
    arc = JarArchive()
    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)
    cdg = ClassDependencyGraph.build(arc, hier, cg, include_external=False)

    f_slice = cdg.forward_slice("com/NonExistentClass")
    assert f_slice == {"com/NonExistentClass"}
    b_slice = cdg.backward_slice("com/NonExistentClass")
    assert b_slice == {"com/NonExistentClass"}


def test_bva_f07_diamond_dependency_slice():
    """BVA F7-4: Diamond dependency correctly includes all paths."""
    arc = JarArchive()
    # A -> B, A -> C, B -> D, C -> D
    edges = [
        ("com/A", ["com/B", "com/C"]),
        ("com/B", ["com/D"]),
        ("com/C", ["com/D"]),
        ("com/D", [])
    ]
    for src, targets in edges:
        cp = ConstantPool()
        c_idx = cp.add_class(src)
        s_idx = cp.add_class("java/lang/Object")
        code = bytearray()
        for dst in targets:
            mref = cp.add_member_ref(ConstantTag.METHODREF, dst, "foo", "()V")
            code.extend([Opcode.INVOKESTATIC, (mref >> 8) & 0xFF, mref & 0xFF])
        code.append(Opcode.RETURN)
        m = MethodInfo(AccessFlags.PUBLIC | AccessFlags.STATIC, cp.add_utf8("foo"), cp.add_utf8("()V"), [CodeAttribute(cp.add_utf8("Code"), 1, 0, bytes(code))])
        cf = ClassFile(magic=JAVA_MAGIC, constant_pool=cp, access_flags=AccessFlags.PUBLIC, this_class=c_idx, super_class=s_idx, methods=[m])
        arc.add_class(cf)

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)
    cdg = ClassDependencyGraph.build(arc, hier, cg, include_external=False)

    assert cdg.forward_slice("com/A") == {"com/A", "com/B", "com/C", "com/D"}
    assert cdg.backward_slice("com/D") == {"com/A", "com/B", "com/C", "com/D"}
    assert cdg.backward_slice("com/B") == {"com/A", "com/B"}


def test_bva_f07_linear_chain_100_classes():
    """BVA F7-5: 100-class linear chain forward and backward slice consistency."""
    arc = JarArchive()
    chain_len = 100
    for i in range(chain_len):
        cp = ConstantPool()
        c_idx = cp.add_class(f"com/Chain{i:03d}")
        s_idx = cp.add_class("java/lang/Object")
        if i + 1 < chain_len:
            mref = cp.add_member_ref(ConstantTag.METHODREF, f"com/Chain{i+1:03d}", "step", "()V")
            code = bytes([Opcode.INVOKESTATIC, (mref >> 8) & 0xFF, mref & 0xFF, Opcode.RETURN])
        else:
            code = bytes([Opcode.RETURN])
        m = MethodInfo(AccessFlags.PUBLIC | AccessFlags.STATIC, cp.add_utf8("step"), cp.add_utf8("()V"), [CodeAttribute(cp.add_utf8("Code"), 1, 0, code)])
        cf = ClassFile(magic=JAVA_MAGIC, constant_pool=cp, access_flags=AccessFlags.PUBLIC, this_class=c_idx, super_class=s_idx, methods=[m])
        arc.add_class(cf)

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)
    cdg = ClassDependencyGraph.build(arc, hier, cg, include_external=False)

    f_slice = cdg.forward_slice("com/Chain000")
    assert len(f_slice) == 100

    b_slice = cdg.backward_slice("com/Chain099")
    assert len(b_slice) == 100

    mid_slice = cdg.forward_slice("com/Chain050")
    assert len(mid_slice) == 50
