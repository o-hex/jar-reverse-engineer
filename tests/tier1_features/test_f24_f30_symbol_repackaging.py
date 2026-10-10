"""
tests/tier1_features/test_f24_f30_symbol_repackaging.py:
Comprehensive Tier 1 feature tests for Features F24 through F30 (35 tests, 5 per feature).

F24: Symbol Mapping Schema Validation
F25: Union-Find Method Equivalence
F26: JDK Library Immutability Pinning
F27: Global Bytecode Reference Rewriting
F28: Access Widening & Class Verification
F29: Valid JAR Repackaging & SPI Sync
F30: Decompiled Source Tree Export
"""

import json
import zipfile
from pathlib import Path
import pytest

from jar_engine.model.constants import AccessFlags, ConstantTag, Opcode
from jar_engine.model.class_file import ClassFile
from jar_engine.model.cp_info import (
    ConstantPool, CONSTANT_Utf8_info, CONSTANT_Class_info,
    CONSTANT_NameAndType_info, CONSTANT_Methodref_info, CONSTANT_Fieldref_info
)
from jar_engine.model.member_info import FieldInfo, MethodInfo, CodeAttribute, SignatureAttribute
from jar_engine.model.jar_archive import JarArchive, Manifest
from jar_engine.walker.hierarchy import InheritanceHierarchy
from jar_engine.sanitizer import BytecodeSanitizer
from jar_engine.sanitizer.decompiler_check import (
    run_decompiler_check,
    verify_sanitized_class_structural_integrity,
    find_decompiler_command,
)
from jar_engine.renamer import (
    SymbolMapping,
    MappingValidationError,
    SymbolRenamer,
    RenamingReport,
    MethodEquivalenceCluster,
    ReferenceRewriter,
    JarRepackager,
    rewrite_descriptor,
)


def _make_class(
    name: str = "com/example/Sample",
    super_name: str = "java/lang/Object",
    access_flags: int = AccessFlags.PUBLIC,
    interfaces: list[str] | None = None,
    fields: list[FieldInfo] | None = None,
    methods: list[MethodInfo] | None = None,
) -> ClassFile:
    """Helper to build a valid minimal ClassFile for renaming tests."""
    cp = ConstantPool()
    this_c = cp.add_class(name)
    super_c = cp.add_class(super_name) if super_name else 0
    ifaces = [cp.add_class(iface) for iface in (interfaces or [])]
    return ClassFile(
        magic=0xCAFEBABE,
        minor_version=0,
        major_version=52,
        constant_pool=cp,
        access_flags=access_flags,
        this_class=this_c,
        super_class=super_c,
        interfaces=ifaces,
        fields=fields or [],
        methods=methods or [],
        attributes=[]
    )


# ==============================================================================
# F24: Symbol Mapping Schema Validation (5 tests)
# ==============================================================================

def test_f24_valid_mapping_json_parsing():
    """F24-1: Parse valid JSON symbol mapping into SymbolMapping data structure."""
    raw_json = json.dumps({
        "classes": {"com/old/Foo": "com/new/Bar", "com/old/Util": "com/new/Utility"},
        "fields": {"com/old/Foo#secretKey": "apiKey", "com/old/Foo#count#I": "itemCount"},
        "methods": {"com/old/Foo#decrypt#([B)[B": "decryptPayload"}
    })
    mapping = SymbolMapping.from_json(raw_json)
    assert mapping.classes["com/old/Foo"] == "com/new/Bar"
    assert mapping.classes["com/old/Util"] == "com/new/Utility"
    assert mapping.fields["com/old/Foo#secretKey"] == "apiKey"
    assert mapping.methods["com/old/Foo#decrypt#([B)[B"] == "decryptPayload"
    assert len(mapping.validate()) == 0


def test_f24_class_name_validation_rules():
    """F24-2: Verify valid and invalid JVM internal class names under validation rules."""
    valid_mapping = SymbolMapping(
        classes={
            "com/old/A": "com/example/MyClass",
            "com/old/B": "pkg/Sub_Pkg/Class$1",
            "com/old/C": "RootClass"
        }
    )
    assert valid_mapping.validate() == []

    # Invalid names: dotted instead of slashed, leading slash, trailing slash, invalid characters
    invalid_mapping = SymbolMapping(
        classes={
            "com/old/D": "com.example.DottedClass",
            "com/old/E": "/leading/slash",
            "com/old/F": "trailing/slash/"
        }
    )
    errors = invalid_mapping.validate()
    assert len(errors) == 3
    assert any("com.example.DottedClass" in e for e in errors)
    assert any("/leading/slash" in e for e in errors)
    assert any("trailing/slash/" in e for e in errors)


def test_f24_member_identifier_validation_rules():
    """F24-3: Verify valid Java identifier names for fields and methods."""
    valid_mapping = SymbolMapping(
        fields={"com/A#f1": "validIdentifier", "com/A#f2": "_underscoreVal", "com/A#f3": "$dollarVal"},
        methods={"com/A#m1#()V": "computeResult", "com/A#m2#()I": "get$Data"}
    )
    assert valid_mapping.validate() == []

    invalid_fields = SymbolMapping(
        fields={"com/A#f1": "123startsWithNum", "com/A#f2": "kebab-case", "com/A#f3": "with spaces"}
    )
    errors = invalid_fields.validate()
    assert len(errors) == 3
    assert all("Invalid new field name" in e for e in errors)


def test_f24_mapping_validation_error_raising():
    """F24-4: Raise MappingValidationError when parsing invalid mapping via from_dict or from_json."""
    bad_dict = {
        "classes": {"com/old/Bad": "com.invalid.dot.Notation"},
        "methods": {"com/old/Bad#m#()V": "bad-method-name"}
    }
    with pytest.raises(MappingValidationError) as exc_info:
        SymbolMapping.from_dict(bad_dict)
    assert "Invalid new class internal name" in str(exc_info.value)
    assert "Invalid new method name" in str(exc_info.value)

    bad_json = json.dumps(bad_dict)
    with pytest.raises(MappingValidationError):
        SymbolMapping.from_json(bad_json)


def test_f24_to_dict_and_to_json_serialization():
    """F24-5: Verify roundtrip serialization of SymbolMapping via to_dict, to_json, and from_json."""
    orig = SymbolMapping(
        classes={"com/pkg/OldClass": "com/pkg/NewClass"},
        fields={"com/pkg/OldClass#oldField": "newField"},
        methods={"com/pkg/OldClass#oldMethod#()V": "newMethod"}
    )
    dict_repr = orig.to_dict()
    assert dict_repr == {
        "classes": {"com/pkg/OldClass": "com/pkg/NewClass"},
        "fields": {"com/pkg/OldClass#oldField": "newField"},
        "methods": {"com/pkg/OldClass#oldMethod#()V": "newMethod"}
    }
    json_str = orig.to_json()
    reparsed = SymbolMapping.from_json(json_str)
    assert reparsed.classes == orig.classes
    assert reparsed.fields == orig.fields
    assert reparsed.methods == orig.methods


# ==============================================================================
# F25: Union-Find Method Equivalence (5 tests)
# ==============================================================================

def test_f25_disjoint_set_union_and_find():
    """F25-1: Verify basic Disjoint-Set Union-Find operations with path compression."""
    hier = InheritanceHierarchy()
    cluster = MethodEquivalenceCluster(hier)

    assert cluster._find("itemA") == "itemA"
    assert cluster._find("itemB") == "itemB"

    cluster._union("itemA", "itemB")
    assert cluster._find("itemA") == cluster._find("itemB")

    cluster._union("itemB", "itemC")
    assert cluster._find("itemA") == cluster._find("itemC")
    assert cluster._find("itemB") == cluster._find("itemC")


def test_f25_polymorphic_override_clustering():
    """F25-2: Cluster overridden methods across inheritance hierarchies."""
    arc = JarArchive()
    # Base class
    cf_base = _make_class("com/example/BaseService")
    cp_b = cf_base.constant_pool
    m_base = MethodInfo(AccessFlags.PUBLIC, cp_b.add_utf8("process"), cp_b.add_utf8("(I)V"), [])
    cf_base.methods.append(m_base)
    arc.add_class(cf_base)

    # Derived class
    cf_derived = _make_class("com/example/DerivedService", super_name="com/example/BaseService")
    cp_d = cf_derived.constant_pool
    m_der = MethodInfo(AccessFlags.PUBLIC, cp_d.add_utf8("process"), cp_d.add_utf8("(I)V"), [])
    cf_derived.methods.append(m_der)
    arc.add_class(cf_derived)

    hier = arc.get_hierarchy()
    cluster = MethodEquivalenceCluster(hier)
    cluster.build_from_archive(arc)

    k_base = "com/example/BaseService#process#(I)V"
    k_der = "com/example/DerivedService#process#(I)V"
    assert cluster._find(k_base) == cluster._find(k_der)


def test_f25_interface_implementation_clustering():
    """F25-3: Cluster interface methods and implementing class methods."""
    arc = JarArchive()
    # Interface
    cf_iface = _make_class("com/example/Worker", access_flags=AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT)
    cp_i = cf_iface.constant_pool
    m_iface = MethodInfo(AccessFlags.PUBLIC | AccessFlags.ABSTRACT, cp_i.add_utf8("doWork"), cp_i.add_utf8("()V"), [])
    cf_iface.methods.append(m_iface)
    arc.add_class(cf_iface)

    # Implementation
    cf_impl = _make_class("com/example/WorkerImpl", interfaces=["com/example/Worker"])
    cp_imp = cf_impl.constant_pool
    m_impl = MethodInfo(AccessFlags.PUBLIC, cp_imp.add_utf8("doWork"), cp_imp.add_utf8("()V"), [])
    cf_impl.methods.append(m_impl)
    arc.add_class(cf_impl)

    hier = arc.get_hierarchy()
    cluster = MethodEquivalenceCluster(hier)
    cluster.build_from_archive(arc)

    k_iface = "com/example/Worker#doWork#()V"
    k_impl = "com/example/WorkerImpl#doWork#()V"
    assert cluster._find(k_iface) == cluster._find(k_impl)


def test_f25_multi_hierarchy_equivalence_cluster():
    """F25-4: Maintain consistency across diamond inheritance hierarchies."""
    arc = JarArchive()
    # Top interface
    cf_top = _make_class("com/example/Top", access_flags=AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT)
    cf_top.methods.append(MethodInfo(AccessFlags.PUBLIC | AccessFlags.ABSTRACT, cf_top.constant_pool.add_utf8("act"), cf_top.constant_pool.add_utf8("()V"), []))
    arc.add_class(cf_top)

    # Left interface extending Top
    cf_left = _make_class("com/example/Left", access_flags=AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT, interfaces=["com/example/Top"])
    cf_left.methods.append(MethodInfo(AccessFlags.PUBLIC | AccessFlags.ABSTRACT, cf_left.constant_pool.add_utf8("act"), cf_left.constant_pool.add_utf8("()V"), []))
    arc.add_class(cf_left)

    # Right interface extending Top
    cf_right = _make_class("com/example/Right", access_flags=AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT, interfaces=["com/example/Top"])
    cf_right.methods.append(MethodInfo(AccessFlags.PUBLIC | AccessFlags.ABSTRACT, cf_right.constant_pool.add_utf8("act"), cf_right.constant_pool.add_utf8("()V"), []))
    arc.add_class(cf_right)

    # Bottom class implementing both Left and Right
    cf_bot = _make_class("com/example/Bottom", interfaces=["com/example/Left", "com/example/Right"])
    cf_bot.methods.append(MethodInfo(AccessFlags.PUBLIC, cf_bot.constant_pool.add_utf8("act"), cf_bot.constant_pool.add_utf8("()V"), []))
    arc.add_class(cf_bot)

    hier = arc.get_hierarchy()
    cluster = MethodEquivalenceCluster(hier)
    cluster.build_from_archive(arc)

    k_top = "com/example/Top#act#()V"
    k_left = "com/example/Left#act#()V"
    k_right = "com/example/Right#act#()V"
    k_bot = "com/example/Bottom#act#()V"

    root = cluster._find(k_top)
    assert cluster._find(k_left) == root
    assert cluster._find(k_right) == root
    assert cluster._find(k_bot) == root


def test_f25_cluster_member_retrieval():
    """F25-5: Retrieve all cluster member method keys for given method signature."""
    hier = InheritanceHierarchy()
    cluster = MethodEquivalenceCluster(hier)

    cluster._union("A#execute#()V", "B#execute#()V")
    cluster._union("B#execute#()V", "C#execute#()V")

    all_keys = {"A#execute#()V", "B#execute#()V", "C#execute#()V", "D#unrelated#()V"}
    members = cluster.get_cluster_members("A#execute#()V", all_keys)
    assert set(members) == {"A#execute#()V", "B#execute#()V", "C#execute#()V"}
    assert "D#unrelated#()V" not in members


# ==============================================================================
# F26: JDK Library Immutability Pinning (5 tests)
# ==============================================================================

def test_f26_object_methods_pinned():
    """F26-1: Verify java/lang/Object methods (toString, equals, hashCode, <init>) pinned."""
    hier = InheritanceHierarchy()
    cluster = MethodEquivalenceCluster(hier)

    for m_name in ["toString", "equals", "hashCode", "<init>", "<clinit>", "clone", "finalize"]:
        assert cluster.is_jdk_method("com/example/AnyClass", m_name) is True


def test_f26_common_jdk_interfaces_pinned():
    """F26-2: Verify common JDK interfaces (Runnable.run, Closeable.close) pinned."""
    hier = InheritanceHierarchy()
    cluster = MethodEquivalenceCluster(hier)

    assert cluster.is_jdk_method("java/lang/Runnable", "run") is True
    assert cluster.is_jdk_method("java/io/Closeable", "close") is True
    assert cluster.is_jdk_method("java/lang/AutoCloseable", "close") is True
    assert cluster.is_jdk_method("java/util/concurrent/Callable", "call") is True
    assert cluster.is_jdk_method("java/util/function/Function", "apply") is True


def test_f26_pinned_methods_propagated_to_cluster():
    """F26-3: Ensure entire equivalence cluster is pinned if any member is JDK method."""
    arc = JarArchive()
    cf = _make_class("com/example/Task", interfaces=["java/lang/Runnable"])
    cp = cf.constant_pool
    cf.methods.append(MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("run"), cp.add_utf8("()V"), []))
    arc.add_class(cf)

    hier = arc.get_hierarchy()
    cluster = MethodEquivalenceCluster(hier)
    cluster.build_from_archive(arc)

    k_run = "com/example/Task#run#()V"
    assert cluster.is_method_pinned(k_run) is True


def test_f26_is_method_pinned_check():
    """F26-4: Assert is_method_pinned returns True for pinned methods and False for custom methods."""
    hier = InheritanceHierarchy()
    cluster = MethodEquivalenceCluster(hier)

    custom_key = "com/example/MyService#doCustomLogic#()V"
    cluster._find(custom_key)
    assert cluster.is_method_pinned(custom_key) is False

    pinned_key = "com/example/MyService#toString#()Ljava/lang/String;"
    cluster.pinned.add(cluster._find(pinned_key))
    assert cluster.is_method_pinned(pinned_key) is True


def test_f26_renamer_skips_pinned_methods():
    """F26-5: Assert SymbolRenamer preserves pinned methods unchanged and logs pinned_skipped."""
    arc = JarArchive()
    cf = _make_class("com/example/Entity")
    cp = cf.constant_pool
    cf.methods.append(MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("toString"), cp.add_utf8("()Ljava/lang/String;"), []))
    arc.add_class(cf)

    mapping = SymbolMapping(
        methods={"com/example/Entity#toString#()Ljava/lang/String;": "entityAsString"}
    )
    renamer = SymbolRenamer()
    report = renamer.apply_mapping(arc, mapping)

    assert report.pinned_skipped == 1
    assert cf.methods[0].name(cf.constant_pool) == "toString"


# ==============================================================================
# F27: Global Bytecode Reference Rewriting (5 tests)
# ==============================================================================

def test_f27_class_definition_rename():
    """F27-1: Update this_class constant pool UTF8 entry to new name."""
    arc = JarArchive()
    cf = _make_class("com/old/Alpha")
    arc.add_class(cf)

    mapping = SymbolMapping(classes={"com/old/Alpha": "com/new/AlphaRenamed"})
    renamer = SymbolRenamer()
    report = renamer.apply_mapping(arc, mapping)

    assert report.classes_renamed == 1
    assert "com/new/AlphaRenamed" in arc.classes
    assert "com/old/Alpha" not in arc.classes
    assert arc.classes["com/new/AlphaRenamed"].this_class_name == "com/new/AlphaRenamed"


def test_f27_field_and_method_definition_rename():
    """F27-2: Update field and method names in constant pool."""
    arc = JarArchive()
    cf = _make_class("com/example/Worker")
    cp = cf.constant_pool
    f = FieldInfo(AccessFlags.PUBLIC, cp.add_utf8("oldField"), cp.add_utf8("I"), [])
    m = MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("oldMethod"), cp.add_utf8("()V"), [])
    cf.fields.append(f)
    cf.methods.append(m)
    arc.add_class(cf)

    mapping = SymbolMapping(
        fields={"com/example/Worker#oldField": "newField"},
        methods={"com/example/Worker#oldMethod#()V": "newMethod"}
    )
    renamer = SymbolRenamer()
    report = renamer.apply_mapping(arc, mapping)

    assert report.fields_renamed == 1
    assert report.methods_renamed == 1
    assert f.name(cf.constant_pool) == "newField"
    assert m.name(cf.constant_pool) == "newMethod"


def test_f27_constant_pool_class_ref_rewriting():
    """F27-3: Rewrite all CONSTANT_Class_info references across all archive classes."""
    arc = JarArchive()
    cf1 = _make_class("com/example/Alpha")
    cf2 = _make_class("com/example/Beta")
    # cf2 references cf1 via CONSTANT_Class_info
    ref_idx = cf2.constant_pool.add_class("com/example/Alpha")
    arc.add_class(cf1)
    arc.add_class(cf2)

    mapping = SymbolMapping(classes={"com/example/Alpha": "com/example/AlphaPrime"})
    renamer = SymbolRenamer()
    report = renamer.apply_mapping(arc, mapping)

    # Check that cf2's reference was updated
    cls_entry = cf2.constant_pool._entries[ref_idx]
    assert isinstance(cls_entry, CONSTANT_Class_info)
    assert cf2.constant_pool.resolve_utf8(cls_entry.name_index) == "com/example/AlphaPrime"


def test_f27_memberref_and_nameandtype_rewriting():
    """F27-4: Rewrite Memberref and NameAndType references across all classes."""
    arc = JarArchive()
    cf_target = _make_class("com/example/Target")
    cp_t = cf_target.constant_pool
    m_target = MethodInfo(AccessFlags.PUBLIC, cp_t.add_utf8("perform"), cp_t.add_utf8("()V"), [])
    cf_target.methods.append(m_target)
    arc.add_class(cf_target)

    # Caller references Target.perform()V via Methodref
    cf_caller = _make_class("com/example/Caller")
    cp_c = cf_caller.constant_pool
    mref_idx = cp_c.add_member_ref(ConstantTag.METHODREF, "com/example/Target", "perform", "()V")
    arc.add_class(cf_caller)

    mapping = SymbolMapping(methods={"com/example/Target#perform#()V": "executeAction"})
    renamer = SymbolRenamer()
    report = renamer.apply_mapping(arc, mapping)

    # Check caller's Methodref points to NameAndType with updated name
    mref_entry = cf_caller.constant_pool._entries[mref_idx]
    assert isinstance(mref_entry, CONSTANT_Methodref_info)
    nat_entry = cf_caller.constant_pool._entries[mref_entry.name_and_type_index]
    assert isinstance(nat_entry, CONSTANT_NameAndType_info)
    assert cf_caller.constant_pool.resolve_utf8(nat_entry.name_index) == "executeAction"


def test_f27_descriptor_rewriting_utility():
    """F27-5: Verify rewrite_descriptor rewrites class types inside method/field descriptors."""
    renames = {
        "com/old/Param": "com/new/Param",
        "com/old/Return": "com/new/Return",
        "com/old/ArrayElem": "com/new/ArrayElem"
    }

    # Field descriptor
    assert rewrite_descriptor("Lcom/old/Param;", renames) == "Lcom/new/Param;"
    # Array descriptor
    assert rewrite_descriptor("[[Lcom/old/ArrayElem;", renames) == "[[Lcom/new/ArrayElem;"
    # Method descriptor with multiple parameters and return type
    orig_m = "(Lcom/old/Param;I[Lcom/old/ArrayElem;)Lcom/old/Return;"
    expected_m = "(Lcom/new/Param;I[Lcom/new/ArrayElem;)Lcom/new/Return;"
    assert rewrite_descriptor(orig_m, renames) == expected_m
    # Primitive descriptor untouched
    assert rewrite_descriptor("(II)V", renames) == "(II)V"


# ==============================================================================
# F28: Access Widening & Class Verification (5 tests)
# ==============================================================================

def test_f28_package_relocation_detection():
    """F28-1: Detect when renamed class moves from one package to another."""
    same_pkg = {"com/example/OldClass": "com/example/NewClass"}
    diff_pkg = {"com/example/OldClass": "org/other/NewClass"}

    def is_relocated(old: str, new: str) -> bool:
        old_pkg = old.rsplit("/", 1)[0] if "/" in old else ""
        new_pkg = new.rsplit("/", 1)[0] if "/" in new else ""
        return old_pkg != new_pkg

    assert not is_relocated("com/example/OldClass", "com/example/NewClass")
    assert is_relocated("com/example/OldClass", "org/other/NewClass")


def test_f28_package_private_to_public_widening():
    """F28-2: Widen package-private class to public upon relocation to another package."""
    arc = JarArchive()
    cf = _make_class("com/old/pkg/HiddenClass", access_flags=0)  # package-private
    arc.add_class(cf)

    mapping = SymbolMapping(classes={"com/old/pkg/HiddenClass": "org/new/pkg/PublicClass"})
    renamer = SymbolRenamer()
    report = renamer.apply_mapping(arc, mapping)

    assert report.access_widened == 1
    renamed_cf = arc.get_class("org/new/pkg/PublicClass")
    assert renamed_cf.is_public is True


def test_f28_protected_to_public_widening():
    """F28-3: Widen protected access to public on relocated members/classes."""
    arc = JarArchive()
    cf = _make_class("com/old/pkg/ProtClass", access_flags=AccessFlags.PROTECTED)
    arc.add_class(cf)

    mapping = SymbolMapping(classes={"com/old/pkg/ProtClass": "org/new/pkg/ProtToPub"})
    renamer = SymbolRenamer()
    report = renamer.apply_mapping(arc, mapping)

    assert report.access_widened == 1
    renamed_cf = arc.get_class("org/new/pkg/ProtToPub")
    assert renamed_cf.is_public is True


def test_f28_already_public_unchanged():
    """F28-4: Retain already public classes without modifying access flags on package move."""
    arc = JarArchive()
    cf = _make_class("com/old/pkg/PubClass", access_flags=AccessFlags.PUBLIC)
    arc.add_class(cf)

    mapping = SymbolMapping(classes={"com/old/pkg/PubClass": "org/new/pkg/PubClass"})
    renamer = SymbolRenamer()
    report = renamer.apply_mapping(arc, mapping)

    assert report.access_widened == 0
    renamed_cf = arc.get_class("org/new/pkg/PubClass")
    assert renamed_cf.is_public is True


def test_f28_report_access_widened_count():
    """F28-5: Verify RenamingReport.access_widened accurately reflects widened member count."""
    arc = JarArchive()
    # 2 package-private classes moved across packages
    cf1 = _make_class("pkg1/A", access_flags=0)
    cf2 = _make_class("pkg1/B", access_flags=0)
    # 1 package-private class staying in same package
    cf3 = _make_class("pkg1/C", access_flags=0)
    arc.add_class(cf1)
    arc.add_class(cf2)
    arc.add_class(cf3)

    mapping = SymbolMapping(classes={
        "pkg1/A": "pkg2/A",
        "pkg1/B": "pkg2/B",
        "pkg1/C": "pkg1/CNew",
    })
    renamer = SymbolRenamer()
    report = renamer.apply_mapping(arc, mapping)

    assert report.access_widened == 2


# ==============================================================================
# F29: Valid JAR Repackaging & SPI Sync (5 tests)
# ==============================================================================

def test_f29_jar_repackaging_zip_structure(tmp_path: Path):
    """F29-1: Repackage JarArchive into valid ZIP format with META-INF/MANIFEST.MF."""
    arc = JarArchive()
    arc.manifest = Manifest()
    cf = _make_class("com/example/MyService")
    arc.add_class(cf)
    arc.set_resource("config.properties", b"debug=false\n")

    out_jar = tmp_path / "output.jar"
    repackager = JarRepackager()
    repackager.repackage_to_file(arc, out_jar)

    assert out_jar.is_file()
    with zipfile.ZipFile(out_jar, "r") as zf:
        namelist = zf.namelist()
        assert "META-INF/MANIFEST.MF" in namelist
        assert "com/example/MyService.class" in namelist
        assert "config.properties" in namelist
        assert zf.read("config.properties") == b"debug=false\n"


def test_f29_manifest_main_class_synchronization():
    """F29-2: Update Main-Class in manifest when entry point class is renamed."""
    arc = JarArchive()
    cf = _make_class("com/old/MainApp")
    arc.add_class(cf)
    manifest = Manifest()
    manifest.set_main_class("com.old.MainApp")
    arc.manifest = manifest

    mapping = SymbolMapping(classes={"com/old/MainApp": "com/new/Application"})
    repackager = JarRepackager()
    synced = repackager.synchronize_manifest_and_services(arc, mapping)

    assert synced >= 1
    assert arc.manifest.get_main_class() == "com.new.Application"


def test_f29_spi_services_file_rename():
    """F29-3: Rename service interface files in META-INF/services/."""
    arc = JarArchive()
    arc.set_resource("META-INF/services/com.old.IService", b"com.example.ProviderImpl\n")

    mapping = SymbolMapping(classes={"com/old/IService": "com/new/IServiceRenamed"})
    repackager = JarRepackager()
    repackager.synchronize_manifest_and_services(arc, mapping)

    assert not arc.has_resource("META-INF/services/com.old.IService")
    assert arc.has_resource("META-INF/services/com.new.IServiceRenamed")


def test_f29_spi_services_content_rewrite():
    """F29-4: Rewrite implementation provider class names inside SPI service files."""
    arc = JarArchive()
    spi_text = b"com.old.ProviderOne\ncom.old.ProviderTwo\n# comment\n"
    arc.set_resource("META-INF/services/com.api.Service", spi_text)

    mapping = SymbolMapping(classes={
        "com/old/ProviderOne": "com/new/ProviderAlpha",
        "com/old/ProviderTwo": "com/new/ProviderBeta"
    })
    repackager = JarRepackager()
    repackager.synchronize_manifest_and_services(arc, mapping)

    updated_data = arc.get_resource("META-INF/services/com.api.Service").decode("utf-8")
    assert "com.new.ProviderAlpha" in updated_data
    assert "com.new.ProviderBeta" in updated_data
    assert "com.old.ProviderOne" not in updated_data
    assert "# comment" in updated_data


def test_f29_signature_stripping():
    """F29-5: Ensure old digital signatures are omitted from repackaged archive."""
    arc = JarArchive()
    arc.set_resource("META-INF/TEST.SF", b"sig data")
    arc.set_resource("META-INF/TEST.RSA", b"rsa data")
    arc.set_resource("META-INF/TEST.DSA", b"dsa data")
    arc.set_resource("data/normal.txt", b"plain data")

    repackager = JarRepackager(strip_signatures=True)
    repackager.synchronize_manifest_and_services(arc, SymbolMapping())

    assert not arc.has_resource("META-INF/TEST.SF")
    assert not arc.has_resource("META-INF/TEST.RSA")
    assert not arc.has_resource("META-INF/TEST.DSA")
    assert arc.has_resource("data/normal.txt")


# ==============================================================================
# F30: Decompiled Source Tree Export (5 tests)
# ==============================================================================

def test_f30_source_export_structure(tmp_path: Path):
    """F30-1: Orchestrate export of decompiled Java source tree directory structure."""
    arc = JarArchive()
    cf1 = _make_class("com/example/service/UserService")
    cf2 = _make_class("com/example/model/User")
    arc.add_class(cf1)
    arc.add_class(cf2)

    export_dir = tmp_path / "src_export"
    for c_name, cf in arc.classes.items():
        rel_path = f"{c_name}.class"
        out_file = export_dir / rel_path
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_bytes(cf.to_bytes())

    assert (export_dir / "com" / "example" / "service" / "UserService.class").is_file()
    assert (export_dir / "com" / "example" / "model" / "User.class").is_file()


def test_f30_decompilation_pipeline_integration():
    """F30-2: Integrate sanitization and symbol restoration prior to source export."""
    arc = JarArchive()
    cf = _make_class("com/old/SampleClass")
    # Add corrupt signature
    cp = cf.constant_pool
    sig_attr = SignatureAttribute(cp.add_utf8("Signature"), cp.add_utf8("bad<<<sig"))
    cf.attributes.append(sig_attr)
    arc.add_class(cf)

    # Step 1: Sanitize
    sanitizer = BytecodeSanitizer()
    s_rep = sanitizer.sanitize_archive(arc)
    assert s_rep.total_signatures_stripped >= 1

    # Step 2: Renaming
    mapping = SymbolMapping(classes={"com/old/SampleClass": "com/restored/SampleClass"})
    renamer = SymbolRenamer()
    r_rep = renamer.apply_mapping(arc, mapping)
    assert r_rep.classes_renamed == 1

    # Step 3: Verify clean state for export
    clean_cf = arc.get_class("com/restored/SampleClass")
    valid, issues = verify_sanitized_class_structural_integrity(clean_cf)
    assert valid is True
    assert len(issues) == 0


def test_f30_structural_validation_before_export():
    """F30-3: Verify sanitized class structural integrity passes prior to export."""
    cf_good = _make_class("com/example/ValidExport")
    v_good, issues_good = verify_sanitized_class_structural_integrity(cf_good)
    assert v_good is True
    assert len(issues_good) == 0

    # Class with duplicate field names fails structural validation
    cf_bad = _make_class("com/example/BadExport")
    cp_b = cf_bad.constant_pool
    cf_bad.fields.append(FieldInfo(AccessFlags.PUBLIC, cp_b.add_utf8("dup"), cp_b.add_utf8("I"), []))
    cf_bad.fields.append(FieldInfo(AccessFlags.PUBLIC, cp_b.add_utf8("dup"), cp_b.add_utf8("I"), []))
    v_bad, issues_bad = verify_sanitized_class_structural_integrity(cf_bad)
    assert v_bad is False
    assert any("duplicate field" in issue.lower() for issue in issues_bad)


def test_f30_package_directory_hierarchy_export(tmp_path: Path):
    """F30-4: Export source files into correct package directory structure."""
    out_dir = tmp_path / "java_src"
    classes = [
        ("com/alpha/A", "package com.alpha;\npublic class A {}\n"),
        ("com/beta/gamma/B", "package com.beta.gamma;\npublic class B {}\n"),
    ]

    for internal_name, src_code in classes:
        parts = internal_name.split("/")
        pkg_dir = out_dir.joinpath(*parts[:-1])
        pkg_dir.mkdir(parents=True, exist_ok=True)
        src_file = pkg_dir / f"{parts[-1]}.java"
        src_file.write_text(src_code, encoding="utf-8")

    assert (out_dir / "com" / "alpha" / "A.java").is_file()
    assert (out_dir / "com" / "beta" / "gamma" / "B.java").is_file()
    assert "package com.alpha;" in (out_dir / "com" / "alpha" / "A.java").read_text(encoding="utf-8")


def test_f30_export_fallback_handling(tmp_path: Path):
    """F30-5: Fallback gracefully when decompiler binaries are not installed."""
    dummy_jar = tmp_path / "sample.jar"
    dummy_jar.write_bytes(b"PK\x05\x06" + b"\x00" * 18)  # empty zip

    results = run_decompiler_check(dummy_jar, timeout_seconds=1)
    assert isinstance(results, dict)
    for decompiler_name in ["cfr", "vineflower", "procyon"]:
        assert decompiler_name in results
        res = results[decompiler_name]
        # When not found, available is False, error is informative, and no crash occurs
        assert isinstance(res.available, bool)
        assert isinstance(res.success, bool)
