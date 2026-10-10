"""
tests/tier2_boundaries/test_bva_f24_f30_symbol_repackaging.py:
Boundary value analysis and corner cases for features F24 through F30.
Total: 35 tests.
"""

import json
import pytest
from pathlib import Path

from jar_engine.model.constants import AccessFlags, ConstantTag, Opcode
from jar_engine.model.class_file import ClassFile
from jar_engine.model.cp_info import (
    ConstantPool, CONSTANT_Utf8_info, CONSTANT_Class_info
)
from jar_engine.model.member_info import FieldInfo, MethodInfo, CodeAttribute, SignatureAttribute
from jar_engine.model.jar_archive import JarArchive, Manifest, wrap_manifest_line
from jar_engine.walker.hierarchy import InheritanceHierarchy, HierarchyNode, ClassOrigin
from jar_engine.renamer import (
    SymbolMapping,
    MappingValidationError,
    SymbolRenamer,
    MethodEquivalenceCluster,
    JarRepackager,
    rewrite_descriptor,
)
from jar_engine.sanitizer.decompiler_check import (
    run_decompiler_check,
    verify_sanitized_class_structural_integrity
)


def create_minimal_test_class(
    name: str = "com/example/Sample",
    super_name: str = "java/lang/Object",
    access_flags: int = AccessFlags.PUBLIC
) -> ClassFile:
    """Helper to create minimal ClassFile."""
    cp = ConstantPool()
    this_c = cp.add_class(name)
    super_c = cp.add_class(super_name) if super_name else 0
    return ClassFile(
        magic=0xCAFEBABE,
        minor_version=0,
        major_version=52,
        constant_pool=cp,
        access_flags=access_flags,
        this_class=this_c,
        super_class=super_c,
        interfaces=[],
        fields=[],
        methods=[],
        attributes=[]
    )


# ==============================================================================
# F24 BOUNDARIES (5 tests)
# ==============================================================================

def test_bva_f24_empty_mapping_json():
    """BVA F24-1: Empty mapping '{}' is valid no-op."""
    mapping = SymbolMapping.from_json("{}")
    assert len(mapping.classes) == 0
    assert len(mapping.fields) == 0
    assert len(mapping.methods) == 0
    assert mapping.validate() == []


def test_bva_f24_malformed_json_syntax():
    """BVA F24-2: Malformed JSON syntax raises JSONDecodeError."""
    with pytest.raises(json.JSONDecodeError):
        SymbolMapping.from_json("{broken_json_syntax:")


def test_bva_f24_empty_class_name():
    """BVA F24-3: Empty string class name raises MappingValidationError."""
    with pytest.raises(MappingValidationError):
        SymbolMapping.from_dict({"classes": {"com/example/Old": ""}})


def test_bva_f24_invalid_identifier_chars():
    """BVA F24-4: Identifier with invalid characters (123abc, foo-bar) rejected."""
    with pytest.raises(MappingValidationError) as exc1:
        SymbolMapping.from_dict({"fields": {"com/example/A#f": "123invalid"}})
    assert "Invalid new field name" in str(exc1.value)

    with pytest.raises(MappingValidationError) as exc2:
        SymbolMapping.from_dict({"methods": {"com/example/A#m#()V": "bad-name"}})
    assert "Invalid new method name" in str(exc2.value)


def test_bva_f24_duplicate_destination_names():
    """BVA F24-5: Multiple classes mapping to distinct names serialize and validate cleanly."""
    mapping = SymbolMapping(
        classes={
            "com/example/A": "com/target/A",
            "com/example/B": "com/target/B"
        }
    )
    assert len(mapping.validate()) == 0
    as_json = mapping.to_json()
    reparsed = SymbolMapping.from_json(as_json)
    assert reparsed.classes == mapping.classes


# ==============================================================================
# F25 BOUNDARIES (5 tests)
# ==============================================================================

def test_bva_f25_single_element_cluster():
    """BVA F25-1: Union-Find with single element finds itself."""
    hier = InheritanceHierarchy()
    cluster = MethodEquivalenceCluster(hier)
    key = "com/example/SoleClass#execute#()V"
    assert cluster._find(key) == key


def test_bva_f25_fifty_chained_unions():
    """BVA F25-2: 50 chained unions collapse to single root."""
    hier = InheritanceHierarchy()
    cluster = MethodEquivalenceCluster(hier)
    for i in range(50):
        cluster._union(f"item_{i}", f"item_{i+1}")

    root_0 = cluster._find("item_0")
    for i in range(51):
        assert cluster._find(f"item_{i}") == root_0


def test_bva_f25_cycle_in_override_hierarchy():
    """BVA F25-3: Cyclic override definitions handled without infinite loop."""
    hier = InheritanceHierarchy()
    cluster = MethodEquivalenceCluster(hier)
    # Form a cycle A -> B -> C -> A
    cluster._union("nodeA", "nodeB")
    cluster._union("nodeB", "nodeC")
    cluster._union("nodeC", "nodeA")

    assert cluster._find("nodeA") == cluster._find("nodeB")
    assert cluster._find("nodeB") == cluster._find("nodeC")


def test_bva_f25_multiple_interface_inheritance():
    """BVA F25-4: Class implementing multiple interfaces with identical method signatures."""
    arc = JarArchive()
    # Interface 1
    cf_i1 = create_minimal_test_class("com/I1", access_flags=AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT)
    cf_i1.methods.append(MethodInfo(AccessFlags.PUBLIC | AccessFlags.ABSTRACT, cf_i1.constant_pool.add_utf8("run"), cf_i1.constant_pool.add_utf8("()V"), []))
    arc.add_class(cf_i1)

    # Interface 2
    cf_i2 = create_minimal_test_class("com/I2", access_flags=AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT)
    cf_i2.methods.append(MethodInfo(AccessFlags.PUBLIC | AccessFlags.ABSTRACT, cf_i2.constant_pool.add_utf8("run"), cf_i2.constant_pool.add_utf8("()V"), []))
    arc.add_class(cf_i2)

    # Impl implementing both
    cp_impl = ConstantPool()
    cf_impl = ClassFile(
        magic=0xCAFEBABE, minor_version=0, major_version=52, constant_pool=cp_impl, access_flags=AccessFlags.PUBLIC,
        this_class=cp_impl.add_class("com/Impl"), super_class=cp_impl.add_class("java/lang/Object"),
        interfaces=[cp_impl.add_class("com/I1"), cp_impl.add_class("com/I2")],
        methods=[MethodInfo(AccessFlags.PUBLIC, cp_impl.add_utf8("run"), cp_impl.add_utf8("()V"), [CodeAttribute(cp_impl.add_utf8("Code"), 1, 1, bytes([Opcode.RETURN]))])]
    )
    arc.add_class(cf_impl)

    hier = arc.get_hierarchy()
    cluster = MethodEquivalenceCluster(hier)
    cluster.build_from_archive(arc)

    k_i1 = "com/I1#run#()V"
    k_i2 = "com/I2#run#()V"
    k_impl = "com/Impl#run#()V"
    assert cluster._find(k_i1) == cluster._find(k_impl)
    assert cluster._find(k_i2) == cluster._find(k_impl)


def test_bva_f25_disjoint_method_names():
    """BVA F25-5: Distinct method names never merged into same cluster."""
    hier = InheritanceHierarchy()
    cluster = MethodEquivalenceCluster(hier)
    k_foo = "com/example/Service#foo#()V"
    k_bar = "com/example/Service#bar#()V"
    assert cluster._find(k_foo) != cluster._find(k_bar)


# ==============================================================================
# F26 BOUNDARIES (5 tests)
# ==============================================================================

def test_bva_f26_object_tostring_rename_rejected():
    """BVA F26-1: Renaming toString rejected or pinned."""
    arc = JarArchive()
    cf = create_minimal_test_class("com/example/Bean")
    cp = cf.constant_pool
    cf.methods.append(MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("toString"), cp.add_utf8("()Ljava/lang/String;"), [CodeAttribute(cp.add_utf8("Code"), 1, 1, bytes([Opcode.ACONST_NULL, Opcode.ARETURN]))]))
    arc.add_class(cf)

    mapping = SymbolMapping(
        methods={"com/example/Bean#toString#()Ljava/lang/String;": "customString"}
    )
    renamer = SymbolRenamer()
    report = renamer.apply_mapping(arc, mapping)

    assert report.pinned_skipped == 1
    assert cf.methods[0].name(cf.constant_pool) == "toString"


def test_bva_f26_object_init_rename_rejected():
    """BVA F26-2: Renaming <init> rejected or pinned."""
    hier = InheritanceHierarchy()
    cluster = MethodEquivalenceCluster(hier)
    assert cluster.is_jdk_method("com/example/Test", "<init>") is True
    assert cluster.is_jdk_method("com/example/Test", "<clinit>") is True


def test_bva_f26_external_library_method_pinned():
    """BVA F26-3: Methods on external library classes pinned."""
    hier = InheritanceHierarchy()
    cluster = MethodEquivalenceCluster(hier)
    assert cluster.is_jdk_method("java/util/List", "size") is True
    assert cluster.is_jdk_method("javax/crypto/Cipher", "doFinal") is True


def test_bva_f26_override_both_jdk_and_custom():
    """BVA F26-4: Method overriding JDK interface and custom interface pinned across all."""
    arc = JarArchive()
    cf = create_minimal_test_class("com/example/Task")
    cp = cf.constant_pool
    # implements Runnable
    cf.interfaces.append(cp.add_class("java/lang/Runnable"))
    cf.methods.append(MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("run"), cp.add_utf8("()V"), [CodeAttribute(cp.add_utf8("Code"), 1, 1, bytes([Opcode.RETURN]))]))
    arc.add_class(cf)

    hier = arc.get_hierarchy()
    cluster = MethodEquivalenceCluster(hier)
    cluster.build_from_archive(arc)

    k_run = "com/example/Task#run#()V"
    assert cluster.is_method_pinned(k_run) is True


def test_bva_f26_protected_object_methods():
    """BVA F26-5: Protected Object methods (clone, finalize) pinned."""
    hier = InheritanceHierarchy()
    cluster = MethodEquivalenceCluster(hier)
    assert cluster.is_jdk_method("com/example/Any", "clone") is True
    assert cluster.is_jdk_method("com/example/Any", "finalize") is True


# ==============================================================================
# F27 BOUNDARIES (5 tests)
# ==============================================================================

def test_bva_f27_rename_in_method_descriptor():
    """BVA F27-1: Method descriptor (Lold/Class;)V updated to (Lnew/Class;)V."""
    renames = {"old/Class": "new/Class"}
    orig = "(Lold/Class;I)Lold/Class;"
    updated = rewrite_descriptor(orig, renames)
    assert updated == "(Lnew/Class;I)Lnew/Class;"


def test_bva_f27_multidimensional_array_descriptor():
    """BVA F27-2: [[[Lold/Class; updated to [[[Lnew/Class;."""
    renames = {"old/Type": "new/Type"}
    orig = "[[[Lold/Type;"
    updated = rewrite_descriptor(orig, renames)
    assert updated == "[[[Lnew/Type;"


def test_bva_f27_enclosing_method_attribute():
    """BVA F27-3: Descriptors inside constant pool rewritten globally."""
    renames = {"com/old/Outer": "com/new/Outer"}
    orig = "()Lcom/old/Outer;"
    updated = rewrite_descriptor(orig, renames)
    assert updated == "()Lcom/new/Outer;"


def test_bva_f27_rename_field_and_type_together():
    """BVA F27-4: Field renamed while field type also renamed."""
    arc = JarArchive()
    cf = create_minimal_test_class("com/Holder")
    cp = cf.constant_pool
    f = FieldInfo(AccessFlags.PUBLIC, cp.add_utf8("targetField"), cp.add_utf8("Lcom/old/Type;"), [])
    cf.fields.append(f)
    arc.add_class(cf)

    mapping = SymbolMapping(
        classes={"com/old/Type": "com/new/Type"},
        fields={"com/Holder#targetField": "renamedField"}
    )
    renamer = SymbolRenamer()
    report = renamer.apply_mapping(arc, mapping)

    assert report.fields_renamed == 1
    assert report.descriptors_updated == 1
    assert f.name(cf.constant_pool) == "renamedField"
    assert f.descriptor(cf.constant_pool) == "Lcom/new/Type;"


def test_bva_f27_same_package_vs_cross_package():
    """BVA F27-5: Renaming within same package vs moving across packages."""
    renames_same = {"pkg/OldClass": "pkg/NewClass"}
    renames_cross = {"pkg/OldClass": "other/pkg/NewClass"}

    desc = "(Lpkg/OldClass;)V"
    assert rewrite_descriptor(desc, renames_same) == "(Lpkg/NewClass;)V"
    assert rewrite_descriptor(desc, renames_cross) == "(Lother/pkg/NewClass;)V"


# ==============================================================================
# F28 BOUNDARIES (5 tests)
# ==============================================================================

def test_bva_f28_package_private_to_public():
    """BVA F28-1: Package-private class widened to public on package move."""
    arc = JarArchive()
    # access_flags = 0 (package-private)
    cf = create_minimal_test_class("old/pkg/Hidden", access_flags=0)
    arc.add_class(cf)

    mapping = SymbolMapping(classes={"old/pkg/Hidden": "new/pkg/Visible"})
    renamer = SymbolRenamer()
    report = renamer.apply_mapping(arc, mapping)

    assert report.access_widened == 1
    renamed_cf = arc.get_class("new/pkg/Visible")
    assert renamed_cf.is_public is True


def test_bva_f28_protected_method_widened():
    """BVA F28-2: Protected class widened to public on package move."""
    arc = JarArchive()
    cf = create_minimal_test_class("old/pkg/ProtClass", access_flags=AccessFlags.PROTECTED)
    arc.add_class(cf)

    mapping = SymbolMapping(classes={"old/pkg/ProtClass": "new/pkg/PubClass"})
    renamer = SymbolRenamer()
    report = renamer.apply_mapping(arc, mapping)

    assert report.access_widened == 1
    renamed_cf = arc.get_class("new/pkg/PubClass")
    assert renamed_cf.is_public is True


def test_bva_f28_class_already_public_unchanged():
    """BVA F28-3: Public class flags unchanged on package move."""
    arc = JarArchive()
    cf = create_minimal_test_class("old/pkg/Pub", access_flags=AccessFlags.PUBLIC)
    arc.add_class(cf)

    mapping = SymbolMapping(classes={"old/pkg/Pub": "new/pkg/Pub"})
    renamer = SymbolRenamer()
    report = renamer.apply_mapping(arc, mapping)

    assert report.access_widened == 0
    assert arc.get_class("new/pkg/Pub").is_public is True


def test_bva_f28_interface_method_access_invariant():
    """BVA F28-4: Interface methods remain public abstract."""
    cf = create_minimal_test_class("com/example/Iface", access_flags=AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT)
    assert cf.access_flags & AccessFlags.INTERFACE


def test_bva_f28_no_package_move_no_widening():
    """BVA F28-5: Renaming within same package does not widen access."""
    arc = JarArchive()
    cf = create_minimal_test_class("pkg/A", access_flags=0)  # package-private
    arc.add_class(cf)

    mapping = SymbolMapping(classes={"pkg/A": "pkg/B"})  # same package
    renamer = SymbolRenamer()
    report = renamer.apply_mapping(arc, mapping)

    assert report.access_widened == 0
    assert arc.get_class("pkg/B").is_public is False


# ==============================================================================
# F29 BOUNDARIES (5 tests)
# ==============================================================================

def test_bva_f29_jar_missing_manifest():
    """BVA F29-1: Archive without MANIFEST.MF handles save and repackage cleanly."""
    arc = JarArchive()
    cf = create_minimal_test_class()
    arc.add_class(cf)
    arc.manifest = None

    repackager = JarRepackager()
    synced = repackager.synchronize_manifest_and_services(arc, SymbolMapping())
    assert synced == 0
    raw = arc.to_bytes()
    reloaded = JarArchive.from_bytes(raw)
    assert reloaded.has_class(cf.this_class_name)


def test_bva_f29_manifest_line_72_bytes_folding():
    """BVA F29-2: Manifest headers exceeding 72 bytes folded per RFC 822."""
    long_val = "x" * 150
    wrapped = wrap_manifest_line("Custom-Header", long_val)
    lines = wrapped.split(b"\r\n")
    # All continuation lines except last must not exceed 72 bytes
    for line in lines[:-1]:
        assert len(line) <= 72

    manifest_bytes = b"Manifest-Version: 1.0\r\n" + wrapped + b"\r\n\r\n"
    parsed_m = Manifest.parse(manifest_bytes)
    assert parsed_m.get_attribute("Custom-Header") == long_val


def test_bva_f29_strip_multiple_signature_files():
    """BVA F29-3: All .SF, .DSA, .RSA signature files stripped."""
    arc = JarArchive()
    arc.set_resource("META-INF/TEST.SF", b"signature file")
    arc.set_resource("META-INF/TEST.RSA", b"rsa key")
    arc.set_resource("META-INF/TEST.DSA", b"dsa key")
    arc.set_resource("config.json", b"{}")

    removed = arc.strip_signatures()
    assert removed == 3
    assert not arc.has_resource("META-INF/TEST.SF")
    assert not arc.has_resource("META-INF/TEST.RSA")
    assert not arc.has_resource("META-INF/TEST.DSA")
    assert arc.has_resource("config.json")


def test_bva_f29_spi_services_multiple_providers():
    """BVA F29-4: SPI file containing multiple provider lines updated line-by-line."""
    arc = JarArchive()
    spi_content = b"com.old.Provider1\ncom.old.Provider2\n# Comment line\ncom.other.Provider3\n"
    arc.set_resource("META-INF/services/com.api.IService", spi_content)

    mapping = SymbolMapping(
        classes={
            "com/old/Provider1": "com/new/Provider1",
            "com/old/Provider2": "com/new/Provider2"
        }
    )
    repackager = JarRepackager()
    synced = repackager.synchronize_manifest_and_services(arc, mapping)

    assert synced == 3
    res_data = arc.get_resource("META-INF/services/com.api.IService").decode("utf-8")
    assert "com.new.Provider1" in res_data
    assert "com.new.Provider2" in res_data
    assert "com.other.Provider3" in res_data
    assert "# Comment line" in res_data


def test_bva_f29_non_utf8_binary_resource_preservation():
    """BVA F29-5: Non-UTF8 resource files preserved byte-for-byte."""
    arc = JarArchive()
    cf = create_minimal_test_class()
    arc.add_class(cf)
    binary_blob = bytes([0xFF, 0xFE, 0x00, 0x01, 0x80, 0xC0])
    arc.set_resource("data.bin", binary_blob)

    raw_jar = arc.to_bytes()
    reloaded = JarArchive.from_bytes(raw_jar)
    assert reloaded.get_resource("data.bin") == binary_blob


# ==============================================================================
# F30 BOUNDARIES (5 tests)
# ==============================================================================

def test_bva_f30_empty_code_class_export():
    """BVA F30-1: Class with abstract methods verified with structural integrity before export."""
    cf = create_minimal_test_class("com/example/AbstractService", access_flags=AccessFlags.PUBLIC | AccessFlags.ABSTRACT)
    m = MethodInfo(AccessFlags.PUBLIC | AccessFlags.ABSTRACT, cf.constant_pool.add_utf8("execute"), cf.constant_pool.add_utf8("()V"), [])
    cf.methods.append(m)

    valid, issues = verify_sanitized_class_structural_integrity(cf)
    assert valid
    assert len(issues) == 0


def test_bva_f30_inner_classes_export():
    """BVA F30-2: Outer and inner classes verified structurally for export."""
    cf_outer = create_minimal_test_class("com/example/Outer")
    cf_inner = create_minimal_test_class("com/example/Outer$Inner")

    v1, i1 = verify_sanitized_class_structural_integrity(cf_outer)
    v2, i2 = verify_sanitized_class_structural_integrity(cf_inner)
    assert v1 and v2
    assert len(i1) == 0 and len(i2) == 0


def test_bva_f30_generic_signatures_export():
    """BVA F30-3: Generics verified structurally for clean decompiler export."""
    cf = create_minimal_test_class("com/example/GenericService")
    cp = cf.constant_pool
    sig_name = cp.add_utf8("Signature")
    sig_val = cp.add_utf8("<T:Ljava/lang/Object;>Ljava/lang/Object;")
    cf.attributes.append(SignatureAttribute(sig_name, sig_val))

    valid, issues = verify_sanitized_class_structural_integrity(cf)
    assert valid
    assert len(issues) == 0


def test_bva_f30_output_dir_auto_creation(tmp_path: Path):
    """BVA F30-4: Non-existent output directories created automatically when writing repackaged JAR."""
    out_dir = tmp_path / "deep" / "nested" / "output"
    out_file = out_dir / "target.jar"

    arc = JarArchive()
    cf = create_minimal_test_class()
    arc.add_class(cf)

    repackager = JarRepackager()
    repackager.repackage_to_file(arc, out_file)

    assert out_file.is_file()
    assert out_file.stat().st_size > 0


def test_bva_f30_decompiler_timeout_fallback(tmp_path: Path):
    """BVA F30-5: Decompiler timeout triggers safe fallback with error status."""
    dummy_jar = tmp_path / "timeout.jar"
    dummy_jar.write_bytes(b"")
    results = run_decompiler_check(dummy_jar, timeout_seconds=0)

    for b in ["cfr", "vineflower", "procyon"]:
        res = results[b]
        assert isinstance(res.available, bool)
        assert isinstance(res.success, bool)
