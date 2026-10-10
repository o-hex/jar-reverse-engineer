"""
tests/test_m5_renamer_and_cli.py: Test suite for Milestone M5 (Symbol Restoration, Repackaging, and CLI).
Covers features F24, F25, F26, F27, F28, F29, F34.
"""

import pytest
from pathlib import Path

from jar_engine.model.constants import AccessFlags, ConstantTag
from jar_engine.model.class_file import ClassFile
from jar_engine.model.cp_info import ConstantPool
from jar_engine.model.member_info import FieldInfo, MethodInfo
from jar_engine.model.jar_archive import JarArchive, Manifest
from jar_engine.renamer import (
    SymbolMapping,
    MappingValidationError,
    SymbolRenamer,
    MethodEquivalenceCluster,
    JarRepackager,
    rewrite_descriptor,
)
from jar_engine.cli.main import build_parser


def create_test_interface_and_impl() -> JarArchive:
    """Creates an interface and implementing class in an archive."""
    archive = JarArchive()

    # 1. Interface Service: void execute(String s)
    cp_iface = ConstantPool()
    c_iface = cp_iface.add_class("com/example/IService")
    cp_iface.add_class("java/lang/Object")
    m_name = cp_iface.add_utf8("execute")
    m_desc = cp_iface.add_utf8("(Ljava/lang/String;)V")
    iface_cf = ClassFile(
        magic=0xCAFEBABE,
        minor_version=0,
        major_version=52,
        constant_pool=cp_iface,
        access_flags=AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT,
        this_class=c_iface,
        super_class=cp_iface.add_class("java/lang/Object"),
        methods=[MethodInfo(AccessFlags.PUBLIC | AccessFlags.ABSTRACT, m_name, m_desc, [])]
    )
    archive.add_class(iface_cf)

    # 2. Implementation Class: Impl implements Service
    cp_impl = ConstantPool()
    c_impl = cp_impl.add_class("com/example/ServiceImpl")
    c_obj = cp_impl.add_class("java/lang/Object")
    c_iface_ref = cp_impl.add_class("com/example/IService")
    f_name = cp_impl.add_utf8("tag")
    f_desc = cp_impl.add_utf8("Lcom/example/IService;")
    m_impl_name = cp_impl.add_utf8("execute")
    m_impl_desc = cp_impl.add_utf8("(Ljava/lang/String;)V")

    impl_cf = ClassFile(
        magic=0xCAFEBABE,
        minor_version=0,
        major_version=52,
        constant_pool=cp_impl,
        access_flags=AccessFlags.PUBLIC,
        this_class=c_impl,
        super_class=c_obj,
        interfaces=[c_iface_ref],
        fields=[FieldInfo(AccessFlags.PRIVATE, f_name, f_desc, [])],
        methods=[MethodInfo(AccessFlags.PUBLIC, m_impl_name, m_impl_desc, [])]
    )
    archive.add_class(impl_cf)

    # Add manifest
    manifest = Manifest()
    manifest.set_main_class("com.example.ServiceImpl")
    archive.manifest = manifest

    # Add SPI provider registration
    archive.set_resource(
        "META-INF/services/com.example.IService",
        b"com.example.ServiceImpl\n"
    )

    return archive


# --- Feature F24: Schema Validation ---

def test_symbol_mapping_validation():
    valid_map = SymbolMapping(
        classes={"com/example/Old": "com/example/NewClass"},
        fields={"com/example/Old#f": "newField"},
        methods={"com/example/Old#m#()V": "newMethod"}
    )
    assert len(valid_map.validate()) == 0

    # Invalid names
    invalid_map = SymbolMapping(
        classes={"a": "123InvalidClass"},
        fields={"a#f": "bad-field-name"}
    )
    errors = invalid_map.validate()
    assert len(errors) == 2


# --- Feature F25 & F26: Union-Find Override Equivalence & JDK Pinning ---

def test_union_find_override_clustering():
    archive = create_test_interface_and_impl()
    hier = archive.get_hierarchy()

    cluster = MethodEquivalenceCluster(hier)
    cluster.build_from_archive(archive)

    # Both interface execute() and impl execute() should resolve to the same root
    iface_key = "com/example/IService#execute#(Ljava/lang/String;)V"
    impl_key = "com/example/ServiceImpl#execute#(Ljava/lang/String;)V"

    assert cluster._find(iface_key) == cluster._find(impl_key)


def test_jdk_immutability_pinning():
    archive = create_test_interface_and_impl()
    hier = archive.get_hierarchy()

    # Add toString() to Impl
    impl_cf = archive.get_class("com/example/ServiceImpl")
    cp = impl_cf.constant_pool
    impl_cf.methods.append(
        MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("toString"), cp.add_utf8("()Ljava/lang/String;"), [])
    )

    cluster = MethodEquivalenceCluster(hier)
    cluster.build_from_archive(archive)

    to_string_key = "com/example/ServiceImpl#toString#()Ljava/lang/String;"
    assert cluster.is_method_pinned(to_string_key) is True


# --- Feature F27, F28, F29: Reference Rewriting & Repackaging ---

def test_symbol_renamer_end_to_end():
    archive = create_test_interface_and_impl()

    mapping = SymbolMapping(
        classes={
            "com/example/IService": "org/api/DataService",
            "com/example/ServiceImpl": "org/api/impl/DataServiceImpl"
        },
        fields={
            "com/example/ServiceImpl#tag": "serviceTag"
        },
        methods={
            "com/example/IService#execute#(Ljava/lang/String;)V": "processData"
        }
    )

    renamer = SymbolRenamer()
    report = renamer.apply_mapping(archive, mapping)

    # 1. Classes renamed
    assert report.classes_renamed == 2
    assert "org/api/DataService" in archive.classes
    assert "org/api/impl/DataServiceImpl" in archive.classes
    assert "com/example/IService" not in archive.classes

    # 2. Methods renamed across interface and implementation
    iface_cf = archive.get_class("org/api/DataService")
    assert iface_cf.methods[0].name(iface_cf.constant_pool) == "processData"

    impl_cf = archive.get_class("org/api/impl/DataServiceImpl")
    assert impl_cf.methods[0].name(impl_cf.constant_pool) == "processData"

    # 3. Field and descriptor updated
    assert impl_cf.fields[0].name(impl_cf.constant_pool) == "serviceTag"
    assert impl_cf.fields[0].descriptor(impl_cf.constant_pool) == "Lorg/api/DataService;"

    # 4. Manifest Main-Class synchronized
    assert archive.manifest.get_main_class() == "org.api.impl.DataServiceImpl"

    # 5. SPI provider resource file synchronized
    assert "META-INF/services/org.api.DataService" in archive.resources
    assert "META-INF/services/com.example.IService" not in archive.resources
    spi_text = archive.resources["META-INF/services/org.api.DataService"].decode("utf-8")
    assert "org.api.impl.DataServiceImpl" in spi_text


def test_rewrite_descriptor_utility():
    renames = {"a/b/Foo": "com/example/NewFoo", "x/y/Bar": "org/test/NewBar"}
    desc = "(La/b/Foo;I)Lx/y/Bar;"
    new_desc = rewrite_descriptor(desc, renames)
    assert new_desc == "(Lcom/example/NewFoo;I)Lorg/test/NewBar;"


# --- Feature F34: CLI Parser ---

def test_cli_parser_commands():
    parser = build_parser()
    args_analyze = parser.parse_args(["analyze", "sample.jar"])
    assert args_analyze.command == "analyze"
    assert args_analyze.jar == "sample.jar"

    args_norm = parser.parse_args(["normalize", "in.jar", "-o", "out.jar"])
    assert args_norm.command == "normalize"
    assert args_norm.output == "out.jar"

    args_rename = parser.parse_args(["rename", "in.jar", "-m", "map.json"])
    assert args_rename.command == "rename"
    assert args_rename.mapping == "map.json"
