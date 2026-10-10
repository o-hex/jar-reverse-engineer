"""
tests/tier3_pairwise/test_pairwise_triage_repackaging.py: Pairwise interactions between M4 Triage and M5 Repackaging/Renamer.
Covers:
- F20 (10-Domain Architectural Classifier) + F24 (Symbol Mapping Schema Validation)
- F21 (Multi-Factor Triage Scoring) + F27 (Global Bytecode Reference Rewriting)
- F22 (LLM-Friendly Contextual Slicing) + F30 (Decompiled Source Tree Export Readiness)
- F23 (Security Sink/Source Slicing) + F29 (Valid JAR Repackaging & SPI Sync)
- F20 (10-Domain Architectural Classifier) + F29 (Valid JAR Repackaging)
- F22 (LLM-Friendly Contextual Slicing) + F27 (Global Bytecode Reference Rewriting)
"""

import json
import struct
from pathlib import Path
import pytest

from jar_engine.model.constants import Opcode, AccessFlags, ConstantTag
from jar_engine.model.class_file import ClassFile
from jar_engine.model.cp_info import ConstantPool, CONSTANT_Class_info
from jar_engine.model.member_info import MethodInfo, FieldInfo, CodeAttribute
from jar_engine.model.jar_archive import JarArchive, Manifest
from jar_engine.walker.call_graph import CallGraph
from jar_engine.triage import (
    CapabilityDomain,
    CapabilityTriager,
    ContextualSlicer,
)
from jar_engine.renamer import (
    SymbolRenamer,
    SymbolMapping,
    JarRepackager,
)
from jar_engine.sanitizer import verify_sanitized_class_structural_integrity


def create_crypto_class(name: str = "a/a") -> ClassFile:
    """Helper creating a class with cryptography API usages."""
    cp = ConstantPool()
    this_c = cp.add_class(name)
    super_c = cp.add_class("java/lang/Object")

    cp.add_class("javax/crypto/Cipher")
    cp.add_member_ref(ConstantTag.METHODREF, "javax/crypto/Cipher", "getInstance", "(Ljava/lang/String;)Ljavax/crypto/Cipher;")
    cp.add_member_ref(ConstantTag.METHODREF, "javax/crypto/Cipher", "doFinal", "([B)[B")
    cp.add_string("AES/CBC/PKCS5Padding")

    fld_name = cp.add_utf8("keyBytes")
    fld_desc = cp.add_utf8("[B")
    m_name = cp.add_utf8("encryptPayload")
    m_desc = cp.add_utf8("([B)[B")

    return ClassFile(
        magic=0xCAFEBABE, minor_version=0, major_version=52, constant_pool=cp,
        access_flags=AccessFlags.PUBLIC, this_class=this_c, super_class=super_c, interfaces=[],
        fields=[FieldInfo(AccessFlags.PRIVATE, fld_name, fld_desc, [])],
        methods=[MethodInfo(AccessFlags.PUBLIC, m_name, m_desc, [CodeAttribute(cp.add_utf8("Code"), 2, 2, bytes([Opcode.ALOAD_1, Opcode.ARETURN]))])],
        attributes=[]
    )


def create_network_class(name: str = "com/example/net/RawClient") -> ClassFile:
    """Helper creating a class with socket networking API usages."""
    cp = ConstantPool()
    this_c = cp.add_class(name)
    super_c = cp.add_class("java/lang/Object")

    cp.add_class("java/net/Socket")
    cp.add_member_ref(ConstantTag.METHODREF, "java/net/Socket", "connect", "(Ljava/net/SocketAddress;)V")
    cp.add_string("https://api.example.com")

    return ClassFile(
        magic=0xCAFEBABE, minor_version=0, major_version=52, constant_pool=cp,
        access_flags=AccessFlags.PUBLIC, this_class=this_c, super_class=super_c, interfaces=[],
        fields=[],
        methods=[MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("connectRemote"), cp.add_utf8("()V"), [CodeAttribute(cp.add_utf8("Code"), 1, 1, bytes([Opcode.RETURN]))])],
        attributes=[]
    )


def test_pairwise_f20_f24_triage_domain_symbol_mapping():
    """
    Pairwise F20 + F24: 10-domain architectural classification used to
    generate semantic symbol mapping suggestions adhering to schema validation.
    """
    archive = JarArchive()
    cf = create_crypto_class("a/a")
    archive.add_class(cf)

    # 1. Triage capability domain (F20)
    triager = CapabilityTriager()
    res = triager.triage_class("a/a", archive)
    assert res.primary_domain == CapabilityDomain.CRYPTOGRAPHY
    assert res.is_security_sensitive is True

    # 2. Formulate semantic symbol mapping based on classified domain (F24)
    mapping_data = {
        "classes": {
            "a/a": f"com/example/crypto/CryptoService"
        },
        "fields": {
            "a/a#keyBytes": "cipherKeyBytes"
        },
        "methods": {
            "a/a#encryptPayload#([B)[B": "executeEncryption"
        }
    }
    mapping = SymbolMapping.from_dict(mapping_data)
    validation_errors = mapping.validate()
    assert len(validation_errors) == 0
    assert "com/example/crypto/CryptoService" in mapping.classes.values()


def test_pairwise_f21_f27_tfidf_triage_reference_rewriting():
    """
    Pairwise F21 + F27: TF-IDF capability triage scores guide rename mappings
    applied via global multi-class bytecode reference rewriting.
    """
    archive = JarArchive()
    cf_net = create_network_class("com/example/net/RawClient")
    archive.add_class(cf_net)

    # Caller class referencing RawClient
    cp_caller = ConstantPool()
    net_m_ref = cp_caller.add_member_ref(ConstantTag.METHODREF, "com/example/net/RawClient", "connectRemote", "()V")
    caller_code = bytes([Opcode.INVOKESTATIC]) + struct.pack(">H", net_m_ref) + bytes([Opcode.RETURN])

    cf_caller = ClassFile(
        magic=0xCAFEBABE, minor_version=0, major_version=52, constant_pool=cp_caller,
        access_flags=AccessFlags.PUBLIC, this_class=cp_caller.add_class("com/example/app/MainApp"),
        super_class=cp_caller.add_class("java/lang/Object"), interfaces=[],
        fields=[],
        methods=[MethodInfo(
            AccessFlags.PUBLIC | AccessFlags.STATIC, cp_caller.add_utf8("main"), cp_caller.add_utf8("([Ljava/lang/String;)V"),
            [CodeAttribute(cp_caller.add_utf8("Code"), 1, 1, caller_code)]
        )],
        attributes=[]
    )
    archive.add_class(cf_caller)

    # 1. TF-IDF triage scoring (F21)
    triager = CapabilityTriager()
    report = triager.triage_archive(archive)
    assert report.domain_distribution[CapabilityDomain.NETWORK] >= 1

    # 2. Rename based on highest-scored capability (F27)
    mapping = SymbolMapping.from_dict({
        "classes": {"com/example/net/RawClient": "com/example/net/NetworkConnectionManager"}
    })
    renamer = SymbolRenamer()
    ren_report = renamer.apply_mapping(archive, mapping)
    assert ren_report.classes_renamed >= 1

    # Verify caller constant pool was rewritten
    caller_classes = [
        cf_caller.resolve_utf8(entry.name_index)
        for _, entry in cf_caller.constant_pool.entries()
        if isinstance(entry, CONSTANT_Class_info)
    ]
    assert "com/example/net/NetworkConnectionManager" in caller_classes
    assert "com/example/net/RawClient" not in caller_classes


def test_pairwise_f22_f30_llm_slices_decompiled_source():
    """
    Pairwise F22 + F30: Contextual LLM slices generated alongside
    verification of class structural integrity for source export.
    """
    archive = JarArchive()
    cf = create_crypto_class("com/example/CryptoProcessor")
    archive.add_class(cf)

    # 1. Generate LLM slices (F22)
    slicer = ContextualSlicer()
    slice_json = slicer.generate_llm_slice("com/example/CryptoProcessor", archive, format="json")
    slice_md = slicer.generate_llm_slice("com/example/CryptoProcessor", archive, format="markdown")

    data = json.loads(slice_json)
    assert data["class"] == "com/example/CryptoProcessor"
    assert data["triage"]["domain"] == "Cryptography"
    assert "# Class: `com/example/CryptoProcessor`" in slice_md

    # 2. Verify structural integrity for clean decompiler source export (F30)
    is_valid, issues = verify_sanitized_class_structural_integrity(cf)
    assert is_valid is True
    assert len(issues) == 0


def test_pairwise_f23_f29_security_sinks_repackaging(temp_workspace: Path):
    """
    Pairwise F23 + F29: Security sink slices extracted and archive repackaged
    with synchronized RFC 822 manifest and SPI service metadata.
    """
    archive = JarArchive()
    cf = create_crypto_class("com/example/SecService")
    archive.add_class(cf)

    # Add manifest and SPI provider
    manifest = Manifest()
    manifest.set_main_class("com.example.SecService")
    archive.manifest = manifest
    archive.set_resource("META-INF/services/com.example.ISecService", b"com.example.SecService\n")

    # 1. Build CallGraph and extract security sink slice (F23)
    cg = CallGraph()
    caller_m = "Lcom/example/SecService;->encryptPayload([B)[B"
    sink_m = "Ljavax/crypto/Cipher;->doFinal([B)[B"
    cg.graph.add_edge(caller_m, sink_m)

    slicer = ContextualSlicer()
    sink_data = slicer.generate_sink_slice("Cipher;->doFinal", archive, cg)
    assert sink_data["sink_pattern"] == "Cipher;->doFinal"
    assert caller_m in sink_data["direct_callers"]

    # 2. Repackage archive to disk (F29)
    out_jar = temp_workspace / "repackaged_security.jar"
    repackager = JarRepackager()
    repackager.repackage_to_file(archive, out_jar)
    assert out_jar.is_file() and out_jar.stat().st_size > 0

    # Reload repackaged archive and verify metadata
    reloaded = JarArchive.from_file(out_jar)
    assert reloaded.manifest is not None
    assert reloaded.manifest.get_main_class() == "com.example.SecService"
    assert "META-INF/services/com.example.ISecService" in reloaded.resources


def test_pairwise_f20_f29_triage_metadata_repackaged_jar(temp_workspace: Path):
    """
    Pairwise F20 + F29: Triage capability reports serialized into
    repackaged JAR resources and preserved across file roundtrip.
    """
    archive = JarArchive()
    archive.add_class(create_crypto_class("com/example/Crypto1"))
    archive.add_class(create_network_class("com/example/Net1"))

    # 1. Triage whole archive (F20)
    triager = CapabilityTriager()
    report = triager.triage_archive(archive)

    # Serialize triage summary as metadata resource in JAR
    summary = {
        "total_classes": report.total_classes,
        "high_priority": report.high_priority_classes,
        "domains": {k.value: v for k, v in report.domain_distribution.items()}
    }
    archive.set_resource("META-INF/triage_summary.json", json.dumps(summary, indent=2).encode("utf-8"))

    # 2. Repackage archive (F29)
    out_jar = temp_workspace / "triage_packaged.jar"
    archive.save(out_jar)
    assert out_jar.is_file()

    # Verify resource roundtrip
    reloaded = JarArchive.from_file(out_jar)
    assert "META-INF/triage_summary.json" in reloaded.resources
    loaded_summary = json.loads(reloaded.resources["META-INF/triage_summary.json"].decode("utf-8"))
    assert loaded_summary["total_classes"] == 2
    assert "Cryptography" in loaded_summary["domains"]
    assert "Network" in loaded_summary["domains"]


def test_pairwise_f22_f27_llm_slices_post_renaming():
    """
    Pairwise F22 + F27: Contextual LLM slices generated before and after
    symbol renaming reflect newly refactored class and method names.
    """
    archive = JarArchive()
    cf = create_crypto_class("com/example/LegacyCrypto")
    archive.add_class(cf)

    slicer = ContextualSlicer()
    # Slice before renaming
    slice_before = slicer.generate_llm_slice("com/example/LegacyCrypto", archive, format="json")
    data_before = json.loads(slice_before)
    assert data_before["class"] == "com/example/LegacyCrypto"

    # Apply symbol renaming (F27)
    mapping = SymbolMapping.from_dict({
        "classes": {"com/example/LegacyCrypto": "com/example/ModernCryptoService"}
    })
    renamer = SymbolRenamer()
    report = renamer.apply_mapping(archive, mapping)
    assert report.classes_renamed == 1

    # Slice after renaming
    slice_after = slicer.generate_llm_slice("com/example/ModernCryptoService", archive, format="json")
    data_after = json.loads(slice_after)
    assert data_after["class"] == "com/example/ModernCryptoService"
    assert data_after["triage"]["domain"] == "Cryptography"
