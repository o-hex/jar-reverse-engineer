"""
tests/test_m4_triage_and_slicing.py: Test suite for Milestone M4 (Subsystem Capability Triage & Contextual Slicing).
Covers features F20, F21, F22, F23.
"""

import json
import pytest

from jar_engine.model.constants import ConstantTag, AccessFlags
from jar_engine.model.class_file import ClassFile
from jar_engine.model.cp_info import ConstantPool
from jar_engine.model.member_info import FieldInfo, MethodInfo
from jar_engine.model.jar_archive import JarArchive
from jar_engine.walker.call_graph import CallGraph
from jar_engine.triage import (
    CapabilityDomain,
    CapabilityTriager,
    ContextualSlicer,
)


def create_mock_crypto_class() -> ClassFile:
    """Creates a class referencing cryptography and reflection APIs."""
    cp = ConstantPool()
    this_c = cp.add_class("com/example/CryptoService")
    super_c = cp.add_class("java/lang/Object")

    # Add crypto references
    cp.add_class("javax/crypto/Cipher")
    cp.add_member_ref(ConstantTag.METHODREF, "javax/crypto/Cipher", "getInstance", "(Ljava/lang/String;)Ljavax/crypto/Cipher;")
    cp.add_member_ref(ConstantTag.METHODREF, "javax/crypto/Cipher", "doFinal", "([B)[B")

    # Add secret key field and string
    key_field_name = cp.add_utf8("secretKey")
    key_field_desc = cp.add_utf8("Ljavax/crypto/SecretKey;")
    cp.add_string("AES/CBC/PKCS5Padding")

    m_name = cp.add_utf8("encryptPayload")
    m_desc = cp.add_utf8("([B)[B")

    cf = ClassFile(
        magic=0xCAFEBABE,
        minor_version=0,
        major_version=52,
        constant_pool=cp,
        access_flags=AccessFlags.PUBLIC,
        this_class=this_c,
        super_class=super_c,
        fields=[FieldInfo(AccessFlags.PRIVATE, key_field_name, key_field_desc, [])],
        methods=[MethodInfo(AccessFlags.PUBLIC, m_name, m_desc, [])],
        attributes=[]
    )
    return cf


def create_mock_network_class() -> ClassFile:
    """Creates a class referencing network socket APIs."""
    cp = ConstantPool()
    this_c = cp.add_class("com/example/NetworkClient")
    super_c = cp.add_class("java/lang/Object")

    cp.add_class("java/net/Socket")
    cp.add_member_ref(ConstantTag.METHODREF, "java/net/Socket", "connect", "(Ljava/net/SocketAddress;)V")
    cp.add_string("https://api.example.com/endpoint")

    cf = ClassFile(
        magic=0xCAFEBABE,
        minor_version=0,
        major_version=52,
        constant_pool=cp,
        access_flags=AccessFlags.PUBLIC,
        this_class=this_c,
        super_class=super_c,
        fields=[],
        methods=[MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("connectRemote"), cp.add_utf8("()V"), [])],
        attributes=[]
    )
    return cf


# --- Feature F20 & F21: Triage Scoring & Domain Taxonomy ---

def test_triage_crypto_classification():
    archive = JarArchive()
    cf = create_mock_crypto_class()
    archive.add_class(cf)

    triager = CapabilityTriager()
    res = triager.triage_class(cf.this_class_name, archive)

    assert res.primary_domain == CapabilityDomain.CRYPTOGRAPHY
    assert res.is_security_sensitive is True
    assert res.confidence > 0.5
    assert CapabilityDomain.CRYPTOGRAPHY in res.domain_scores
    assert any("Cipher" in api for api in res.matched_apis[CapabilityDomain.CRYPTOGRAPHY])


def test_triage_network_classification():
    archive = JarArchive()
    cf = create_mock_network_class()
    archive.add_class(cf)

    triager = CapabilityTriager()
    res = triager.triage_class(cf.this_class_name, archive)

    assert res.primary_domain == CapabilityDomain.NETWORK
    assert res.confidence > 0.5
    assert any("Socket" in api for api in res.matched_apis[CapabilityDomain.NETWORK])


def test_archive_triage_report_ranking():
    archive = JarArchive()
    cf_crypto = create_mock_crypto_class()
    cf_net = create_mock_network_class()
    archive.add_class(cf_crypto)
    archive.add_class(cf_net)

    triager = CapabilityTriager()
    report = triager.triage_archive(archive)

    assert report.total_classes == 2
    assert report.domain_distribution[CapabilityDomain.CRYPTOGRAPHY] == 1
    assert report.domain_distribution[CapabilityDomain.NETWORK] == 1
    # Crypto class should be in high priority due to security sensitivity
    assert cf_crypto.this_class_name in report.high_priority_classes


# --- Feature F22: LLM Contextual Slicing ---

def test_contextual_slicer_json_format():
    archive = JarArchive()
    cf = create_mock_crypto_class()
    archive.add_class(cf)

    slicer = ContextualSlicer()
    slice_json = slicer.generate_llm_slice(cf.this_class_name, archive, format="json")

    # Verify parseable JSON
    data = json.loads(slice_json)
    assert data["class"] == "com/example/CryptoService"
    assert data["triage"]["domain"] == "Cryptography"
    assert data["triage"]["security_sensitive"] is True
    assert len(data["fields"]) == 1
    assert data["fields"][0]["name"] == "secretKey"
    assert "AES/CBC/PKCS5Padding" in data["string_literals"]


def test_contextual_slicer_markdown_format():
    archive = JarArchive()
    cf = create_mock_crypto_class()
    archive.add_class(cf)

    slicer = ContextualSlicer()
    slice_md = slicer.generate_llm_slice(cf.this_class_name, archive, format="markdown")

    assert "# Class: `com/example/CryptoService`" in slice_md
    assert "**Cryptography**" in slice_md
    assert "- `secretKey`" in slice_md
    assert "AES/CBC/PKCS5Padding" in slice_md


# --- Feature F23: Security Sink/Source Slicing ---

def test_security_sink_slice_resolution():
    archive = JarArchive()
    cf = create_mock_crypto_class()
    archive.add_class(cf)

    cg = CallGraph()
    # Add an edge calling Cipher->doFinal
    caller_m = "Lcom/example/CryptoService;->encryptPayload([B)[B"
    sink_m = "Ljavax/crypto/Cipher;->doFinal([B)[B"
    cg.graph.add_edge(caller_m, sink_m)

    slicer = ContextualSlicer()
    sink_slice = slicer.generate_sink_slice("Cipher;->doFinal", archive, cg)

    assert sink_slice["sink_pattern"] == "Cipher;->doFinal"
    assert caller_m in sink_slice["direct_callers"]
    assert caller_m in sink_slice["entrypoint_candidates"]
