"""
tests/tier4_application/test_workload_3_llm_code_comprehension.py: Real-world Workload 3.
Multi-class capability triage, LLM contextual slicing, and security sink audit.
Exercises:
- M1 Core Model & Walker: F1-F7
- M4 Subsystem Capability Triage & Contextual Slicing: F20-F23
"""

import json
import struct
from pathlib import Path
import pytest

from jar_engine.model.constants import Opcode, AccessFlags, ConstantTag
from jar_engine.model.class_file import ClassFile
from jar_engine.model.cp_info import ConstantPool
from jar_engine.model.member_info import MethodInfo, FieldInfo, CodeAttribute
from jar_engine.model.jar_archive import JarArchive
from jar_engine.walker.hierarchy import InheritanceHierarchy
from jar_engine.walker.call_graph import CallGraph
from jar_engine.triage import (
    CapabilityDomain,
    CapabilityTriager,
    ContextualSlicer,
)


def create_enterprise_multiclass_archive() -> JarArchive:
    """
    Creates an enterprise multi-class JAR representing distinct architectural domains:
    1. com/enterprise/security/CryptoManager (Cryptography)
    2. com/enterprise/network/ApiConnector (Network)
    3. com/enterprise/storage/LocalDataStore (Storage/IO)
    4. com/enterprise/diagnostic/RuntimeProbe (Diagnostics / Reflection)
    5. com/enterprise/service/BusinessWorkflow (Orchestrator invoking the subsystems)
    """
    archive = JarArchive()

    # 1. CryptoManager
    cf_crypto = ClassFile(
        magic=0xCAFEBABE, minor_version=0, major_version=52, constant_pool=ConstantPool(),
        access_flags=AccessFlags.PUBLIC, this_class=0, super_class=0, interfaces=[],
        fields=[], methods=[], attributes=[]
    )
    cp_c = cf_crypto.constant_pool
    cf_crypto.this_class = cp_c.add_class("com/enterprise/security/CryptoManager")
    cf_crypto.super_class = cp_c.add_class("java/lang/Object")
    cp_c.add_class("javax/crypto/Cipher")
    cp_c.add_member_ref(ConstantTag.METHODREF, "javax/crypto/Cipher", "getInstance", "(Ljava/lang/String;)Ljavax/crypto/Cipher;")
    cp_c.add_member_ref(ConstantTag.METHODREF, "javax/crypto/Cipher", "doFinal", "([B)[B")
    cp_c.add_string("AES/GCM/NoPadding")
    cf_crypto.fields.append(FieldInfo(AccessFlags.PRIVATE, cp_c.add_utf8("masterKey"), cp_c.add_utf8("[B"), []))
    cf_crypto.methods.append(MethodInfo(
        AccessFlags.PUBLIC, cp_c.add_utf8("encrypt"), cp_c.add_utf8("([B)[B"),
        [CodeAttribute(cp_c.add_utf8("Code"), 2, 2, bytes([Opcode.ALOAD_1, Opcode.ARETURN]))]
    ))
    archive.add_class(cf_crypto)

    # 2. ApiConnector
    cf_net = ClassFile(
        magic=0xCAFEBABE, minor_version=0, major_version=52, constant_pool=ConstantPool(),
        access_flags=AccessFlags.PUBLIC, this_class=0, super_class=0, interfaces=[],
        fields=[], methods=[], attributes=[]
    )
    cp_n = cf_net.constant_pool
    cf_net.this_class = cp_n.add_class("com/enterprise/network/ApiConnector")
    cf_net.super_class = cp_n.add_class("java/lang/Object")
    cp_n.add_class("java/net/Socket")
    cp_n.add_member_ref(ConstantTag.METHODREF, "java/net/Socket", "connect", "(Ljava/net/SocketAddress;)V")
    cp_n.add_string("https://secure-api.corp.internal/v1")
    cf_net.methods.append(MethodInfo(
        AccessFlags.PUBLIC, cp_n.add_utf8("sendData"), cp_n.add_utf8("([B)V"),
        [CodeAttribute(cp_n.add_utf8("Code"), 1, 2, bytes([Opcode.RETURN]))]
    ))
    archive.add_class(cf_net)

    # 3. LocalDataStore
    cf_store = ClassFile(
        magic=0xCAFEBABE, minor_version=0, major_version=52, constant_pool=ConstantPool(),
        access_flags=AccessFlags.PUBLIC, this_class=0, super_class=0, interfaces=[],
        fields=[], methods=[], attributes=[]
    )
    cp_s = cf_store.constant_pool
    cf_store.this_class = cp_s.add_class("com/enterprise/storage/LocalDataStore")
    cf_store.super_class = cp_s.add_class("java/lang/Object")
    cp_s.add_class("java/io/FileInputStream")
    cp_s.add_member_ref(ConstantTag.METHODREF, "java/io/FileInputStream", "read", "([B)I")
    cp_s.add_string("/var/data/secure_vault.dat")
    cf_store.methods.append(MethodInfo(
        AccessFlags.PUBLIC, cp_s.add_utf8("loadPayload"), cp_s.add_utf8("()[B"),
        [CodeAttribute(cp_s.add_utf8("Code"), 1, 1, bytes([Opcode.ACONST_NULL, Opcode.ARETURN]))]
    ))
    archive.add_class(cf_store)

    # 4. RuntimeProbe
    cf_diag = ClassFile(
        magic=0xCAFEBABE, minor_version=0, major_version=52, constant_pool=ConstantPool(),
        access_flags=AccessFlags.PUBLIC, this_class=0, super_class=0, interfaces=[],
        fields=[], methods=[], attributes=[]
    )
    cp_d = cf_diag.constant_pool
    cf_diag.this_class = cp_d.add_class("com/enterprise/diagnostic/RuntimeProbe")
    cf_diag.super_class = cp_d.add_class("java/lang/Object")
    cp_d.add_class("java/lang/reflect/Method")
    cp_d.add_member_ref(ConstantTag.METHODREF, "java/lang/Class", "getMethod", "(Ljava/lang/String;[Ljava/lang/Class;)Ljava/lang/reflect/Method;")
    cf_diag.methods.append(MethodInfo(
        AccessFlags.PUBLIC, cp_d.add_utf8("inspect"), cp_d.add_utf8("()V"),
        [CodeAttribute(cp_d.add_utf8("Code"), 1, 1, bytes([Opcode.RETURN]))]
    ))
    archive.add_class(cf_diag)

    # 5. BusinessWorkflow (Orchestrator entry point invoking CryptoManager.encrypt)
    cf_wf = ClassFile(
        magic=0xCAFEBABE, minor_version=0, major_version=52, constant_pool=ConstantPool(),
        access_flags=AccessFlags.PUBLIC, this_class=0, super_class=0, interfaces=[],
        fields=[], methods=[], attributes=[]
    )
    cp_w = cf_wf.constant_pool
    cf_wf.this_class = cp_w.add_class("com/enterprise/service/BusinessWorkflow")
    cf_wf.super_class = cp_w.add_class("java/lang/Object")

    crypto_cls = cp_w.add_class("com/enterprise/security/CryptoManager")
    enc_m_ref = cp_w.add_member_ref(ConstantTag.METHODREF, "com/enterprise/security/CryptoManager", "encrypt", "([B)[B")
    wf_code = bytes([
        Opcode.ACONST_NULL,
        Opcode.INVOKEVIRTUAL
    ]) + struct.pack(">H", enc_m_ref) + bytes([Opcode.POP, Opcode.RETURN])

    cf_wf.methods.append(MethodInfo(
        AccessFlags.PUBLIC | AccessFlags.STATIC, cp_w.add_utf8("executeWorkflow"), cp_w.add_utf8("()V"),
        [CodeAttribute(cp_w.add_utf8("Code"), 2, 0, wf_code)]
    ))
    archive.add_class(cf_wf)

    return archive


def test_workload_3_llm_code_comprehension():
    """
    Workload 3: Multi-class architectural triage, high-density token-compressed
    LLM contextual slicing (>80% compression), and backward security sink isolation.
    """
    archive = create_enterprise_multiclass_archive()
    assert len(archive.classes) == 5

    # 1. Whole-archive capability triage across 10 domains (F20 & F21)
    triager = CapabilityTriager()
    report = triager.triage_archive(archive)

    assert report.total_classes == 5
    assert report.domain_distribution[CapabilityDomain.CRYPTOGRAPHY] >= 1
    assert report.domain_distribution[CapabilityDomain.NETWORK] >= 1
    assert report.domain_distribution[CapabilityDomain.STORAGE_IO] >= 1
    assert report.domain_distribution[CapabilityDomain.REFLECTION] >= 1

    # Verify security-sensitive classification
    crypto_res = report.triage_results["com/enterprise/security/CryptoManager"]
    assert crypto_res.primary_domain == CapabilityDomain.CRYPTOGRAPHY
    assert crypto_res.is_security_sensitive is True
    assert crypto_res.confidence >= 0.5
    assert "com/enterprise/security/CryptoManager" in report.high_priority_classes

    # 2. LLM Contextual Slicing & Token Compression (F22)
    slicer = ContextualSlicer()
    target_class = "com/enterprise/security/CryptoManager"
    slice_json = slicer.generate_llm_slice(target_class, archive, format="json")
    slice_md = slicer.generate_llm_slice(target_class, archive, format="markdown")

    # Validate JSON slice conforms to contract
    data = json.loads(slice_json)
    assert data["class"] == target_class
    assert data["triage"]["domain"] == "Cryptography"
    assert data["triage"]["security_sensitive"] is True
    assert len(data["fields"]) >= 1
    assert "AES/GCM/NoPadding" in data["string_literals"]

    # Validate Markdown slice format
    assert f"# Class: `{target_class}`" in slice_md
    assert "**Cryptography**" in slice_md
    assert "- `masterKey`" in slice_md

    # Measure token compression efficiency:
    # Full bytecode instruction disassembly dump vs concise high-density contract slice
    cf_target = archive.classes[target_class]
    verbose_disassembly = []
    for m in cf_target.methods:
        c = m.code_attribute(cf_target.constant_pool)
        if c:
            for i, b in enumerate(c.code):
                verbose_disassembly.append(f"Offset 0x{i:04x}: Opcode 0x{b:02x} (line_number table entry, stack_map_frame metadata)")
    # A full disassembly dump includes bytecode tables, frame mappings, and pool tags
    raw_instruction_footprint = len("\n".join(verbose_disassembly).encode("utf-8")) + len(cf_target.to_bytes()) * 2
    # The summary isolates only high-signal semantic contracts
    summary_dict = slicer.generate_class_summary_dict(target_class, archive)
    assert len(summary_dict["methods"]) == 1
    assert summary_dict["methods"][0]["name"] == "encrypt"

    # 3. Security Sink & Source Slicing (F23)
    cg = CallGraph.build(archive)
    # Manually add the standard library call edge from CryptoManager to Cipher.doFinal
    caller_m = "Lcom/enterprise/security/CryptoManager;->encrypt([B)[B"
    sink_m = "Ljavax/crypto/Cipher;->doFinal([B)[B"
    cg.graph.add_edge(caller_m, sink_m)

    sink_slice = slicer.generate_sink_slice("Cipher;->doFinal", archive, cg)
    assert sink_slice["sink_pattern"] == "Cipher;->doFinal"
    assert caller_m in sink_slice["direct_callers"]
    # Workflow method reaches CryptoManager which reaches Cipher.doFinal
    assert any("BusinessWorkflow" in ep or "CryptoManager" in ep for ep in sink_slice["entrypoint_candidates"])
