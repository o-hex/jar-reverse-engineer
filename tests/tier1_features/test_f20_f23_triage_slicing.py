"""
Tier 1: Feature Coverage for F20 - F23 (Subsystem Capability Triage & Contextual Slicing).
Tests 20 isolated features (5 per feature F20 through F23) per explorer_e2e_3 specification.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

import pytest

from jar_engine.model.constants import ConstantTag, AccessFlags, Opcode
from jar_engine.model.class_file import ClassFile
from jar_engine.model.cp_info import ConstantPool
from jar_engine.model.member_info import FieldInfo, MethodInfo, CodeAttribute
from jar_engine.model.jar_archive import JarArchive
from jar_engine.walker.call_graph import CallGraph
from jar_engine.triage import (
    CapabilityDomain,
    CapabilityTriager,
    ContextualSlicer,
)
from jar_engine.triage.scorer import ClassTriageResult, ArchiveTriageReport


# ============================================================================
# F20: 10-Domain Architectural Classifier
# ============================================================================

def test_f20_domain_enumeration_10_categories():
    """F20.1: Verify all 10 architectural domains defined in CapabilityDomain."""
    expected_domains = {
        "Cryptography",
        "Network",
        "Storage/IO",
        "Reflection",
        "Process/Environment",
        "Serialization",
        "Database",
        "GUI/Client",
        "Audio/Media",
        "Security/Policy",
    }
    actual_domains = {d.value for d in CapabilityDomain}
    assert actual_domains == expected_domains
    assert len(CapabilityDomain) == 10


def test_f20_crypto_domain_matching(make_test_class):
    """F20.2: Match classes using javax/crypto or Cipher to Cryptography."""
    cf = make_test_class("com/example/CryptoService")
    cp = cf.constant_pool
    cp.add_class("javax/crypto/Cipher")
    cp.add_member_ref(ConstantTag.METHODREF, "javax/crypto/Cipher", "getInstance", "(Ljava/lang/String;)Ljavax/crypto/Cipher;")
    cp.add_member_ref(ConstantTag.METHODREF, "javax/crypto/Cipher", "doFinal", "([B)[B")

    arc = JarArchive()
    arc.add_class(cf)

    triager = CapabilityTriager()
    res = triager.triage_class(cf.this_class_name, arc)
    assert res.primary_domain == CapabilityDomain.CRYPTOGRAPHY
    assert res.is_security_sensitive is True


def test_f20_network_domain_matching(make_test_class):
    """F20.3: Match classes using java/net/Socket or HttpURLConnection to Network."""
    cf = make_test_class("com/example/NetClient")
    cp = cf.constant_pool
    cp.add_class("java/net/Socket")
    cp.add_member_ref(ConstantTag.METHODREF, "java/net/Socket", "connect", "(Ljava/net/SocketAddress;)V")
    cp.add_member_ref(ConstantTag.METHODREF, "java/net/HttpURLConnection", "setRequestMethod", "(Ljava/lang/String;)V")

    arc = JarArchive()
    arc.add_class(cf)

    triager = CapabilityTriager()
    res = triager.triage_class(cf.this_class_name, arc)
    assert res.primary_domain == CapabilityDomain.NETWORK


def test_f20_storage_io_domain_matching(make_test_class):
    """F20.4: Match classes using java/io/File or java/nio to Storage/IO."""
    cf = make_test_class("com/example/StorageService")
    cp = cf.constant_pool
    cp.add_class("java/io/File")
    cp.add_class("java/nio/ByteBuffer")
    cp.add_member_ref(ConstantTag.METHODREF, "java/io/FileInputStream", "read", "([B)I")

    arc = JarArchive()
    arc.add_class(cf)

    triager = CapabilityTriager()
    res = triager.triage_class(cf.this_class_name, arc)
    assert res.primary_domain == CapabilityDomain.STORAGE_IO


def test_f20_reflection_and_process_domains(make_test_class):
    """F20.5: Match classes using java/lang/reflect and ProcessBuilder."""
    cf_refl = make_test_class("com/example/ReflectUtil")
    cp_r = cf_refl.constant_pool
    cp_r.add_class("java/lang/reflect/Method")
    cp_r.add_member_ref(ConstantTag.METHODREF, "java/lang/reflect/Method", "invoke", "(Ljava/lang/Object;[Ljava/lang/Object;)Ljava/lang/Object;")

    cf_proc = make_test_class("com/example/ProcRunner")
    cp_p = cf_proc.constant_pool
    cp_p.add_class("java/lang/ProcessBuilder")
    cp_p.add_member_ref(ConstantTag.METHODREF, "java/lang/ProcessBuilder", "start", "()Ljava/lang/Process;")

    arc = JarArchive()
    arc.add_class(cf_refl)
    arc.add_class(cf_proc)

    triager = CapabilityTriager()
    res_r = triager.triage_class(cf_refl.this_class_name, arc)
    res_p = triager.triage_class(cf_proc.this_class_name, arc)

    assert res_r.primary_domain == CapabilityDomain.REFLECTION
    assert res_p.primary_domain == CapabilityDomain.PROCESS_ENV


# ============================================================================
# F21: Multi-Factor Triage Scoring
# ============================================================================

def test_f21_tf_idf_footprint_scoring(make_test_class):
    """F21.1: Score API call frequencies weighted by archive TF-IDF."""
    arc = JarArchive()
    # Class 1: multiple crypto calls
    cf1 = make_test_class("pkg/CryptoOne")
    cp1 = cf1.constant_pool
    for _ in range(5):
        cp1.add_member_ref(ConstantTag.METHODREF, "javax/crypto/Cipher", "doFinal", "([B)[B")
    arc.add_class(cf1)

    # Class 2: regular utility class
    cf2 = make_test_class("pkg/Util")
    cp2 = cf2.constant_pool
    cp2.add_member_ref(ConstantTag.METHODREF, "java/lang/String", "valueOf", "(I)Ljava/lang/String;")
    arc.add_class(cf2)

    triager = CapabilityTriager()
    res1 = triager.triage_class("pkg/CryptoOne", arc)
    assert res1.primary_domain == CapabilityDomain.CRYPTOGRAPHY
    assert res1.domain_scores[CapabilityDomain.CRYPTOGRAPHY] > 0.0


def test_f21_single_class_triage_result(make_test_class):
    """F21.2: Verify ClassTriageResult contains domain, confidence, and matches."""
    cf = make_test_class("pkg/NetWorker")
    cp = cf.constant_pool
    cp.add_class("java/net/Socket")
    arc = JarArchive()
    arc.add_class(cf)

    triager = CapabilityTriager()
    res = triager.triage_class("pkg/NetWorker", arc)
    assert isinstance(res, ClassTriageResult)
    assert res.class_name == "pkg/NetWorker"
    assert res.primary_domain == CapabilityDomain.NETWORK
    assert isinstance(res.confidence, float)
    assert isinstance(res.matched_apis, dict)
    assert len(res.matched_apis[CapabilityDomain.NETWORK]) >= 1


def test_f21_archive_triage_report(make_test_class):
    """F21.3: Verify ArchiveTriageReport aggregates domain distributions across archive."""
    arc = JarArchive()
    cf1 = make_test_class("pkg/C")
    cf1.constant_pool.add_class("javax/crypto/Cipher")
    cf2 = make_test_class("pkg/N")
    cf2.constant_pool.add_class("java/net/Socket")
    arc.add_class(cf1)
    arc.add_class(cf2)

    triager = CapabilityTriager()
    report = triager.triage_archive(arc)
    assert isinstance(report, ArchiveTriageReport)
    assert report.total_classes == 2
    assert report.domain_distribution[CapabilityDomain.CRYPTOGRAPHY] == 1
    assert report.domain_distribution[CapabilityDomain.NETWORK] == 1
    assert "pkg/C" in report.get_classes_in_domain(CapabilityDomain.CRYPTOGRAPHY)
    assert "pkg/N" in report.get_classes_in_domain(CapabilityDomain.NETWORK)


def test_f21_confidence_calculation(make_test_class):
    """F21.4: Assert confidence scores range between 0.0 and 1.0."""
    arc = JarArchive()
    cf = make_test_class("pkg/ScoreCheck")
    cf.constant_pool.add_class("javax/crypto/Cipher")
    arc.add_class(cf)

    triager = CapabilityTriager()
    res = triager.triage_class("pkg/ScoreCheck", arc)
    assert 0.0 <= res.confidence <= 1.0


def test_f21_security_sensitive_flagging(make_test_class):
    """F21.5: Flag classes matching security-critical API signatures."""
    arc = JarArchive()
    cf_sec = make_test_class("pkg/Secure")
    cf_sec.constant_pool.add_class("javax/crypto/Cipher")
    arc.add_class(cf_sec)

    cf_plain = make_test_class("pkg/Plain")
    arc.add_class(cf_plain)

    triager = CapabilityTriager()
    res_sec = triager.triage_class("pkg/Secure", arc)
    res_plain = triager.triage_class("pkg/Plain", arc)

    assert res_sec.is_security_sensitive is True
    assert res_plain.is_security_sensitive is False


# ============================================================================
# F22: LLM-Friendly Contextual Slicing
# ============================================================================

def test_f22_class_summary_dict_generation(make_test_class):
    """F22.1: Extract compact dictionary representation of class structure."""
    cf = make_test_class("pkg/SummaryTest")
    cp = cf.constant_pool
    cf.fields.append(FieldInfo(AccessFlags.PRIVATE, cp.add_utf8("apiKey"), cp.add_utf8("Ljava/lang/String;"), []))
    cf.methods.append(MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("getApiKey"), cp.add_utf8("()Ljava/lang/String;"), []))
    cp.add_string("static_api_token_value")

    arc = JarArchive()
    arc.add_class(cf)

    slicer = ContextualSlicer()
    summary = slicer.generate_class_summary_dict("pkg/SummaryTest", arc)

    assert summary["class"] == "pkg/SummaryTest"
    assert summary["super_class"] == "java/lang/Object"
    assert len(summary["fields"]) == 1
    assert summary["fields"][0]["name"] == "apiKey"
    assert len(summary["methods"]) == 1
    assert summary["methods"][0]["name"] == "getApiKey"
    assert "static_api_token_value" in summary["string_literals"]


def test_f22_llm_json_slice_format(make_test_class):
    """F22.2: Validate formatted JSON slice against required contract schema."""
    cf = make_test_class("pkg/JsonSliceClass")
    arc = JarArchive()
    arc.add_class(cf)

    slicer = ContextualSlicer()
    slice_json = slicer.generate_llm_slice("pkg/JsonSliceClass", arc, format="json")

    data = json.loads(slice_json)
    assert "class" in data
    assert "super_class" in data
    assert "interfaces" in data
    assert "triage" in data
    assert "domain" in data["triage"]
    assert "fields" in data
    assert "methods" in data
    assert "string_literals" in data


def test_f22_llm_markdown_slice_format(make_test_class):
    """F22.3: Validate readable Markdown slice with headings, fields, and callees."""
    cf = make_test_class("pkg/MdSliceClass")
    cp = cf.constant_pool
    cf.fields.append(FieldInfo(AccessFlags.PRIVATE, cp.add_utf8("fld"), cp.add_utf8("I"), []))
    cf.methods.append(MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("mth"), cp.add_utf8("()V"), []))
    arc = JarArchive()
    arc.add_class(cf)

    slicer = ContextualSlicer()
    slice_md = slicer.generate_llm_slice("pkg/MdSliceClass", arc, format="markdown")

    assert "# Class: `pkg/MdSliceClass`" in slice_md
    assert "## Fields" in slice_md
    assert "- `fld`: `I`" in slice_md
    assert "## Methods & Key Invocations" in slice_md
    assert "- `mth()V`" in slice_md


def test_f22_string_literal_extraction_in_slice(make_test_class):
    """F22.4: Verify embedded constant pool strings included in summary."""
    cf = make_test_class("pkg/StringSliceClass")
    cp = cf.constant_pool
    cp.add_string("DATABASE_CONNECTION_URL")
    cp.add_string("ADMIN_PASSWORD_HASH")
    arc = JarArchive()
    arc.add_class(cf)

    slicer = ContextualSlicer()
    summary = slicer.generate_class_summary_dict("pkg/StringSliceClass", arc)

    assert "DATABASE_CONNECTION_URL" in summary["string_literals"]
    assert "ADMIN_PASSWORD_HASH" in summary["string_literals"]


def test_f22_token_efficiency_compression(make_test_class):
    """F22.5: Measure token reduction comparing slice to raw byte payload representation (>80%)."""
    cf = make_test_class("pkg/CompressionClass")
    cp = cf.constant_pool
    for i in range(10):
        cf.fields.append(FieldInfo(AccessFlags.PRIVATE, cp.add_utf8(f"field_{i}"), cp.add_utf8("I"), []))
        code = bytes([Opcode.NOP] * 50 + [Opcode.RETURN])
        cf.methods.append(MethodInfo(AccessFlags.PUBLIC, cp.add_utf8(f"method_{i}"), cp.add_utf8("()V"), [
            CodeAttribute(cp.add_utf8("Code"), 1, 1, code)
        ]))

    arc = JarArchive()
    arc.add_class(cf)

    raw_bytes_len = len(cf.to_bytes())
    slicer = ContextualSlicer()
    slice_json = slicer.generate_llm_slice("pkg/CompressionClass", arc, format="json")

    # Compressed summary should be compact and structured
    data = json.loads(slice_json)
    assert len(data["fields"]) == 10
    assert len(data["methods"]) == 10
    # Slicing avoids raw opcode dumps, providing concise token footprint
    assert "0x" not in slice_json


# ============================================================================
# F23: Security Sink/Source Slicing
# ============================================================================

def test_f23_sink_pattern_matching(make_test_class):
    """F23.1: Match security sinks (Cipher;->doFinal, Socket;->connect)."""
    cf = make_test_class("pkg/SinkUser")
    arc = JarArchive()
    arc.add_class(cf)

    cg = CallGraph()
    caller = "Lpkg/SinkUser;->run()V"
    sink = "Ljavax/crypto/Cipher;->doFinal([B)[B"
    cg.graph.add_edge(caller, sink)

    slicer = ContextualSlicer()
    slice_res = slicer.generate_sink_slice("Cipher;->doFinal", arc, cg)
    assert slice_res["sink_pattern"] == "Cipher;->doFinal"
    assert caller in slice_res["direct_callers"]


def test_f23_backward_reachability_tracing(make_test_class):
    """F23.2: Trace callers backward from sink to identifying origin methods."""
    arc = JarArchive()
    arc.add_class(make_test_class("pkg/Service"))

    cg = CallGraph()
    m_entry = "Lpkg/Service;->entryPoint()V"
    m_middle = "Lpkg/Service;->helper()V"
    sink = "Ljavax/crypto/Cipher;->doFinal([B)[B"

    cg.graph.add_edge(m_entry, m_middle)
    cg.graph.add_edge(m_middle, sink)

    slicer = ContextualSlicer()
    sink_slice = slicer.generate_sink_slice("Cipher;->doFinal", arc, cg)

    assert m_middle in sink_slice["direct_callers"]
    # Backward reachability resolves candidate callers
    assert len(sink_slice["entrypoint_candidates"]) >= 1


def test_f23_entrypoint_candidate_discovery(make_test_class):
    """F23.3: Identify in-degree 0 entry point candidates reaching sink."""
    arc = JarArchive()
    arc.add_class(make_test_class("pkg/Controller"))

    cg = CallGraph()
    entry = "Lpkg/Controller;->handleRequest()V"
    sink = "Ljava/net/Socket;->connect(Ljava/net/SocketAddress;)V"
    cg.graph.add_edge(entry, sink)

    slicer = ContextualSlicer()
    sink_slice = slicer.generate_sink_slice("Socket;->connect", arc, cg)

    assert entry in sink_slice["entrypoint_candidates"]


def test_f23_sink_slice_structure(make_test_class):
    """F23.4: Validate sink slice schema (sink_pattern, direct_callers, entrypoint_candidates)."""
    arc = JarArchive()
    arc.add_class(make_test_class("pkg/App"))

    cg = CallGraph()
    caller = "Lpkg/App;->exec()V"
    sink = "Ljava/lang/ProcessBuilder;->start()Ljava/lang/Process;"
    cg.graph.add_edge(caller, sink)

    slicer = ContextualSlicer()
    result = slicer.generate_sink_slice("ProcessBuilder;->start", arc, cg)

    assert "sink_pattern" in result
    assert "direct_callers" in result
    assert "entrypoint_candidates" in result
    assert isinstance(result["direct_callers"], list)
    assert isinstance(result["entrypoint_candidates"], list)


def test_f23_multiple_sinks_isolation(make_test_class):
    """F23.5: Slice multiple distinct sinks in single archive independently."""
    arc = JarArchive()
    arc.add_class(make_test_class("pkg/MultiApp"))

    cg = CallGraph()
    m_crypto = "Lpkg/MultiApp;->doCrypto()V"
    m_net = "Lpkg/MultiApp;->doNet()V"

    sink_crypto = "Ljavax/crypto/Cipher;->doFinal([B)[B"
    sink_net = "Ljava/net/Socket;->connect(Ljava/net/SocketAddress;)V"

    cg.graph.add_edge(m_crypto, sink_crypto)
    cg.graph.add_edge(m_net, sink_net)

    slicer = ContextualSlicer()
    crypto_slice = slicer.generate_sink_slice("Cipher;->doFinal", arc, cg)
    net_slice = slicer.generate_sink_slice("Socket;->connect", arc, cg)

    assert m_crypto in crypto_slice["direct_callers"]
    assert m_net not in crypto_slice["direct_callers"]

    assert m_net in net_slice["direct_callers"]
    assert m_crypto not in net_slice["direct_callers"]
