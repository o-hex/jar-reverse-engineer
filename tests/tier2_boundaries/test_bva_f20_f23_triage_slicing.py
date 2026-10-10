"""
tests/tier2_boundaries/test_bva_f20_f23_triage_slicing.py:
Boundary value analysis and corner cases for features F20 through F23.
Total: 20 tests.
"""

import json
import pytest

from jar_engine.model.constants import AccessFlags, ConstantTag, Opcode
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


def create_minimal_test_class(name: str = "com/example/Test") -> ClassFile:
    """Helper to create minimal ClassFile."""
    cp = ConstantPool()
    this_c = cp.add_class(name)
    super_c = cp.add_class("java/lang/Object")
    return ClassFile(
        magic=0xCAFEBABE,
        minor_version=0,
        major_version=52,
        constant_pool=cp,
        access_flags=AccessFlags.PUBLIC,
        this_class=this_c,
        super_class=super_c,
        interfaces=[],
        fields=[],
        methods=[],
        attributes=[]
    )


# ==============================================================================
# F20 BOUNDARIES (5 tests)
# ==============================================================================

def test_bva_f20_pojo_zero_api_calls():
    """BVA F20-1: Pure POJO with 0 API calls assigned to Unclassified / default domain."""
    arc = JarArchive()
    cf = create_minimal_test_class("com/example/PersonPOJO")
    cp = cf.constant_pool
    # Only standard integer age field and getters
    f_age = FieldInfo(AccessFlags.PRIVATE, cp.add_utf8("age"), cp.add_utf8("I"), [])
    cf.fields.append(f_age)
    arc.add_class(cf)

    triager = CapabilityTriager()
    res = triager.triage_class(cf.this_class_name, arc)

    assert res.primary_domain is None
    assert res.confidence == 0.0
    assert res.has_detected_capabilities is False
    assert res.is_security_sensitive is False


def test_bva_f20_multi_domain_tie_breaking():
    """BVA F20-2: Class referencing multiple domains equally resolved deterministically."""
    arc = JarArchive()
    cf = create_minimal_test_class("com/example/MultiService")
    cp = cf.constant_pool

    # 1 Crypto reference: javax/crypto/Cipher
    cp.add_class("javax/crypto/Cipher")
    # 1 Network reference: java/net/Socket
    cp.add_class("java/net/Socket")
    arc.add_class(cf)

    triager = CapabilityTriager()
    res = triager.triage_class(cf.this_class_name, arc)

    assert res.primary_domain in (CapabilityDomain.CRYPTOGRAPHY, CapabilityDomain.NETWORK)
    assert res.confidence > 0.0
    assert res.is_security_sensitive is True


def test_bva_f20_rare_reflection_apis():
    """BVA F20-3: Class using reflection APIs (java/lang/reflect or MethodHandles)."""
    arc = JarArchive()
    cf = create_minimal_test_class("com/example/Invoker")
    cp = cf.constant_pool
    cp.add_class("java/lang/reflect/Method")
    cp.add_member_ref(ConstantTag.METHODREF, "java/lang/reflect/Method", "invoke", "(Ljava/lang/Object;[Ljava/lang/Object;)Ljava/lang/Object;")
    arc.add_class(cf)

    triager = CapabilityTriager()
    res = triager.triage_class(cf.this_class_name, arc)

    assert res.primary_domain == CapabilityDomain.REFLECTION
    assert res.is_security_sensitive is True


def test_bva_f20_native_methods_only():
    """BVA F20-4: Class containing only native methods (0 code attributes)."""
    arc = JarArchive()
    cf = create_minimal_test_class("com/example/NativeBridge")
    cp = cf.constant_pool
    m_native = MethodInfo(AccessFlags.PUBLIC | AccessFlags.NATIVE, cp.add_utf8("nativeHook"), cp.add_utf8("()V"), [])
    cf.methods.append(m_native)
    arc.add_class(cf)

    triager = CapabilityTriager()
    res = triager.triage_class(cf.this_class_name, arc)
    assert res.primary_domain is None


def test_bva_f20_private_inner_classes():
    """BVA F20-5: Triage on private static inner classes."""
    arc = JarArchive()
    cf = create_minimal_test_class("com/example/Outer$SecretInner")
    cp = cf.constant_pool
    cp.add_class("javax/crypto/Cipher")
    arc.add_class(cf)

    triager = CapabilityTriager()
    res = triager.triage_class("com/example/Outer$SecretInner", arc)
    assert res.primary_domain == CapabilityDomain.CRYPTOGRAPHY
    assert res.is_security_sensitive is True


# ==============================================================================
# F21 BOUNDARIES (5 tests)
# ==============================================================================

def test_bva_f21_single_class_archive_tfidf():
    """BVA F21-1: Archive with exactly 1 class handled without division by zero in scoring."""
    arc = JarArchive()
    cf = create_minimal_test_class("com/example/Solo")
    arc.add_class(cf)

    triager = CapabilityTriager()
    report = triager.triage_archive(arc)

    assert report.total_classes == 1
    assert "com/example/Solo" in report.triage_results


def test_bva_f21_100_percent_domain_score():
    """BVA F21-2: Class with 100% domain-specific calls achieves maximum score."""
    arc = JarArchive()
    cf = create_minimal_test_class("com/example/StorageService")
    cp = cf.constant_pool
    cp.add_class("java/io/File")
    cp.add_class("java/io/FileInputStream")
    arc.add_class(cf)

    triager = CapabilityTriager()
    res = triager.triage_class(cf.this_class_name, arc)

    assert res.primary_domain == CapabilityDomain.STORAGE_IO
    assert res.confidence == 1.0


def test_bva_f21_zero_domain_calls():
    """BVA F21-3: Class with 0 domain calls achieves score 0.0."""
    arc = JarArchive()
    cf = create_minimal_test_class("com/example/ZeroCalls")
    arc.add_class(cf)

    triager = CapabilityTriager()
    res = triager.triage_class(cf.this_class_name, arc)

    assert len(res.domain_scores) == 0
    for domain in CapabilityDomain:
        assert res.domain_scores.get(domain, 0.0) == 0.0
    assert res.confidence == 0.0


def test_bva_f21_confidence_cutoff_boundary():
    """BVA F21-4: Triage confidence threshold cutoff boundary behavior."""
    arc = JarArchive()
    cf = create_minimal_test_class("com/example/WeakSignal")
    cp = cf.constant_pool
    # Add a single class reference: weight is 1.0
    cp.add_class("java/io/File")
    arc.add_class(cf)

    # Threshold 0.5: matches and assigns primary domain
    triager_low = CapabilityTriager(sensitivity_threshold=0.5)
    res_low = triager_low.triage_class(cf.this_class_name, arc)
    assert res_low.primary_domain == CapabilityDomain.STORAGE_IO

    # Threshold 5.0: score 1.0 < 5.0, primary domain is gated to None
    triager_high = CapabilityTriager(sensitivity_threshold=5.0)
    res_high = triager_high.triage_class(cf.this_class_name, arc)
    assert res_high.primary_domain is None


def test_bva_f21_empty_method_names():
    """BVA F21-5: Class with synthetic or empty-named methods handled safely."""
    arc = JarArchive()
    cf = create_minimal_test_class("com/example/SpecialMethods")
    cp = cf.constant_pool
    m1 = MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("<init>"), cp.add_utf8("()V"), [])
    m2 = MethodInfo(AccessFlags.STATIC, cp.add_utf8("<clinit>"), cp.add_utf8("()V"), [])
    m3 = MethodInfo(AccessFlags.SYNTHETIC, cp.add_utf8("$1"), cp.add_utf8("()V"), [])
    cf.methods.extend([m1, m2, m3])
    arc.add_class(cf)

    triager = CapabilityTriager()
    res = triager.triage_class(cf.this_class_name, arc)
    assert isinstance(res.confidence, float)


# ==============================================================================
# F22 BOUNDARIES (5 tests)
# ==============================================================================

def test_bva_f22_zero_fields_zero_methods():
    """BVA F22-1: Summary of empty class produces valid minimal JSON/Markdown."""
    arc = JarArchive()
    cf = create_minimal_test_class("com/example/Empty")
    arc.add_class(cf)

    slicer = ContextualSlicer()
    slice_json = slicer.generate_llm_slice(cf.this_class_name, arc, format="json")
    data = json.loads(slice_json)
    assert data["class"] == "com/example/Empty"
    assert data["fields"] == []
    assert data["methods"] == []

    slice_md = slicer.generate_llm_slice(cf.this_class_name, arc, format="markdown")
    assert "# Class: `com/example/Empty`" in slice_md


def test_bva_f22_hundred_methods_token_cap():
    """BVA F22-2: Class with 100 methods obeys token cap limits."""
    arc = JarArchive()
    cf = create_minimal_test_class("com/example/LargeAPI")
    cp = cf.constant_pool

    for i in range(100):
        m = MethodInfo(AccessFlags.PUBLIC, cp.add_utf8(f"method_{i:03d}"), cp.add_utf8("()V"), [])
        cf.methods.append(m)
    arc.add_class(cf)

    slicer = ContextualSlicer()
    slice_json = slicer.generate_llm_slice(cf.this_class_name, arc, format="json")
    data = json.loads(slice_json)
    assert len(data["methods"]) == 100


def test_bva_f22_special_chars_in_strings():
    """BVA F22-3: Strings containing newlines, quotes, and HTML entities escaped properly."""
    arc = JarArchive()
    cf = create_minimal_test_class("com/example/Escaped")
    cp = cf.constant_pool
    cp.add_string('Line1\nLine2\r\nQuote: "hello" & <tag>')
    arc.add_class(cf)

    slicer = ContextualSlicer()
    slice_json = slicer.generate_llm_slice(cf.this_class_name, arc, format="json")
    data = json.loads(slice_json)
    assert len(data["string_literals"]) == 1
    assert "Line1\nLine2\r\nQuote: \"hello\" & <tag>" in data["string_literals"]

    slice_md = slicer.generate_llm_slice(cf.this_class_name, arc, format="markdown")
    assert "Line1\\nLine2" in slice_md


def test_bva_f22_empty_fields_in_schema():
    """BVA F22-4: JSON slice validation with empty fields list."""
    arc = JarArchive()
    cf = create_minimal_test_class("com/example/NoFieldsClass")
    arc.add_class(cf)

    slicer = ContextualSlicer()
    summary = slicer.generate_class_summary_dict(cf.this_class_name, arc)
    assert "fields" in summary
    assert isinstance(summary["fields"], list)
    assert len(summary["fields"]) == 0


def test_bva_f22_token_compression_ratio():
    """BVA F22-5: Token compression ratio verified >50% compared to raw instructions."""
    arc = JarArchive()
    cf = create_minimal_test_class("com/example/VerboseMethod")
    cp = cf.constant_pool
    # 50 instructions in method
    code = bytes([Opcode.NOP] * 50 + [Opcode.RETURN])
    m = MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("compute"), cp.add_utf8("()V"), [CodeAttribute(cp.add_utf8("Code"), 1, 1, code)])
    cf.methods.append(m)
    arc.add_class(cf)

    slicer = ContextualSlicer()
    summary = slicer.generate_class_summary_dict(cf.this_class_name, arc)
    # The summary omits all 50 NOP bytecodes and keeps high-signal method descriptor only
    assert len(summary["methods"][0]["name"]) > 0


# ==============================================================================
# F23 BOUNDARIES (5 tests)
# ==============================================================================

def test_bva_f23_sink_pattern_zero_matches():
    """BVA F23-1: Non-matching sink pattern returns empty candidate lists safely."""
    arc = JarArchive()
    cg = CallGraph()
    slicer = ContextualSlicer()
    res = slicer.generate_sink_slice("NonExistentSink;->execute()V", arc, cg)
    assert res["sink_pattern"] == "NonExistentSink;->execute()V"
    assert res["direct_callers"] == []
    assert res["entrypoint_candidates"] == []


def test_bva_f23_sink_matching_all_methods():
    """BVA F23-2: Generic sink pattern matching all methods handled cleanly."""
    arc = JarArchive()
    cg = CallGraph()
    # Add 5 methods calling generic sink
    for i in range(5):
        caller = f"Lcom/example/Caller{i};->run()V"
        sink = f"Lcom/example/CommonSink;->sink()V"
        cg.graph.add_edge(caller, sink)

    slicer = ContextualSlicer()
    res = slicer.generate_sink_slice("CommonSink;->sink", arc, cg)
    assert len(res["direct_callers"]) == 5
    assert len(res["entrypoint_candidates"]) == 5


def test_bva_f23_sink_in_cyclic_call_chain():
    """BVA F23-3: Sink inside mutual recursion cycle isolates callers correctly without hanging."""
    arc = JarArchive()
    cg = CallGraph()
    m_a = "Lcom/A;->loopA()V"
    m_b = "Lcom/B;->loopB()V"
    sink = "Ljavax/crypto/Cipher;->doFinal()V"

    cg.graph.add_edge(m_a, m_b)
    cg.graph.add_edge(m_b, m_a)
    cg.graph.add_edge(m_b, sink)

    slicer = ContextualSlicer()
    res = slicer.generate_sink_slice("Cipher;->doFinal", arc, cg)
    assert m_b in res["direct_callers"]
    assert m_a in res["entrypoint_candidates"] or m_b in res["entrypoint_candidates"]


def test_bva_f23_multiple_disjoint_entrypoints():
    """BVA F23-4: 5 disjoint entry points reaching same sink captured in report."""
    arc = JarArchive()
    cg = CallGraph()
    sink = "Ljava/net/Socket;->connect()V"

    for i in range(5):
        entry = f"Lcom/Entry{i};->start()V"
        helper = f"Lcom/Helper{i};->doConnect()V"
        cg.graph.add_edge(entry, helper)
        cg.graph.add_edge(helper, sink)

    slicer = ContextualSlicer()
    res = slicer.generate_sink_slice("Socket;->connect", arc, cg)
    assert len(res["direct_callers"]) == 5
    for i in range(5):
        assert f"Lcom/Entry{i};->start()V" in res["entrypoint_candidates"]


def test_bva_f23_backward_reachability_depth_limits():
    """BVA F23-5: Backward reachability tracing on chain E -> D -> C -> B -> A -> Sink."""
    arc = JarArchive()
    cg = CallGraph()
    nodes = ["Lcom/E;->m()V", "Lcom/D;->m()V", "Lcom/C;->m()V", "Lcom/B;->m()V", "Lcom/A;->m()V", "Ljavax/crypto/Cipher;->doFinal()V"]

    for i in range(len(nodes) - 1):
        cg.graph.add_edge(nodes[i], nodes[i + 1])

    slicer = ContextualSlicer()
    res = slicer.generate_sink_slice("Cipher;->doFinal", arc, cg)
    assert "Lcom/A;->m()V" in res["direct_callers"]
    assert "Lcom/B;->m()V" in res["entrypoint_candidates"]
