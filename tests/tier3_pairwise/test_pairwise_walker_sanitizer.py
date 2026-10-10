"""
tests/tier3_pairwise/test_pairwise_walker_sanitizer.py: Pairwise interactions between M1 Walker and M2 Sanitizer.
Covers:
- F01 (ClassFile Parser/Serializer) + F08 (StackMapTable Normalization)
- F03 (Inheritance Hierarchy DAG) + F08 (StackMapTable Normalization)
- F03 (Inheritance Hierarchy DAG) + F09 (Generic Signature Sanitizer)
- F04 (Polymorphic Call Graph) + F10 (Irregular Exception Handler Fixer)
- F06 (Field Access Mapping) + F11 (Duplicate Synthetic Member Deduplication)
- F07 (Inter-Class Dependency Slicing) + F10 (Irregular Exception Handler Fixer)
"""

import struct
from pathlib import Path
import pytest

from jar_engine.model.constants import Opcode, AccessFlags, ConstantTag
from jar_engine.model.class_file import ClassFile
from jar_engine.model.cp_info import ConstantPool
from jar_engine.model.member_info import (
    FieldInfo, MethodInfo, CodeAttribute, SignatureAttribute,
    StackMapTableAttribute, ExceptionTableEntry
)
from jar_engine.model.jar_archive import JarArchive
from jar_engine.walker.hierarchy import InheritanceHierarchy
from jar_engine.walker.call_graph import CallGraph
from jar_engine.walker.dependency_slice import ClassDependencyGraph, DependencySlicer
from jar_engine.sanitizer import (
    BytecodeSanitizer,
    sanitize_class_signatures,
    sanitize_class_exceptions,
    sanitize_class_stack_maps,
    deduplicate_fields,
    verify_sanitized_class_structural_integrity,
)


def test_pairwise_f01_f08_parser_sanitizer_stackmap(corrupted_structures_jar: Path):
    """
    Pairwise F01 + F08: ClassFile parser parses non-standard/corrupt StackMapTable,
    sanitizer repairs or strips frames, and serializer produces byte-exact verified output.
    """
    archive = JarArchive.from_file(corrupted_structures_jar)
    assert "com/synthetic/corrupted/CorruptStackMaps" in archive.classes

    cf = archive.classes["com/synthetic/corrupted/CorruptStackMaps"]
    raw_before = cf.to_bytes()
    cf_parsed = ClassFile.parse(raw_before)

    # Sanitize stack maps
    modified = sanitize_class_stack_maps(cf_parsed)

    # Verify serializer produces valid bytes that roundtrip cleanly
    raw_after = cf_parsed.to_bytes()
    assert len(raw_after) > 0
    cf_reparsed = ClassFile.parse(raw_after)
    assert cf_reparsed.magic == 0xCAFEBABE
    assert cf_reparsed.major_version == cf.major_version
    assert cf_reparsed.to_bytes() == raw_after


def test_pairwise_f03_f08_hierarchy_lca_stackmap():
    """
    Pairwise F03 + F08: Inheritance hierarchy DAG LCA resolution directly
    informs StackMapTable object frame type verification and merging.
    """
    archive = JarArchive()

    # Base class Shape
    cp_s = ConstantPool()
    cf_shape = ClassFile(
        magic=0xCAFEBABE, minor_version=0, major_version=52, constant_pool=cp_s,
        access_flags=AccessFlags.PUBLIC, this_class=cp_s.add_class("com/example/Shape"),
        super_class=cp_s.add_class("java/lang/Object"), interfaces=[],
        fields=[], methods=[], attributes=[]
    )
    archive.add_class(cf_shape)

    # Subclass Circle extends Shape
    cp_c = ConstantPool()
    cf_circle = ClassFile(
        magic=0xCAFEBABE, minor_version=0, major_version=52, constant_pool=cp_c,
        access_flags=AccessFlags.PUBLIC, this_class=cp_c.add_class("com/example/Circle"),
        super_class=cp_c.add_class("com/example/Shape"), interfaces=[],
        fields=[], methods=[], attributes=[]
    )
    archive.add_class(cf_circle)

    # Subclass Square extends Shape
    cp_sq = ConstantPool()
    cf_square = ClassFile(
        magic=0xCAFEBABE, minor_version=0, major_version=52, constant_pool=cp_sq,
        access_flags=AccessFlags.PUBLIC, this_class=cp_sq.add_class("com/example/Square"),
        super_class=cp_sq.add_class("com/example/Shape"), interfaces=[],
        fields=[], methods=[], attributes=[]
    )
    archive.add_class(cf_square)

    hierarchy = InheritanceHierarchy.from_archive(archive)
    lca = hierarchy.find_lowest_common_ancestor("com/example/Circle", "com/example/Square")
    assert lca == "com/example/Shape"

    # Add branching method with StackMapTable to Circle
    c_name = cp_c.add_utf8("Code")
    sm_name = cp_c.add_utf8("StackMapTable")
    code = bytes([Opcode.ILOAD_0, Opcode.IFEQ, 0, 7, Opcode.ICONST_1, Opcode.IRETURN, Opcode.ICONST_0, Opcode.IRETURN])
    sm_attr = StackMapTableAttribute(sm_name, b"")
    cf_circle.methods.append(MethodInfo(
        AccessFlags.PUBLIC | AccessFlags.STATIC, cp_c.add_utf8("branchTest"), cp_c.add_utf8("(I)I"),
        [CodeAttribute(c_name, 2, 1, code, attributes=[sm_attr])]
    ))

    # Sanitize stack maps passing hierarchy
    sanitizer = BytecodeSanitizer()
    report = sanitizer.sanitize_class(cf_circle, hierarchy=hierarchy)
    assert report.class_name == "com/example/Circle"


def test_pairwise_f03_f09_hierarchy_generic_signatures():
    """
    Pairwise F03 + F09: Multi-class inheritance hierarchy DAG parsed where
    classes possess sanitized generic signatures.
    """
    archive = JarArchive()

    # Generic Base Interface: IRepository<T> with malformed signature
    cp_repo = ConstantPool()
    cf_repo = ClassFile(
        magic=0xCAFEBABE, minor_version=0, major_version=52, constant_pool=cp_repo,
        access_flags=AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT,
        this_class=cp_repo.add_class("com/example/IRepository"),
        super_class=cp_repo.add_class("java/lang/Object"), interfaces=[],
        fields=[], methods=[],
        attributes=[SignatureAttribute(cp_repo.add_utf8("Signature"), cp_repo.add_utf8("<T:unclosed"))]
    )
    archive.add_class(cf_repo)

    # Implementing Class: UserRepo implements IRepository
    cp_user = ConstantPool()
    cf_user = ClassFile(
        magic=0xCAFEBABE, minor_version=0, major_version=52, constant_pool=cp_user,
        access_flags=AccessFlags.PUBLIC, this_class=cp_user.add_class("com/example/UserRepo"),
        super_class=cp_user.add_class("java/lang/Object"),
        interfaces=[cp_user.add_class("com/example/IRepository")],
        fields=[], methods=[],
        attributes=[SignatureAttribute(cp_user.add_utf8("Signature"), cp_user.add_utf8("Ljava/lang/Object;Lcom/example/IRepository<Ljava/lang/String;>;"))]
    )
    archive.add_class(cf_user)

    # Sanitize signatures across archive
    sanitizer = BytecodeSanitizer()
    rep = sanitizer.sanitize_archive(archive)
    assert rep.total_signatures_stripped >= 1

    # Verify hierarchy DAG navigation still cleanly discovers relationship
    hierarchy = InheritanceHierarchy.from_archive(archive)
    implementors = hierarchy.get_all_implementors("com/example/IRepository")
    assert "com/example/UserRepo" in implementors
    assert "com/example/IRepository" in hierarchy.get_interfaces("com/example/UserRepo")


def test_pairwise_f04_f10_callgraph_exception_handlers():
    """
    Pairwise F04 + F10: Polymorphic call graph traversal traversing methods
    whose exception handler ranges were normalized.
    """
    archive = JarArchive()

    # Target Callee Class
    cp_tgt = ConstantPool()
    cf_tgt = ClassFile(
        magic=0xCAFEBABE, minor_version=0, major_version=52, constant_pool=cp_tgt,
        access_flags=AccessFlags.PUBLIC, this_class=cp_tgt.add_class("com/example/TargetService"),
        super_class=cp_tgt.add_class("java/lang/Object"), interfaces=[],
        fields=[],
        methods=[MethodInfo(
            AccessFlags.PUBLIC | AccessFlags.STATIC, cp_tgt.add_utf8("executeTask"), cp_tgt.add_utf8("()V"),
            [CodeAttribute(cp_tgt.add_utf8("Code"), 1, 0, bytes([Opcode.RETURN]))]
        )],
        attributes=[]
    )
    archive.add_class(cf_tgt)

    # Caller Class with corrupt exception handler around call site
    cp_clr = ConstantPool()
    tgt_m_ref = cp_clr.add_member_ref(ConstantTag.METHODREF, "com/example/TargetService", "executeTask", "()V")
    caller_code = bytes([Opcode.INVOKESTATIC]) + struct.pack(">H", tgt_m_ref) + bytes([Opcode.RETURN])

    # Corrupt exception entries: inverted range and self-catching loop
    bad_handlers = [
        ExceptionTableEntry(start_pc=10, end_pc=2, handler_pc=0, catch_type=0),
        ExceptionTableEntry(start_pc=0, end_pc=5, handler_pc=2, catch_type=0),
    ]
    cf_clr = ClassFile(
        magic=0xCAFEBABE, minor_version=0, major_version=52, constant_pool=cp_clr,
        access_flags=AccessFlags.PUBLIC, this_class=cp_clr.add_class("com/example/CallerService"),
        super_class=cp_clr.add_class("java/lang/Object"), interfaces=[],
        fields=[],
        methods=[MethodInfo(
            AccessFlags.PUBLIC | AccessFlags.STATIC, cp_clr.add_utf8("run"), cp_clr.add_utf8("()V"),
            [CodeAttribute(cp_clr.add_utf8("Code"), 2, 0, caller_code, exception_table=bad_handlers)]
        )],
        attributes=[]
    )
    archive.add_class(cf_clr)

    # Sanitize exceptions
    fixed_count = sanitize_class_exceptions(cf_clr)
    assert fixed_count >= 1

    # Verify CallGraph traversal using canonical method ID
    cg = CallGraph.build(archive)
    callees = cg.get_callees("Lcom/example/CallerService;->run()V")
    assert any("TargetService" in c and "executeTask" in c for c in callees)
    callers = cg.get_callers("Lcom/example/TargetService;->executeTask()V")
    assert any("CallerService" in c and "run" in c for c in callers)


def test_pairwise_f06_f11_field_access_synthetic_dedup():
    """
    Pairwise F06 + F11: Field access mapping constructed after duplicate
    synthetic member deduplication.
    """
    archive = JarArchive()
    cp = ConstantPool()

    # Add duplicate synthetic field
    fld_real = FieldInfo(AccessFlags.PUBLIC, cp.add_utf8("counter"), cp.add_utf8("I"), [])
    fld_synth = FieldInfo(AccessFlags.PUBLIC | AccessFlags.SYNTHETIC, cp.add_utf8("counter"), cp.add_utf8("Ljava/lang/String;"), [])

    # Methods accessing field
    fld_ref = cp.add_member_ref(ConstantTag.FIELDREF, "com/example/DataHolder", "counter", "I")
    read_code = bytes([Opcode.ALOAD_0, Opcode.GETFIELD]) + struct.pack(">H", fld_ref) + bytes([Opcode.IRETURN])
    write_code = bytes([Opcode.ALOAD_0, Opcode.ILOAD_1, Opcode.PUTFIELD]) + struct.pack(">H", fld_ref) + bytes([Opcode.RETURN])

    cf = ClassFile(
        magic=0xCAFEBABE, minor_version=0, major_version=52, constant_pool=cp,
        access_flags=AccessFlags.PUBLIC, this_class=cp.add_class("com/example/DataHolder"),
        super_class=cp.add_class("java/lang/Object"), interfaces=[],
        fields=[fld_real, fld_synth],
        methods=[
            MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("getCounter"), cp.add_utf8("()I"), [CodeAttribute(cp.add_utf8("Code"), 2, 1, read_code)]),
            MethodInfo(AccessFlags.PUBLIC, cp.add_utf8("setCounter"), cp.add_utf8("(I)V"), [CodeAttribute(cp.add_utf8("Code"), 2, 2, write_code)])
        ],
        attributes=[]
    )
    archive.add_class(cf)

    # Deduplicate fields: renames conflicting duplicate synthetic field
    renamed = deduplicate_fields(cf)
    assert renamed == 1
    assert len(cf.fields) == 2
    assert cf.fields[0].name(cp) == "counter"
    assert "$dup$" in cf.fields[1].name(cp)

    # Build CallGraph and inspect field access mapping
    cg = CallGraph.build(archive)
    assert len(cg.field_accesses) >= 2
    fld_id = "Lcom/example/DataHolder;->counter:I"
    assert any(acc.field == fld_id and acc.access_type == "read" for acc in cg.field_accesses)
    assert any(acc.field == fld_id and acc.access_type == "write" for acc in cg.field_accesses)


def test_pairwise_f07_f10_dependency_slicing_exception_fixes():
    """
    Pairwise F07 + F10: Inter-class dependency slicing computed on methods
    where irregular/corrupted exception handler tables were normalized.
    """
    archive = JarArchive()

    # Exception Class
    cp_ex = ConstantPool()
    cf_ex = ClassFile(
        magic=0xCAFEBABE, minor_version=0, major_version=52, constant_pool=cp_ex,
        access_flags=AccessFlags.PUBLIC, this_class=cp_ex.add_class("com/example/CustomException"),
        super_class=cp_ex.add_class("java/lang/Exception"), interfaces=[],
        fields=[], methods=[], attributes=[]
    )
    archive.add_class(cf_ex)

    # Core Service that catches CustomException with invalid bounds
    cp_srv = ConstantPool()
    ex_cp_idx = cp_srv.add_class("com/example/CustomException")
    code = bytes([Opcode.NOP] * 20 + [Opcode.RETURN])

    # Range out of bounds and self-catching
    ex_entries = [
        ExceptionTableEntry(start_pc=0, end_pc=100, handler_pc=10, catch_type=ex_cp_idx),  # OOB
        ExceptionTableEntry(start_pc=0, end_pc=15, handler_pc=18, catch_type=ex_cp_idx),   # Valid
    ]
    cf_srv = ClassFile(
        magic=0xCAFEBABE, minor_version=0, major_version=52, constant_pool=cp_srv,
        access_flags=AccessFlags.PUBLIC, this_class=cp_srv.add_class("com/example/CoreService"),
        super_class=cp_srv.add_class("java/lang/Object"), interfaces=[],
        fields=[],
        methods=[MethodInfo(
            AccessFlags.PUBLIC | AccessFlags.STATIC, cp_srv.add_utf8("safeRun"), cp_srv.add_utf8("()V"),
            [CodeAttribute(cp_srv.add_utf8("Code"), 1, 0, code, exception_table=ex_entries)]
        )],
        attributes=[]
    )
    archive.add_class(cf_srv)

    # Sanitize archive
    sanitizer = BytecodeSanitizer()
    report = sanitizer.sanitize_archive(archive)
    assert report.total_exceptions_fixed >= 1

    # Verify dependency slicing
    forward_slice = DependencySlicer.slice(archive, "com/example/CoreService", direction="forward")
    assert "com/example/CoreService" in forward_slice
    assert "com/example/CustomException" in forward_slice
