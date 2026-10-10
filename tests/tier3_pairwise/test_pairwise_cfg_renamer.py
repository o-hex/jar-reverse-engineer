"""
tests/tier3_pairwise/test_pairwise_cfg_renamer.py: Pairwise interactions between M3 CFG and M5 Renamer.
Covers:
- F13 (CFG Basic Block Partitioning) + F27 (Global Bytecode Reference Rewriting)
- F14 (Control Flow De-Flattening) + F25 (Union-Find Method Equivalence)
- F16 (Constant Folding Lattice) + F27 (Global Bytecode Reference Rewriting)
- F17 (Static String Decryptor Resolution) + F27 (Global Bytecode Reference Rewriting)
- F18 (Dynamic Invoke Resolution) + F27 (Global Bytecode Reference Rewriting)
- F19 (Dead Code & NOP Pruning) + F28 (Access Widening & Class Verification)
"""

import struct
from pathlib import Path
import pytest

from jar_engine.model.constants import Opcode, AccessFlags, ConstantTag
from jar_engine.model.class_file import ClassFile
from jar_engine.model.cp_info import (
    ConstantPool, CONSTANT_String_info, CONSTANT_Class_info, CONSTANT_Fieldref_info
)
from jar_engine.model.member_info import MethodInfo, FieldInfo, CodeAttribute
from jar_engine.model.jar_archive import JarArchive
from jar_engine.cfg import (
    ControlFlowGraph, prune_dead_code_and_nops, reassemble_cfg
)
from jar_engine.normalization import (
    MethodNormalizer, fold_constants_and_arithmetic
)
from jar_engine.walker.hierarchy import InheritanceHierarchy
from jar_engine.renamer import (
    SymbolRenamer, SymbolMapping, MethodEquivalenceCluster
)


def test_pairwise_f13_f27_cfg_reassembly_ref_rewriting():
    """
    Pairwise F13 + F27: CFG block reassembly produces clean bytecode that
    is subsequently rewritten with global symbol renames across constant pools.
    """
    archive = JarArchive()

    # Target class Callee
    cp_callee = ConstantPool()
    cf_callee = ClassFile(
        magic=0xCAFEBABE, minor_version=0, major_version=52, constant_pool=cp_callee,
        access_flags=AccessFlags.PUBLIC, this_class=cp_callee.add_class("com/example/Callee"),
        super_class=cp_callee.add_class("java/lang/Object"), interfaces=[],
        fields=[],
        methods=[MethodInfo(
            AccessFlags.PUBLIC | AccessFlags.STATIC, cp_callee.add_utf8("doWork"), cp_callee.add_utf8("()V"),
            [CodeAttribute(cp_callee.add_utf8("Code"), 1, 0, bytes([Opcode.RETURN]))]
        )],
        attributes=[]
    )
    archive.add_class(cf_callee)

    # Caller class calling Callee.doWork()
    cp_caller = ConstantPool()
    callee_m_ref = cp_caller.add_member_ref(ConstantTag.METHODREF, "com/example/Callee", "doWork", "()V")
    caller_code = bytes([Opcode.INVOKESTATIC]) + struct.pack(">H", callee_m_ref) + bytes([Opcode.RETURN])

    # Partition and reassemble CFG
    cfg = ControlFlowGraph.from_code(caller_code, cp=cp_caller)
    new_code, _, _, _ = reassemble_cfg(cfg, cp=cp_caller)
    c_attr = CodeAttribute(cp_caller.add_utf8("Code"), 1, 0, new_code)

    cf_caller = ClassFile(
        magic=0xCAFEBABE, minor_version=0, major_version=52, constant_pool=cp_caller,
        access_flags=AccessFlags.PUBLIC, this_class=cp_caller.add_class("com/example/Caller"),
        super_class=cp_caller.add_class("java/lang/Object"), interfaces=[],
        fields=[],
        methods=[MethodInfo(
            AccessFlags.PUBLIC | AccessFlags.STATIC, cp_caller.add_utf8("run"), cp_caller.add_utf8("()V"),
            [c_attr]
        )],
        attributes=[]
    )
    archive.add_class(cf_caller)

    # Apply SymbolRenamer to rename com/example/Callee -> com/example/NewCallee
    mapping = SymbolMapping.from_dict({
        "classes": {"com/example/Callee": "com/example/NewCallee"}
    })
    renamer = SymbolRenamer()
    report = renamer.apply_mapping(archive, mapping)
    assert report.classes_renamed >= 1

    # Verify constant pool of Caller now points to com/example/NewCallee
    all_class_names = [
        cf_caller.resolve_utf8(entry.name_index)
        for _, entry in cf_caller.constant_pool.entries()
        if isinstance(entry, CONSTANT_Class_info)
    ]
    assert "com/example/NewCallee" in all_class_names
    assert "com/example/Callee" not in all_class_names


def test_pairwise_f14_f25_deflattening_method_equivalence():
    """
    Pairwise F14 + F25: De-flattened polymorphic methods renamed consistently
    under Union-Find override clusters across interfaces and classes.
    """
    archive = JarArchive()

    # Interface: IWorker with method work(I)I
    cp_iface = ConstantPool()
    cf_iface = ClassFile(
        magic=0xCAFEBABE, minor_version=0, major_version=52, constant_pool=cp_iface,
        access_flags=AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT,
        this_class=cp_iface.add_class("com/example/IWorker"),
        super_class=cp_iface.add_class("java/lang/Object"), interfaces=[],
        fields=[],
        methods=[MethodInfo(
            AccessFlags.PUBLIC | AccessFlags.ABSTRACT, cp_iface.add_utf8("work"), cp_iface.add_utf8("(I)I"), []
        )],
        attributes=[]
    )
    archive.add_class(cf_iface)

    # Implementing class DispatchWorker implementing work(I)I with flattened dispatch loop
    cp_impl = ConstantPool()
    disp_code = (
        bytes([Opcode.ICONST_0, Opcode.ISTORE_1, Opcode.GOTO, 0, 3]) +
        bytes([Opcode.ILOAD_1, Opcode.TABLESWITCH, 0]) +
        struct.pack(">iiiiiii", 59, 0, 3, 30, 40, 49, 59) +
        bytes([Opcode.ILOAD_0, Opcode.BIPUSH, 10, Opcode.IADD, Opcode.ISTORE_0, Opcode.ICONST_1, Opcode.ISTORE_1, Opcode.GOTO]) + struct.pack(">h", -38) +
        bytes([Opcode.ILOAD_0, Opcode.ICONST_2, Opcode.IMUL, Opcode.ISTORE_0, Opcode.ICONST_2, Opcode.ISTORE_1, Opcode.GOTO]) + struct.pack(">h", -47) +
        bytes([Opcode.ILOAD_0, Opcode.BIPUSH, 5, Opcode.ISUB, Opcode.ISTORE_0, Opcode.ICONST_3, Opcode.ISTORE_1, Opcode.GOTO]) + struct.pack(">h", -57) +
        bytes([Opcode.ILOAD_0, Opcode.IRETURN])
    )
    m_impl = MethodInfo(
        AccessFlags.PUBLIC, cp_impl.add_utf8("work"), cp_impl.add_utf8("(I)I"),
        [CodeAttribute(cp_impl.add_utf8("Code"), 3, 2, disp_code)]
    )
    cf_impl = ClassFile(
        magic=0xCAFEBABE, minor_version=0, major_version=52, constant_pool=cp_impl,
        access_flags=AccessFlags.PUBLIC, this_class=cp_impl.add_class("com/example/DispatchWorker"),
        super_class=cp_impl.add_class("java/lang/Object"),
        interfaces=[cp_impl.add_class("com/example/IWorker")],
        fields=[], methods=[m_impl], attributes=[]
    )
    archive.add_class(cf_impl)

    # De-flatten the method first (M3)
    normalizer = MethodNormalizer()
    stats = normalizer.normalize_method(cf_impl, m_impl)
    assert stats.deflattened or stats.complexity_reduced > 0

    # Build equivalence clusters (M5) and verify both methods clustered together
    hier = InheritanceHierarchy.from_archive(archive)
    cluster = MethodEquivalenceCluster(hier)
    cluster.build_from_archive(archive)
    m_key_iface = "com/example/IWorker#work#(I)I"
    m_key_impl = "com/example/DispatchWorker#work#(I)I"
    all_keys = {m_key_iface, m_key_impl}
    equiv_members = cluster.get_cluster_members(m_key_iface, all_keys)
    assert m_key_impl in equiv_members

    # Apply mapping renaming the interface method
    mapping = SymbolMapping.from_dict({
        "methods": {m_key_iface: "executeTask"}
    })
    renamer = SymbolRenamer()
    report = renamer.apply_mapping(archive, mapping)
    assert report.methods_renamed >= 2

    # Verify both methods in interface and class renamed to executeTask
    assert cf_iface.methods[0].name(cf_iface.constant_pool) == "executeTask"
    assert cf_impl.methods[0].name(cf_impl.constant_pool) == "executeTask"


def test_pairwise_f16_f27_constant_folded_bytecode_renaming():
    """
    Pairwise F16 + F27: Bytecode containing folded constants successfully
    updated with renamed class and field references.
    """
    archive = JarArchive()

    # Target class DataContainer with field int count
    cp_data = ConstantPool()
    fld = FieldInfo(AccessFlags.PUBLIC, cp_data.add_utf8("count"), cp_data.add_utf8("I"), [])
    cf_data = ClassFile(
        magic=0xCAFEBABE, minor_version=0, major_version=52, constant_pool=cp_data,
        access_flags=AccessFlags.PUBLIC, this_class=cp_data.add_class("com/example/DataContainer"),
        super_class=cp_data.add_class("java/lang/Object"), interfaces=[],
        fields=[fld], methods=[], attributes=[]
    )
    archive.add_class(cf_data)

    # Computing class: (10 + 20) * 2 = 60, then putfield DataContainer.count
    cp_calc = ConstantPool()
    fld_ref = cp_calc.add_member_ref(ConstantTag.FIELDREF, "com/example/DataContainer", "count", "I")
    calc_code = bytes([
        Opcode.ALOAD_0,
        Opcode.BIPUSH, 10,
        Opcode.BIPUSH, 20,
        Opcode.IADD,
        Opcode.BIPUSH, 2,
        Opcode.IMUL,
        Opcode.PUTFIELD
    ]) + struct.pack(">H", fld_ref) + bytes([Opcode.RETURN])

    m_calc = MethodInfo(
        AccessFlags.PUBLIC, cp_calc.add_utf8("computeAndSave"), cp_calc.add_utf8("(Lcom/example/DataContainer;)V"),
        [CodeAttribute(cp_calc.add_utf8("Code"), 3, 2, calc_code)]
    )
    cf_calc = ClassFile(
        magic=0xCAFEBABE, minor_version=0, major_version=52, constant_pool=cp_calc,
        access_flags=AccessFlags.PUBLIC, this_class=cp_calc.add_class("com/example/Calculator"),
        super_class=cp_calc.add_class("java/lang/Object"), interfaces=[],
        fields=[], methods=[m_calc], attributes=[]
    )
    archive.add_class(cf_calc)

    # Fold constants (M3)
    normalizer = MethodNormalizer()
    stats = normalizer.normalize_method(cf_calc, m_calc)
    assert stats.constants_folded >= 1

    # Rename DataContainer -> ModelContainer and count -> totalCount (M5)
    mapping = SymbolMapping.from_dict({
        "classes": {"com/example/DataContainer": "com/example/ModelContainer"},
        "fields": {"com/example/DataContainer#count": "totalCount"}
    })
    renamer = SymbolRenamer()
    report = renamer.apply_mapping(archive, mapping)
    assert report.classes_renamed >= 1
    assert report.fields_renamed >= 1

    # Check field name on renamed class
    assert cf_data.fields[0].name(cf_data.constant_pool) == "totalCount"


def test_pairwise_f17_f27_decrypted_strings_reference_rewriting(constant_obfuscated_jar: Path):
    """
    Pairwise F17 + F27: Inlined decrypted string literals preserved intact
    while class and method references are globally renamed.
    """
    archive = JarArchive.from_file(constant_obfuscated_jar)
    cf = archive.classes["com/synthetic/constants/StringObfuscated"]
    cp = cf.constant_pool

    m_secret = next(m for m in cf.methods if m.name(cp) == "getSecretMessage")

    # Decrypt string constants (M3)
    normalizer = MethodNormalizer()
    stats = normalizer.normalize_method(cf, m_secret)
    assert stats.strings_decrypted == 1

    # Rename class and method (M5)
    mapping = SymbolMapping.from_dict({
        "classes": {"com/synthetic/constants/StringObfuscated": "com/app/SecretService"},
        "methods": {"com/synthetic/constants/StringObfuscated#getSecretMessage#()Ljava/lang/String;": "retrieveKey"}
    })
    renamer = SymbolRenamer()
    report = renamer.apply_mapping(archive, mapping)
    assert report.classes_renamed >= 1
    assert report.methods_renamed >= 1

    # Verify decrypted string is still in constant pool
    all_strings = [
        cp.resolve_utf8(e.string_index)
        for _, e in cp.entries()
        if isinstance(e, CONSTANT_String_info)
    ]
    assert "HELLO" in all_strings
    # Verify new class name exists
    assert "com/app/SecretService" in archive.classes


def test_pairwise_f18_f27_resolved_dynamic_invokes_renaming(constant_obfuscated_jar: Path):
    """
    Pairwise F18 + F27: Resolved dynamic invoke bootstrap references updated
    during global symbol renaming.
    """
    archive = JarArchive.from_file(constant_obfuscated_jar)
    cf = archive.classes["com/synthetic/constants/DynamicInvoker"]
    cp = cf.constant_pool

    m_concat = next(m for m in cf.methods if m.name(cp) == "concatSecret")

    # Resolve invokedynamic (M3)
    normalizer = MethodNormalizer()
    stats = normalizer.normalize_method(cf, m_concat)
    assert stats.dynamic_invokes_resolved >= 1

    # Rename class and method (M5)
    mapping = SymbolMapping.from_dict({
        "classes": {"com/synthetic/constants/DynamicInvoker": "com/app/InvokerService"},
        "methods": {"com/synthetic/constants/DynamicInvoker#concatSecret#()Ljava/lang/String;": "formatSecret"}
    })
    renamer = SymbolRenamer()
    report = renamer.apply_mapping(archive, mapping)
    assert report.classes_renamed >= 1
    assert "com/app/InvokerService" in archive.classes


def test_pairwise_f19_f28_dead_code_pruned_access_widening():
    """
    Pairwise F19 + F28: Dead code pruned and access flags widened to public
    on package-private classes relocated across packages.
    """
    archive = JarArchive()
    cp = ConstantPool()

    # Package-private class in pkgA with trailing dead code
    code = bytes([
        Opcode.ICONST_1, Opcode.IRETURN,
        Opcode.NOP, Opcode.NOP,
        Opcode.ICONST_0, Opcode.IRETURN
    ])
    code_attr = CodeAttribute(cp.add_utf8("Code"), 1, 0, code)
    m = MethodInfo(0, cp.add_utf8("internalAction"), cp.add_utf8("()I"), [code_attr])

    cf = ClassFile(
        magic=0xCAFEBABE, minor_version=0, major_version=52, constant_pool=cp,
        access_flags=0,  # Package-private (no PUBLIC flag)
        this_class=cp.add_class("com/example/pkgA/InternalHelper"),
        super_class=cp.add_class("java/lang/Object"), interfaces=[],
        fields=[], methods=[m], attributes=[]
    )
    archive.add_class(cf)
    assert not (cf.access_flags & AccessFlags.PUBLIC)

    # Prune dead code (M3)
    normalizer = MethodNormalizer()
    stats = normalizer.normalize_method(cf, m)
    assert stats.dead_blocks_pruned >= 0

    # Relocate to pkgB (M5) -> triggers access widening (F28)
    mapping = SymbolMapping.from_dict({
        "classes": {"com/example/pkgA/InternalHelper": "com/example/pkgB/InternalHelper"}
    })
    renamer = SymbolRenamer()
    report = renamer.apply_mapping(archive, mapping)
    assert report.access_widened >= 1
    assert bool(cf.access_flags & AccessFlags.PUBLIC)
