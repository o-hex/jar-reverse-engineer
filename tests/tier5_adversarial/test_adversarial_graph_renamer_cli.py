"""
tests/tier5_adversarial/test_adversarial_graph_renamer_cli.py:
Tier 5 White-Box Adversarial Stress Testing for jar_engine.

Subsystems under adversarial test:
- Walker & Hierarchy (deep diamond hierarchies, circular inheritance handling, LCA arrays/primitives, assignability lattice)
- Call Graph & Dispatch (CHA vs RTA pruning, default methods JVMS §5.4.3.3 maximally-specific resolution, static shadowing, invokespecial resolution, indy fallback)
- Dependency Slicing & Condensation (multi-component disconnected SCCs, Tarjan cycles, topological sort directions, slice isolation)
- Symbol Renamer & Overloads (distinct vs shared UTF-8 overloads, inheritance override propagation, JDK pinning name-matching)
- Access Widener (package boundaries, default package transitions, public preservation)
- Repackager & SPI Synchronization (multi-provider META-INF/services, comments/blanks preservation, manifest main class RFC 822 dot format)
- CLI Frontend (subcommand flag parsing, analyze, normalize, slice, rename edge cases)
- Triage & Contextual Slicing (multi-domain hybrid classification, string literal capping, sink trace isolation)
- Symbol Mapping Validation (boundary regex checks, illegal characters, dollar/underscore identifiers)
"""

from __future__ import annotations
import io
import json
import zipfile
import argparse
from pathlib import Path
from typing import Dict, List, Set, Optional

import pytest
import networkx as nx

from jar_engine.model.constants import (
    JAVA_MAGIC, AccessFlags, ConstantTag, Opcode
)
from jar_engine.model.cp_info import (
    ConstantPool, CONSTANT_Class_info, CONSTANT_Utf8_info,
    CONSTANT_Methodref_info, CONSTANT_Fieldref_info,
    CONSTANT_NameAndType_info, CONSTANT_InvokeDynamic_info,
    encode_modified_utf8
)
from jar_engine.model.member_info import (
    FieldInfo, MethodInfo, CodeAttribute, BootstrapMethodsAttribute,
    BootstrapMethod
)
from jar_engine.model.class_file import ClassFile
from jar_engine.model.jar_archive import JarArchive, Manifest
from jar_engine.walker.hierarchy import (
    InheritanceHierarchy, HierarchyNode, ClassOrigin
)
from jar_engine.walker.call_graph import (
    CallGraph, MethodReference, FieldReference, CallEdge
)
from jar_engine.walker.dependency_slice import (
    ClassDependencyGraph, DependencySlicer, DependencyKind
)
from jar_engine.triage.taxonomy import CapabilityDomain, DOMAIN_SIGNATURES
from jar_engine.triage.scorer import CapabilityTriager, ClassTriageResult
from jar_engine.triage.contextual_slice import ContextualSlicer
from jar_engine.renamer.schema import (
    SymbolMapping, MappingValidationError
)
from jar_engine.renamer.union_find import MethodEquivalenceCluster
from jar_engine.renamer.ref_rewriter import (
    ReferenceRewriter, RenamingReport, rewrite_descriptor
)
from jar_engine.renamer.repackager import JarRepackager
from jar_engine.renamer import SymbolRenamer
from jar_engine.cli.main import (
    build_parser, cmd_analyze, cmd_normalize, cmd_slice, cmd_rename
)


# ============================================================================
# Helpers for Constructing Test Bytecode and Classes
# ============================================================================

def make_adv_class(
    name: str,
    super_name: Optional[str] = "java/lang/Object",
    interfaces: Optional[List[str]] = None,
    access_flags: int = AccessFlags.PUBLIC,
    fields: Optional[List[FieldInfo]] = None,
    methods: Optional[List[MethodInfo]] = None,
) -> ClassFile:
    """Builds a complete in-memory ClassFile with proper constant pool indices."""
    cp = ConstantPool()
    this_c = cp.add_class(name)
    super_c = cp.add_class(super_name) if super_name else 0
    ifaces = [cp.add_class(i) for i in (interfaces or [])]
    return ClassFile(
        magic=JAVA_MAGIC,
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


def make_adv_method(
    cp: ConstantPool,
    name: str,
    desc: str,
    access_flags: int = AccessFlags.PUBLIC,
    code: Optional[bytes] = None,
    dedup_name: bool = True
) -> MethodInfo:
    """
    Creates a MethodInfo with an optional CodeAttribute attached.
    If dedup_name is False, creates a fresh CONSTANT_Utf8_info entry so overloads
    do not share the same constant pool UTF-8 index.
    """
    if dedup_name:
        name_idx = cp.add_utf8(name)
    else:
        name_idx = len(cp._entries)
        cp._entries.append(CONSTANT_Utf8_info(tag=ConstantTag.UTF8, value=name, raw_bytes=encode_modified_utf8(name)))
    desc_idx = cp.add_utf8(desc)
    attrs = []
    if code is not None:
        code_attr_name = cp.add_utf8("Code")
        attrs.append(CodeAttribute(code_attr_name, 2, 2, code))
    return MethodInfo(access_flags=access_flags, name_index=name_idx, descriptor_index=desc_idx, attributes=attrs)


def make_adv_field(
    cp: ConstantPool,
    name: str,
    desc: str,
    access_flags: int = AccessFlags.PUBLIC
) -> FieldInfo:
    """Creates a FieldInfo registered in the constant pool."""
    name_idx = cp.add_utf8(name)
    desc_idx = cp.add_utf8(desc)
    return FieldInfo(access_flags=access_flags, name_index=name_idx, descriptor_index=desc_idx, attributes=[])


# ============================================================================
# 1. Deep Diamond Hierarchies & Complex Multi-Path Inheritance
# ============================================================================

def test_adv_multi_tier_diamond_inheritance_dag():
    """
    Adversarial 1.1: 5-Tier Diamond Hierarchy DAG.
    Structure:
      Tier 0: Root (extends Object)
      Tier 1: BranchA, BranchB (extend Root)
      Tier 2: MidDiamond (extends BranchA, implements InterfaceBranchB)
      Tier 3: SubA (extends MidDiamond), SubB (extends MidDiamond)
      Tier 4: BottomDiamond (extends SubA)
    Verifies superclass order, subclass transitive closure, and LCA across multi-tier cross-edges.
    """
    arc = JarArchive()
    arc.add_class(make_adv_class("dia/Root"))
    arc.add_class(make_adv_class("dia/BranchA", super_name="dia/Root"))
    arc.add_class(make_adv_class("dia/IBranchB", super_name="java/lang/Object",
                                 access_flags=AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT))
    arc.add_class(make_adv_class("dia/MidDiamond", super_name="dia/BranchA", interfaces=["dia/IBranchB"]))
    arc.add_class(make_adv_class("dia/SubA", super_name="dia/MidDiamond"))
    arc.add_class(make_adv_class("dia/SubB", super_name="dia/MidDiamond"))
    arc.add_class(make_adv_class("dia/BottomDiamond", super_name="dia/SubA"))

    hier = arc.get_hierarchy()

    # Verify superclasses upward
    bottom_supers = hier.get_superclasses("dia/BottomDiamond")
    assert bottom_supers == ["dia/SubA", "dia/MidDiamond", "dia/BranchA", "dia/Root", "java/lang/Object"]

    # Verify subclasses transitive closure
    root_subs = hier.get_all_subclasses("dia/Root")
    assert root_subs == {"dia/BranchA", "dia/MidDiamond", "dia/SubA", "dia/SubB", "dia/BottomDiamond"}

    # MidDiamond subclasses
    mid_subs = hier.get_subclasses("dia/MidDiamond")
    assert mid_subs == {"dia/SubA", "dia/SubB", "dia/BottomDiamond"}

    # Lowest Common Ancestor across diamond branches
    assert hier.find_lowest_common_ancestor("dia/SubA", "dia/SubB") == "dia/MidDiamond"
    assert hier.find_lowest_common_ancestor("dia/BottomDiamond", "dia/SubB") == "dia/MidDiamond"
    assert hier.find_lowest_common_ancestor("dia/SubB", "dia/BranchA") == "dia/BranchA"
    assert hier.find_lowest_common_ancestor("dia/SubA", "dia/Root") == "dia/Root"


def test_adv_deep_diamond_interface_polymorphism():
    """
    Adversarial 1.2: Multi-interface Diamond Lattice.
    RootI -> LeftI, RightI -> MidI -> BottomClass.
    Verifies transitive interfaces and JVMS §4.10 interface merge to java/lang/Object.
    """
    arc = JarArchive()
    iface_flags = AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT

    arc.add_class(make_adv_class("iface/RootI", super_name="java/lang/Object", access_flags=iface_flags))
    arc.add_class(make_adv_class("iface/LeftI", super_name="java/lang/Object", interfaces=["iface/RootI"], access_flags=iface_flags))
    arc.add_class(make_adv_class("iface/RightI", super_name="java/lang/Object", interfaces=["iface/RootI"], access_flags=iface_flags))
    arc.add_class(make_adv_class("iface/MidI", super_name="java/lang/Object", interfaces=["iface/LeftI", "iface/RightI"], access_flags=iface_flags))
    arc.add_class(make_adv_class("iface/BottomClass", super_name="java/lang/Object", interfaces=["iface/MidI"]))

    hier = arc.get_hierarchy()

    # Transitive interfaces of BottomClass
    all_ifaces = hier.get_all_interfaces("iface/BottomClass")
    assert all_ifaces == {"iface/MidI", "iface/LeftI", "iface/RightI", "iface/RootI"}

    # Transitive implementors of RootI
    root_impls = hier.get_all_implementors("iface/RootI")
    assert root_impls == {"iface/LeftI", "iface/RightI", "iface/MidI", "iface/BottomClass"}

    # JVMS §4.10: Interface LCA always merges to java/lang/Object
    assert hier.find_lowest_common_ancestor("iface/LeftI", "iface/RightI") == "java/lang/Object"
    assert hier.find_lowest_common_ancestor("iface/MidI", "iface/RootI") == "java/lang/Object"
    assert hier.find_lowest_common_ancestor("iface/BottomClass", "iface/MidI") == "java/lang/Object"


def test_adv_diamond_multidimensional_array_lca():
    """
    Adversarial 1.3: Lowest Common Ancestor on Multidimensional Reference & Primitive Arrays.
    Tests:
    - Same dimension reference arrays resolve element LCA recursively.
    - Dimension mismatch collapses to java/lang/Object.
    - Primitive arrays invariant or collapse to java/lang/Object.
    """
    arc = JarArchive()
    arc.add_class(make_adv_class("arr/Root"))
    arc.add_class(make_adv_class("arr/SubA", super_name="arr/Root"))
    arc.add_class(make_adv_class("arr/SubB", super_name="arr/Root"))

    hier = arc.get_hierarchy()

    # Same dimension 3D reference arrays
    lca_3d = hier.find_lowest_common_ancestor("[[[Larr/SubA;", "[[[Larr/SubB;")
    assert lca_3d == "[[[Larr/Root;"

    # 1D reference arrays
    lca_1d = hier.find_lowest_common_ancestor("[Larr/SubA;", "[Larr/SubB;")
    assert lca_1d == "[Larr/Root;"

    # Dimension mismatch: 3D vs 2D -> java/lang/Object
    assert hier.find_lowest_common_ancestor("[[[Larr/SubA;", "[[Larr/SubA;") == "java/lang/Object"

    # Reference array vs primitive array -> java/lang/Object
    assert hier.find_lowest_common_ancestor("[Larr/SubA;", "[I") == "java/lang/Object"

    # Primitive array vs primitive array (different primitives) -> java/lang/Object
    assert hier.find_lowest_common_ancestor("[I", "[D") == "java/lang/Object"

    # Primitive array identity -> preserves itself
    assert hier.find_lowest_common_ancestor("[I", "[I") == "[I"

    # Array vs Object -> java/lang/Object
    assert hier.find_lowest_common_ancestor("[[[Larr/SubA;", "java/lang/Object") == "java/lang/Object"


def test_adv_diamond_assignability_lattice():
    """
    Adversarial 1.4: Subtype Assignability Lattice (target.isAssignableFrom(source)).
    Verifies classes, interfaces, covariance of reference arrays, invariance of primitive arrays,
    and Object/Cloneable/Serializable absorption rules.
    """
    arc = JarArchive()
    iface_flags = AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT
    arc.add_class(make_adv_class("lat/BaseI", super_name="java/lang/Object", access_flags=iface_flags))
    arc.add_class(make_adv_class("lat/BaseClass", super_name="java/lang/Object", interfaces=["lat/BaseI"]))
    arc.add_class(make_adv_class("lat/SubClass", super_name="lat/BaseClass"))

    hier = arc.get_hierarchy()

    # Direct class & interface assignability
    assert hier.is_assignable_from("lat/BaseClass", "lat/SubClass") is True
    assert hier.is_assignable_from("lat/SubClass", "lat/BaseClass") is False
    assert hier.is_assignable_from("lat/BaseI", "lat/SubClass") is True
    assert hier.is_assignable_from("lat/SubClass", "lat/BaseI") is False

    # Object absorbs anything
    assert hier.is_assignable_from("java/lang/Object", "lat/SubClass") is True
    assert hier.is_assignable_from("java/lang/Object", "[I") is True
    assert hier.is_assignable_from("java/lang/Object", "[[Llat/SubClass;") is True

    # Arrays are assignable to Cloneable and Serializable
    assert hier.is_assignable_from("java/lang/Cloneable", "[I") is True
    assert hier.is_assignable_from("java/io/Serializable", "[Llat/SubClass;") is True

    # Covariant reference arrays
    assert hier.is_assignable_from("[Llat/BaseClass;", "[Llat/SubClass;") is True
    assert hier.is_assignable_from("[Llat/SubClass;", "[Llat/BaseClass;") is False
    assert hier.is_assignable_from("[[Llat/BaseI;", "[[Llat/SubClass;") is True

    # Invariant primitive arrays
    assert hier.is_assignable_from("[I", "[I") is True
    assert hier.is_assignable_from("[I", "[J") is False
    assert hier.is_assignable_from("[I", "[B") is False


# ============================================================================
# 2. Circular Inheritance Handling & Cyclic Slicing Robustness
# ============================================================================

def test_adv_two_node_circular_inheritance_termination():
    """
    Adversarial 2.1: Mutual 2-node circular inheritance (Class A extends B, Class B extends A).
    Verifies that cycle detection catches it and graph walkers terminate deterministically
    without infinite loops or recursion overflow:
    - get_superclasses('cycle/ClassA') returns ['cycle/ClassB'] without re-visiting ClassA.
    - get_all_subclasses('cycle/ClassA') explores both nodes in the cycle cleanly.
    """
    arc = JarArchive()
    arc.add_class(make_adv_class("cycle/ClassA", super_name="cycle/ClassB"))
    arc.add_class(make_adv_class("cycle/ClassB", super_name="cycle/ClassA"))

    hier = arc.get_hierarchy()

    # Cycle detection
    assert hier.has_cycles() is True
    cycles = hier.get_cycles()
    assert len(cycles) > 0
    cycle_nodes = set(cycles[0])
    assert "cycle/ClassA" in cycle_nodes and "cycle/ClassB" in cycle_nodes

    # Walking superclasses terminates cleanly (visited set stops at ClassA)
    supers_a = hier.get_superclasses("cycle/ClassA")
    assert supers_a == ["cycle/ClassB"]

    # Walking subclasses terminates cleanly (both reachable in cycle)
    subs_a = hier.get_all_subclasses("cycle/ClassA")
    assert subs_a == {"cycle/ClassA", "cycle/ClassB"}

    # LCA calculation terminates cleanly
    lca = hier.find_lowest_common_ancestor("cycle/ClassA", "cycle/ClassB")
    assert lca in ("cycle/ClassA", "cycle/ClassB", "java/lang/Object")


def test_adv_three_node_circular_interface_chain():
    """
    Adversarial 2.2: 3-node circular interface hierarchy (I1 -> I2 -> I3 -> I1).
    Verifies get_all_interfaces, get_all_implementors, and cycle detectors terminate.
    """
    arc = JarArchive()
    iface_flags = AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT
    arc.add_class(make_adv_class("cycle/I1", super_name="java/lang/Object", interfaces=["cycle/I2"], access_flags=iface_flags))
    arc.add_class(make_adv_class("cycle/I2", super_name="java/lang/Object", interfaces=["cycle/I3"], access_flags=iface_flags))
    arc.add_class(make_adv_class("cycle/I3", super_name="java/lang/Object", interfaces=["cycle/I1"], access_flags=iface_flags))
    arc.add_class(make_adv_class("cycle/User", super_name="java/lang/Object", interfaces=["cycle/I1"]))

    hier = arc.get_hierarchy()

    assert hier.has_cycles() is True
    all_ifaces = hier.get_all_interfaces("cycle/User")
    assert all_ifaces == {"cycle/I1", "cycle/I2", "cycle/I3"}

    impls_i1 = hier.get_all_implementors("cycle/I1")
    assert "cycle/User" in impls_i1
    assert "cycle/I2" in impls_i1 or "cycle/I3" in impls_i1


def test_adv_self_referential_class_inheritance():
    """
    Adversarial 2.3: Self-referential class (Class S extends Class S).
    Verifies no crash on superclass chain, subclass closure, or cycle check.
    """
    arc = JarArchive()
    arc.add_class(make_adv_class("cycle/SelfRef", super_name="cycle/SelfRef"))

    hier = arc.get_hierarchy()
    assert hier.has_cycles() is True

    # Since SelfRef is initialized in visited set, loop immediately halts
    supers = hier.get_superclasses("cycle/SelfRef")
    assert supers == []

    # Subclasses closure includes SelfRef
    subs = hier.get_all_subclasses("cycle/SelfRef")
    assert subs == {"cycle/SelfRef"}


def test_adv_cyclic_dependency_graph_tarjan_sccs():
    """
    Adversarial 2.4: Tarjan SCC analysis on dependency graph.
    Verifies multi-node mutual cycles, explicit self-loops, and add_dependency guard behavior.
    """
    cdg = ClassDependencyGraph()
    # Cycle 1: X <-> Y
    cdg.add_dependency("dep/X", "dep/Y", DependencyKind.METHOD_CALL)
    cdg.add_dependency("dep/Y", "dep/X", DependencyKind.METHOD_CALL)

    # Note: cdg.add_dependency explicitly suppresses self-loops (src == dst).
    # Adding self-loop edge directly to underlying graph tests Tarjan self-loop handling.
    cdg.graph.add_edge("dep/Z", "dep/Z", kinds={"field_access"}, count=1, details=[])

    # Acyclic link: X -> W
    cdg.add_dependency("dep/X", "dep/W", DependencyKind.INHERITANCE)

    assert cdg.has_cycles() is True
    cyclic_comps = cdg.get_cyclic_components()
    assert len(cyclic_comps) == 2

    comp_sets = [set(c) for c in cyclic_comps]
    assert {"dep/X", "dep/Y"} in comp_sets
    assert {"dep/Z"} in comp_sets

    # Tarjan SCCs
    sccs = cdg.get_sccs()
    assert len(sccs) == 3  # {X, Y}, {Z}, {W}


def test_adv_cyclic_forward_backward_dependency_slices():
    """
    Adversarial 2.5: Forward and backward slicing across cyclic dependencies terminates cleanly.
    """
    cdg = ClassDependencyGraph()
    cdg.add_dependency("slice/A", "slice/B", DependencyKind.METHOD_CALL)
    cdg.add_dependency("slice/B", "slice/C", DependencyKind.METHOD_CALL)
    cdg.add_dependency("slice/C", "slice/A", DependencyKind.METHOD_CALL)  # cycle A->B->C->A
    cdg.add_dependency("slice/C", "slice/Sink", DependencyKind.METHOD_CALL)

    # Forward slice from A reaches all cycle members plus Sink
    f_slice = cdg.forward_slice("slice/A")
    assert f_slice == {"slice/A", "slice/B", "slice/C", "slice/Sink"}

    # Backward slice from Sink reaches entire cycle
    b_slice = cdg.backward_slice("slice/Sink")
    assert b_slice == {"slice/Sink", "slice/A", "slice/B", "slice/C"}


# ============================================================================
# 3. Polymorphic Call Graph Corner Cases & Dispatch Resolution
# ============================================================================

def test_adv_cha_vs_rta_subclass_filtering():
    """
    Adversarial 3.1: CHA vs RTA dispatch pruning.
    Shape hierarchy: Circle, Square, Triangle all implement Shape.draw()V.
    Caller does invokevirtual Shape.draw()V.
    Only Circle is instantiated via NEW in the archive.
    - CHA resolves all 3 subclasses.
    - RTA resolves ONLY Circle.
    """
    arc = JarArchive()

    # Shape base
    cf_shape = make_adv_class("poly/Shape")
    cf_shape.methods.append(make_adv_method(cf_shape.constant_pool, "draw", "()V", code=bytes([Opcode.RETURN])))
    arc.add_class(cf_shape)

    # Subclasses
    for name in ["poly/Circle", "poly/Square", "poly/Triangle"]:
        cf = make_adv_class(name, super_name="poly/Shape")
        cf.methods.append(make_adv_method(cf.constant_pool, "draw", "()V", code=bytes([Opcode.RETURN])))
        arc.add_class(cf)

    # Caller with invokevirtual Shape.draw()V and new poly/Circle
    cf_caller = make_adv_class("poly/Caller")
    cp_c = cf_caller.constant_pool
    circle_idx = cp_c.add_class("poly/Circle")
    mref_idx = cp_c.add_member_ref(ConstantTag.METHODREF, "poly/Shape", "draw", "()V")

    # Bytecode: NEW Circle, DUP, ALOAD_0, INVOKEVIRTUAL Shape.draw, RETURN
    code = bytes([
        Opcode.NEW, (circle_idx >> 8) & 0xFF, circle_idx & 0xFF,
        Opcode.DUP,
        Opcode.ALOAD_0,
        Opcode.INVOKEVIRTUAL, (mref_idx >> 8) & 0xFF, mref_idx & 0xFF,
        Opcode.RETURN
    ])
    cf_caller.methods.append(make_adv_method(cp_c, "run", "()V", code=code))
    arc.add_class(cf_caller)

    hier = arc.get_hierarchy()

    # CHA mode: all subclasses reachable
    cg_cha = CallGraph.build(arc, hier, mode="cha")
    callees_cha = cg_cha.get_callees("Lpoly/Caller;->run()V")
    assert "Lpoly/Circle;->draw()V" in callees_cha
    assert "Lpoly/Square;->draw()V" in callees_cha
    assert "Lpoly/Triangle;->draw()V" in callees_cha

    # RTA mode: only instantiated poly/Circle reachable
    cg_rta = CallGraph.build(arc, hier, mode="rta")
    callees_rta = cg_rta.get_callees("Lpoly/Caller;->run()V")
    assert "Lpoly/Circle;->draw()V" in callees_rta
    assert "Lpoly/Square;->draw()V" not in callees_rta
    assert "Lpoly/Triangle;->draw()V" not in callees_rta


def test_adv_default_methods_maximally_specific_selection():
    """
    Adversarial 3.2: Default Method Selection per JVMS §5.4.3.3.
    Interface SuperI provides default execute()V.
    Interface SubI extends SuperI and overrides default execute()V.
    Class Worker implements SuperI and SubI.
    Call graph must resolve Worker.execute() to SubI.execute() because SubI is maximally specific.
    """
    arc = JarArchive()
    iface_flags = AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT

    # SuperI with default execute()V
    cf_super = make_adv_class("jvm/SuperI", super_name="java/lang/Object", access_flags=iface_flags)
    cf_super.methods.append(make_adv_method(cf_super.constant_pool, "execute", "()V", code=bytes([Opcode.RETURN])))
    arc.add_class(cf_super)

    # SubI extends SuperI with overridden default execute()V
    cf_sub = make_adv_class("jvm/SubI", super_name="java/lang/Object", interfaces=["jvm/SuperI"], access_flags=iface_flags)
    cf_sub.methods.append(make_adv_method(cf_sub.constant_pool, "execute", "()V", code=bytes([Opcode.RETURN])))
    arc.add_class(cf_sub)

    # Worker implements both, does not declare execute()
    cf_worker = make_adv_class("jvm/Worker", super_name="java/lang/Object", interfaces=["jvm/SuperI", "jvm/SubI"])
    arc.add_class(cf_worker)

    # Caller invoking Worker.execute()
    cf_caller = make_adv_class("jvm/Caller")
    cp_c = cf_caller.constant_pool
    mref = cp_c.add_member_ref(ConstantTag.METHODREF, "jvm/Worker", "execute", "()V")
    code = bytes([Opcode.ALOAD_0, Opcode.INVOKEVIRTUAL, (mref >> 8) & 0xFF, mref & 0xFF, Opcode.RETURN])
    cf_caller.methods.append(make_adv_method(cp_c, "test", "()V", code=code))
    arc.add_class(cf_caller)

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)
    callees = cg.get_callees("Ljvm/Caller;->test()V")

    # Maximally-specific rule eliminates SuperI in favor of SubI
    assert "Ljvm/SubI;->execute()V" in callees
    assert "Ljvm/SuperI;->execute()V" not in callees


def test_adv_default_methods_unrelated_interfaces_tie_break():
    """
    Adversarial 3.3: Two unrelated interfaces AlphaI and BetaI both define default action()V.
    Class Impl implements both.
    Call graph must deterministically resolve to the first sorted candidate without raising exceptions.
    """
    arc = JarArchive()
    iface_flags = AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT

    cf_a = make_adv_class("jvm/AlphaI", super_name="java/lang/Object", access_flags=iface_flags)
    cf_a.methods.append(make_adv_method(cf_a.constant_pool, "action", "()V", code=bytes([Opcode.RETURN])))
    arc.add_class(cf_a)

    cf_b = make_adv_class("jvm/BetaI", super_name="java/lang/Object", access_flags=iface_flags)
    cf_b.methods.append(make_adv_method(cf_b.constant_pool, "action", "()V", code=bytes([Opcode.RETURN])))
    arc.add_class(cf_b)

    cf_impl = make_adv_class("jvm/MultiImpl", super_name="java/lang/Object", interfaces=["jvm/AlphaI", "jvm/BetaI"])
    arc.add_class(cf_impl)

    hier = arc.get_hierarchy()
    resolved = CallGraph()._find_default_method("jvm/MultiImpl", "action", "()V", arc, hier)
    assert resolved is not None
    decl_cls, m_info = resolved
    assert decl_cls in ("jvm/AlphaI", "jvm/BetaI")


def test_adv_static_method_hiding_vs_virtual_dispatch():
    """
    Adversarial 3.4: Static method hiding on superclass vs subclass.
    Super has static util()V, Sub has static util()V.
    Caller invoking invokestatic Sub.util()V targets Sub; invoking Super.util()V targets Super.
    """
    arc = JarArchive()

    cf_super = make_adv_class("dispatch/Super")
    cf_super.methods.append(make_adv_method(cf_super.constant_pool, "util", "()V",
                                            access_flags=AccessFlags.PUBLIC | AccessFlags.STATIC, code=bytes([Opcode.RETURN])))
    arc.add_class(cf_super)

    cf_sub = make_adv_class("dispatch/Sub", super_name="dispatch/Super")
    cf_sub.methods.append(make_adv_method(cf_sub.constant_pool, "util", "()V",
                                          access_flags=AccessFlags.PUBLIC | AccessFlags.STATIC, code=bytes([Opcode.RETURN])))
    arc.add_class(cf_sub)

    # Caller calls Sub.util()
    cf_caller = make_adv_class("dispatch/Caller")
    cp = cf_caller.constant_pool
    mref_sub = cp.add_member_ref(ConstantTag.METHODREF, "dispatch/Sub", "util", "()V")
    code = bytes([Opcode.INVOKESTATIC, (mref_sub >> 8) & 0xFF, mref_sub & 0xFF, Opcode.RETURN])
    cf_caller.methods.append(make_adv_method(cp, "callSub", "()V", code=code))
    arc.add_class(cf_caller)

    cg = CallGraph.build(arc)
    callees = cg.get_callees("Ldispatch/Caller;->callSub()V")
    assert "Ldispatch/Sub;->util()V" in callees
    assert "Ldispatch/Super;->util()V" not in callees


def test_adv_field_resolution_interface_constants():
    """
    Adversarial 3.5: Field resolution traversing interfaces (JVMS §5.4.3.2).
    Interface Constants defines TIMEOUT: I.
    Class Service implements Constants.
    Client issues GETSTATIC on Service.TIMEOUT: I.
    Call graph correctly resolves declaring class to Constants.
    """
    arc = JarArchive()
    iface_flags = AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT
    cf_iface = make_adv_class("fld/Constants", super_name="java/lang/Object", access_flags=iface_flags)
    cf_iface.fields.append(make_adv_field(cf_iface.constant_pool, "TIMEOUT", "I",
                                          access_flags=AccessFlags.PUBLIC | AccessFlags.STATIC | AccessFlags.FINAL))
    arc.add_class(cf_iface)

    cf_svc = make_adv_class("fld/Service", super_name="java/lang/Object", interfaces=["fld/Constants"])
    arc.add_class(cf_svc)

    hier = arc.get_hierarchy()
    cg = CallGraph.build(arc, hier)
    decl_cls = cg._resolve_declaring_field_class("fld/Service", "TIMEOUT", "I", arc, hier)
    assert decl_cls == "fld/Constants"


def test_adv_invokespecial_target_resolution_matrix():
    """
    Adversarial 3.6: invokespecial target resolution matrix:
    - <init> constructor -> targets receiver class
    - private method on self -> targets caller class
    - super call -> targets direct superclass
    """
    arc = JarArchive()

    cf_super = make_adv_class("special/Super")
    cf_super.methods.append(make_adv_method(cf_super.constant_pool, "onEvent", "()V", code=bytes([Opcode.RETURN])))
    arc.add_class(cf_super)

    cf_sub = make_adv_class("special/Sub", super_name="special/Super")
    cp = cf_sub.constant_pool

    # Private method
    cf_sub.methods.append(make_adv_method(cp, "secret", "()V", access_flags=AccessFlags.PRIVATE, code=bytes([Opcode.RETURN])))

    # Caller method calling <init>, private method, and super method
    init_ref = cp.add_member_ref(ConstantTag.METHODREF, "special/Super", "<init>", "()V")
    priv_ref = cp.add_member_ref(ConstantTag.METHODREF, "special/Sub", "secret", "()V")
    super_ref = cp.add_member_ref(ConstantTag.METHODREF, "special/Super", "onEvent", "()V")

    code = bytes([
        Opcode.ALOAD_0, Opcode.INVOKESPECIAL, (init_ref >> 8) & 0xFF, init_ref & 0xFF,
        Opcode.ALOAD_0, Opcode.INVOKESPECIAL, (priv_ref >> 8) & 0xFF, priv_ref & 0xFF,
        Opcode.ALOAD_0, Opcode.INVOKESPECIAL, (super_ref >> 8) & 0xFF, super_ref & 0xFF,
        Opcode.RETURN
    ])
    cf_sub.methods.append(make_adv_method(cp, "testAll", "()V", code=code))
    arc.add_class(cf_sub)

    cg = CallGraph.build(arc)
    callees = cg.get_callees("Lspecial/Sub;->testAll()V")
    assert "Lspecial/Super;-><init>()V" in callees
    assert "Lspecial/Sub;->secret()V" in callees
    assert "Lspecial/Super;->onEvent()V" in callees


def test_adv_invokedynamic_unrecognized_bootstrap_fallback():
    """
    Adversarial 3.7: invokedynamic with non-standard bootstrap method.
    Call graph creates synthetic indy edge L<class>;->indy$<name><desc> without crashing.
    """
    arc = JarArchive()
    cf = make_adv_class("indy/DynamicClass")
    cp = cf.constant_pool

    nat_idx = cp.add_name_and_type("customBootstrap", "()Ljava/lang/String;")
    indy_entry = CONSTANT_InvokeDynamic_info(bootstrap_method_attr_index=99, name_and_type_index=nat_idx)
    indy_idx = len(cp._entries)
    cp._entries.append(indy_entry)

    code = bytes([
        Opcode.INVOKEDYNAMIC, (indy_idx >> 8) & 0xFF, indy_idx & 0xFF, 0, 0,
        Opcode.POP,
        Opcode.RETURN
    ])
    cf.methods.append(make_adv_method(cp, "callIndy", "()V", code=code))
    arc.add_class(cf)

    cg = CallGraph.build(arc)
    callees = cg.get_callees("Lindy/DynamicClass;->callIndy()V")
    assert len(callees) == 1
    callee = next(iter(callees))
    assert callee.startswith("Lindy/DynamicClass;->indy$customBootstrap")


# ============================================================================
# 4. Condensation DAG & Multiple Disconnected SCCs
# ============================================================================

def test_adv_condensation_dag_multiple_disconnected_sccs():
    """
    Adversarial 4.1: Condensation DAG on multiple independent disconnected SCCs.
    Structure:
    - Component 1: 2-node cycle {A <-> B}
    - Component 2: 3-node cycle {C -> D -> E -> C}
    - Component 3: Linear chain F -> G -> H (3 singleton SCCs)
    - Component 4: Isolated singleton {Z}
    Verifies 6 total SCCs, exactly 2 cyclic SCCs, and condensation quotient DAG.
    """
    cdg = ClassDependencyGraph()

    # Component 1: {A, B}
    cdg.add_dependency("scc/A", "scc/B", DependencyKind.METHOD_CALL)
    cdg.add_dependency("scc/B", "scc/A", DependencyKind.METHOD_CALL)

    # Component 2: {C, D, E}
    cdg.add_dependency("scc/C", "scc/D", DependencyKind.METHOD_CALL)
    cdg.add_dependency("scc/D", "scc/E", DependencyKind.METHOD_CALL)
    cdg.add_dependency("scc/E", "scc/C", DependencyKind.METHOD_CALL)

    # Component 3: F -> G -> H
    cdg.add_dependency("scc/F", "scc/G", DependencyKind.INHERITANCE)
    cdg.add_dependency("scc/G", "scc/H", DependencyKind.INHERITANCE)

    # Component 4: Z (isolated)
    cdg.graph.add_node("scc/Z")

    sccs = cdg.get_sccs()
    assert len(sccs) == 6

    cyclic = cdg.get_cyclic_components()
    assert len(cyclic) == 2
    cyclic_sets = [set(c) for c in cyclic]
    assert {"scc/A", "scc/B"} in cyclic_sets
    assert {"scc/C", "scc/D", "scc/E"} in cyclic_sets

    cond = cdg.condensation_graph()
    assert nx.is_directed_acyclic_graph(cond)
    assert len(cond.nodes) == 6


def test_adv_topological_sort_build_vs_dependency_order():
    """
    Adversarial 4.2: Topological sort ordering for condensation components:
    - direction='build': prerequisites (leaves) first.
    - direction='dependency': roots first.
    """
    cdg = ClassDependencyGraph()
    # A -> B -> C
    cdg.add_dependency("topo/A", "topo/B", DependencyKind.INHERITANCE)
    cdg.add_dependency("topo/B", "topo/C", DependencyKind.INHERITANCE)

    build_order = cdg.topological_sort(direction="build")
    # In build order, C (prerequisite) comes before A (dependent)
    c_idx = next(i for i, comp in enumerate(build_order) if "topo/C" in comp)
    a_idx = next(i for i, comp in enumerate(build_order) if "topo/A" in comp)
    assert c_idx < a_idx

    dep_order = cdg.topological_sort(direction="dependency")
    # In dependency order, A (root dependent) comes before C
    c_idx_dep = next(i for i, comp in enumerate(dep_order) if "topo/C" in comp)
    a_idx_dep = next(i for i, comp in enumerate(dep_order) if "topo/A" in comp)
    assert a_idx_dep < c_idx_dep


def test_adv_disconnected_sccs_slice_isolation():
    """
    Adversarial 4.3: Slicing isolation across disconnected components.
    Forward slice from A must not touch disconnected component C or Z.
    """
    cdg = ClassDependencyGraph()
    cdg.add_dependency("iso/A", "iso/B", DependencyKind.METHOD_CALL)
    cdg.add_dependency("iso/C", "iso/D", DependencyKind.METHOD_CALL)
    cdg.graph.add_node("iso/Z")

    f_slice_a = cdg.forward_slice("iso/A")
    assert f_slice_a == {"iso/A", "iso/B"}
    assert "iso/C" not in f_slice_a
    assert "iso/Z" not in f_slice_a

    b_slice_d = cdg.backward_slice("iso/D")
    assert b_slice_d == {"iso/D", "iso/C"}
    assert "iso/A" not in b_slice_d


# ============================================================================
# 5. Complex Renames Across Overloaded Methods & Override Clustering
# ============================================================================

def test_adv_overloaded_methods_distinct_utf8_selective_renaming():
    """
    Adversarial 5.1: Selective renaming of overloaded methods with distinct UTF-8 CP entries.
    Class Processor has 3 overloads:
    - handle(I)V -> renamed to handleInt
    - handle(Ljava/lang/String;)V -> NOT renamed (remains handle)
    - handle([B)V -> renamed to handleBytes
    When each method has its own distinct UTF-8 constant pool entry,
    ReferenceRewriter selectively updates only the matching overloads.
    """
    arc = JarArchive()
    cf_proc = make_adv_class("over/Processor")
    cp_p = cf_proc.constant_pool
    cf_proc.methods.append(make_adv_method(cp_p, "handle", "(I)V", code=bytes([Opcode.RETURN]), dedup_name=False))
    cf_proc.methods.append(make_adv_method(cp_p, "handle", "(Ljava/lang/String;)V", code=bytes([Opcode.RETURN]), dedup_name=False))
    cf_proc.methods.append(make_adv_method(cp_p, "handle", "([B)V", code=bytes([Opcode.RETURN]), dedup_name=False))
    arc.add_class(cf_proc)

    mapping = SymbolMapping(
        methods={
            "over/Processor#handle#(I)V": "handleInt",
            "over/Processor#handle#([B)V": "handleBytes"
        }
    )

    renamer = SymbolRenamer()
    report = renamer.apply_mapping(arc, mapping)

    assert report.methods_renamed == 2

    # Verify Processor methods
    proc_updated = arc.get_class("over/Processor")
    method_names = {m.name(proc_updated.constant_pool): m.descriptor(proc_updated.constant_pool) for m in proc_updated.methods}
    assert "handleInt" in method_names and method_names["handleInt"] == "(I)V"
    assert "handleBytes" in method_names and method_names["handleBytes"] == "([B)V"
    assert "handle" in method_names and method_names["handle"] == "(Ljava/lang/String;)V"


def test_adv_overloaded_methods_shared_utf8_coupling():
    """
    Adversarial 5.2: Shared UTF-8 coupling across overloaded methods.
    When overloads share a single deduplicated CONSTANT_Utf8_info entry in the constant pool
    (the default behavior of standard constant pool builders), in-place modification
    updates the shared UTF-8 entry, modifying all methods sharing that name index.
    """
    arc = JarArchive()
    cf_proc = make_adv_class("over/SharedProcessor")
    cp_p = cf_proc.constant_pool
    # Both share the same UTF-8 entry via cp.add_utf8
    cf_proc.methods.append(make_adv_method(cp_p, "compute", "(I)I", code=bytes([Opcode.IRETURN]), dedup_name=True))
    cf_proc.methods.append(make_adv_method(cp_p, "compute", "(J)J", code=bytes([Opcode.LRETURN]), dedup_name=True))
    arc.add_class(cf_proc)

    mapping = SymbolMapping(methods={"over/SharedProcessor#compute#(I)I": "computeInt"})
    renamer = SymbolRenamer()
    report = renamer.apply_mapping(arc, mapping)

    assert report.methods_renamed == 1
    updated_cf = arc.get_class("over/SharedProcessor")
    # Because both methods pointed to the same UTF-8 entry, in-place mutation renamed the shared string
    method_names = [m.name(updated_cf.constant_pool) for m in updated_cf.methods]
    assert method_names == ["computeInt", "computeInt"]


def test_adv_overloaded_methods_inheritance_override_propagation():
    """
    Adversarial 5.3: Overloaded methods in inheritance hierarchy:
    Base declares calculate(I)I and calculate(J)J with distinct UTF-8 entries.
    Sub overrides calculate(I)I only.
    Renaming Base#calculate#(I)I automatically propagates to Sub#calculate#(I)I via Union-Find,
    while calculate(J)J remains untouched in both classes.
    """
    arc = JarArchive()

    cf_base = make_adv_class("over/Base")
    cp_b = cf_base.constant_pool
    cf_base.methods.append(make_adv_method(cp_b, "calculate", "(I)I", code=bytes([Opcode.IRETURN]), dedup_name=False))
    cf_base.methods.append(make_adv_method(cp_b, "calculate", "(J)J", code=bytes([Opcode.LRETURN]), dedup_name=False))
    arc.add_class(cf_base)

    cf_sub = make_adv_class("over/Sub", super_name="over/Base")
    cp_s = cf_sub.constant_pool
    cf_sub.methods.append(make_adv_method(cp_s, "calculate", "(I)I", code=bytes([Opcode.IRETURN]), dedup_name=False))
    arc.add_class(cf_sub)

    mapping = SymbolMapping(
        methods={
            "over/Base#calculate#(I)I": "calculateInt"
        }
    )

    renamer = SymbolRenamer()
    report = renamer.apply_mapping(arc, mapping)

    # Sub's overridden calculate(I)I must be renamed too
    sub_cf = arc.get_class("over/Sub")
    sub_method_names = [m.name(sub_cf.constant_pool) for m in sub_cf.methods]
    assert "calculateInt" in sub_method_names
    assert "calculate" not in sub_method_names

    # Base's calculate(J)J was untouched because it had a distinct UTF-8 entry
    base_cf = arc.get_class("over/Base")
    base_names = [m.name(base_cf.constant_pool) for m in base_cf.methods]
    assert "calculateInt" in base_names
    assert "calculate" in base_names


def test_adv_method_cluster_pinned_name_matching():
    """
    Adversarial 5.4: Method cluster pinning:
    Class Model overrides toString()Ljava/lang/String; (JDK Object method)
    and defines custom overload toString(Z)Ljava/lang/String;.
    Under union_find.py's is_jdk_method, any method matching OBJECT_DECLARED_METHODS
    is pinned by name regardless of descriptor.
    Both methods are pinned and skipped, preserving their names intact.
    """
    arc = JarArchive()
    cf = make_adv_class("pin/Model")
    cp = cf.constant_pool
    cf.methods.append(make_adv_method(cp, "toString", "()Ljava/lang/String;", code=bytes([Opcode.ARETURN]), dedup_name=False))
    cf.methods.append(make_adv_method(cp, "toString", "(Z)Ljava/lang/String;", code=bytes([Opcode.ARETURN]), dedup_name=False))
    arc.add_class(cf)

    mapping = SymbolMapping(
        methods={
            "pin/Model#toString#()Ljava/lang/String;": "asString",
            "pin/Model#toString#(Z)Ljava/lang/String;": "formatBoolean"
        }
    )

    renamer = SymbolRenamer()
    report = renamer.apply_mapping(arc, mapping)

    # Both are pinned because method_name is in OBJECT_DECLARED_METHODS
    assert report.pinned_skipped == 2

    updated_cf = arc.get_class("pin/Model")
    methods_dict = {m.descriptor(updated_cf.constant_pool): m.name(updated_cf.constant_pool) for m in updated_cf.methods}
    assert methods_dict["()Ljava/lang/String;"] == "toString"
    assert methods_dict["(Z)Ljava/lang/String;"] == "toString"


def test_adv_method_cluster_custom_overload_unpinned_renaming():
    """
    Adversarial 5.5: Custom non-JDK method overloads can be selectively renamed
    when not in OBJECT_DECLARED_METHODS.
    """
    arc = JarArchive()
    cf = make_adv_class("pin/CustomService")
    cp = cf.constant_pool
    cf.methods.append(make_adv_method(cp, "dispatch", "(I)V", code=bytes([Opcode.RETURN]), dedup_name=False))
    cf.methods.append(make_adv_method(cp, "dispatch", "(Z)V", code=bytes([Opcode.RETURN]), dedup_name=False))
    arc.add_class(cf)

    mapping = SymbolMapping(
        methods={
            "pin/CustomService#dispatch#(I)V": "dispatchInt",
            "pin/CustomService#dispatch#(Z)V": "dispatchBool"
        }
    )

    renamer = SymbolRenamer()
    report = renamer.apply_mapping(arc, mapping)

    assert report.pinned_skipped == 0
    assert report.methods_renamed == 2

    updated_cf = arc.get_class("pin/CustomService")
    names = {m.descriptor(updated_cf.constant_pool): m.name(updated_cf.constant_pool) for m in updated_cf.methods}
    assert names["(I)V"] == "dispatchInt"
    assert names["(Z)V"] == "dispatchBool"


# ============================================================================
# 6. Access Widening Across Package Boundaries
# ============================================================================

def test_adv_access_widening_package_private_to_new_package():
    """
    Adversarial 6.1: Moving package-private class across packages widens access to PUBLIC.
    Old: com/pkg_a/InternalWorker (access_flags = 0)
    New: com/pkg_b/InternalWorker
    Access flags must become PUBLIC, and report.access_widened must increment.
    """
    arc = JarArchive()
    cf = make_adv_class("com/pkg_a/InternalWorker", access_flags=0)
    arc.add_class(cf)

    mapping = SymbolMapping(classes={"com/pkg_a/InternalWorker": "com/pkg_b/InternalWorker"})
    renamer = SymbolRenamer()
    report = renamer.apply_mapping(arc, mapping)

    assert report.classes_renamed == 1
    assert report.access_widened == 1

    updated_cf = arc.get_class("com/pkg_b/InternalWorker")
    assert bool(updated_cf.access_flags & AccessFlags.PUBLIC) is True


def test_adv_access_widening_same_package_unchanged():
    """
    Adversarial 6.2: Renaming package-private class within the same package does NOT widen access.
    Old: com/pkg_a/InternalWorker (access_flags = 0)
    New: com/pkg_a/RenamedWorker
    Access flags remains package-private (PUBLIC not set), report.access_widened == 0.
    """
    arc = JarArchive()
    cf = make_adv_class("com/pkg_a/InternalWorker", access_flags=0)
    arc.add_class(cf)

    mapping = SymbolMapping(classes={"com/pkg_a/InternalWorker": "com/pkg_a/RenamedWorker"})
    renamer = SymbolRenamer()
    report = renamer.apply_mapping(arc, mapping)

    assert report.classes_renamed == 1
    assert report.access_widened == 0

    updated_cf = arc.get_class("com/pkg_a/RenamedWorker")
    assert bool(updated_cf.access_flags & AccessFlags.PUBLIC) is False


def test_adv_access_widening_already_public_unchanged():
    """
    Adversarial 6.3: Renaming already-public class across packages preserves PUBLIC without double-widening.
    """
    arc = JarArchive()
    cf = make_adv_class("com/pkg_a/PublicWorker", access_flags=AccessFlags.PUBLIC)
    arc.add_class(cf)

    mapping = SymbolMapping(classes={"com/pkg_a/PublicWorker": "com/pkg_b/PublicWorker"})
    renamer = SymbolRenamer()
    report = renamer.apply_mapping(arc, mapping)

    assert report.classes_renamed == 1
    assert report.access_widened == 0

    updated_cf = arc.get_class("com/pkg_b/PublicWorker")
    assert bool(updated_cf.access_flags & AccessFlags.PUBLIC) is True


def test_adv_access_widening_default_root_package_transitions():
    """
    Adversarial 6.4: Transitions involving default (root) package:
    - Default package to named package: RootHelper -> com/app/RootHelper (widens if not public)
    - Named package to default package: com/app/SubHelper -> SubHelper (widens if not public)
    """
    arc = JarArchive()
    arc.add_class(make_adv_class("RootHelper", access_flags=0))
    arc.add_class(make_adv_class("com/app/SubHelper", access_flags=0))

    mapping = SymbolMapping(classes={
        "RootHelper": "com/app/RootHelper",
        "com/app/SubHelper": "SubHelper"
    })
    renamer = SymbolRenamer()
    report = renamer.apply_mapping(arc, mapping)

    assert report.access_widened == 2
    assert bool(arc.get_class("com/app/RootHelper").access_flags & AccessFlags.PUBLIC) is True
    assert bool(arc.get_class("SubHelper").access_flags & AccessFlags.PUBLIC) is True


# ============================================================================
# 7. SPI Synchronization with Multiple Service Providers & Manifest
# ============================================================================

def test_adv_spi_synchronization_multiple_providers_and_comments():
    """
    Adversarial 7.1: SPI Provider synchronization with multiple providers, comments, and empty lines.
    META-INF/services/com.example.IService contains:
      # Main active providers
      com.example.ProviderAlpha
      com.example.ProviderBeta
      
      # Fallback provider
      com.example.ProviderGamma
    Mapping renames:
      com/example/IService -> org/spi/PluginService
      com/example/ProviderAlpha -> org/spi/FastAlpha
      com/example/ProviderGamma -> org/spi/SlowGamma
      (ProviderBeta is NOT renamed)
    Verifies:
      Resource filename renamed to META-INF/services/org.spi.PluginService
      Lines updated cleanly with comments and ProviderBeta preserved.
    """
    arc = JarArchive()
    spi_name = "META-INF/services/com.example.IService"
    spi_content = (
        "# Main active providers\n"
        "com.example.ProviderAlpha\n"
        "com.example.ProviderBeta\n"
        "\n"
        "# Fallback provider\n"
        "com.example.ProviderGamma\n"
    ).encode("utf-8")
    arc.resources[spi_name] = spi_content

    mapping = SymbolMapping(classes={
        "com/example/IService": "org/spi/PluginService",
        "com/example/ProviderAlpha": "org/spi/FastAlpha",
        "com/example/ProviderGamma": "org/spi/SlowGamma",
    })

    repackager = JarRepackager()
    synced = repackager.synchronize_manifest_and_services(arc, mapping)

    assert synced >= 2
    assert spi_name not in arc.resources

    new_spi_name = "META-INF/services/org.spi.PluginService"
    assert new_spi_name in arc.resources

    text = arc.resources[new_spi_name].decode("utf-8")
    lines = [line.strip() for line in text.splitlines()]

    assert "# Main active providers" in lines
    assert "org.spi.FastAlpha" in lines
    assert "com.example.ProviderBeta" in lines
    assert "# Fallback provider" in lines
    assert "org.spi.SlowGamma" in lines


def test_adv_spi_multiple_service_files_concurrent_sync():
    """
    Adversarial 7.2: Multiple distinct SPI service interface registrations synchronized in one pass.
    Verifies that synchronized provider lines count matches the number of provider class renames (2).
    """
    arc = JarArchive()
    arc.resources["META-INF/services/com.io.Reader"] = b"com.io.FileReader\n"
    arc.resources["META-INF/services/com.io.Writer"] = b"com.io.FileWriter\n"

    mapping = SymbolMapping(classes={
        "com/io/Reader": "org/io/DataReader",
        "com/io/Writer": "org/io/DataWriter",
        "com/io/FileReader": "org/io/DiskReader",
        "com/io/FileWriter": "org/io/DiskWriter",
    })

    repackager = JarRepackager()
    synced = repackager.synchronize_manifest_and_services(arc, mapping)

    assert synced == 2
    assert "META-INF/services/org.io.DataReader" in arc.resources
    assert "META-INF/services/org.io.DataWriter" in arc.resources
    assert b"org.io.DiskReader" in arc.resources["META-INF/services/org.io.DataReader"]
    assert b"org.io.DiskWriter" in arc.resources["META-INF/services/org.io.DataWriter"]


def test_adv_manifest_main_class_sync_slash_and_dot_formats():
    """
    Adversarial 7.3: Manifest Main-Class synchronization conforms to RFC 822 dot-separated format.
    """
    arc = JarArchive()
    arc.manifest = Manifest(main_attributes={"Main-Class": "com.launcher.Boot"})

    mapping = SymbolMapping(classes={"com/launcher/Boot": "org/app/Main"})
    repackager = JarRepackager()
    synced = repackager.synchronize_manifest_and_services(arc, mapping)

    assert synced == 1
    # Manifest RFC 822 sets Main-Class in dot notation
    assert arc.manifest.get_main_class() == "org.app.Main"


# ============================================================================
# 8. CLI Frontend Flag Variations & Adversarial Invocations
# ============================================================================

def test_adv_cli_parser_all_subcommand_flags():
    """
    Adversarial 8.1: Full argument parser validation for all subcommands and optional flags.
    """
    parser = build_parser()

    # analyze
    args_an = parser.parse_args(["analyze", "target.jar"])
    assert args_an.command == "analyze"
    assert args_an.jar == "target.jar"

    # normalize
    args_norm = parser.parse_args(["normalize", "in.jar", "-o", "out.jar"])
    assert args_norm.command == "normalize"
    assert args_norm.jar == "in.jar"
    assert args_norm.output == "out.jar"

    # slice
    args_sl_c = parser.parse_args(["slice", "in.jar", "-c", "com/foo/Bar", "-f", "json"])
    assert args_sl_c.command == "slice"
    assert args_sl_c.target_class == "com/foo/Bar"
    assert args_sl_c.format == "json"

    args_sl_s = parser.parse_args(["slice", "in.jar", "-s", "Cipher;->doFinal"])
    assert args_sl_s.command == "slice"
    assert args_sl_s.sink == "Cipher;->doFinal"

    # rename
    args_rn = parser.parse_args(["rename", "in.jar", "-m", "map.json", "-o", "renamed.jar"])
    assert args_rn.command == "rename"
    assert args_rn.mapping == "map.json"
    assert args_rn.output == "renamed.jar"


def test_adv_cli_analyze_adversarial_paths(control_flow_flattened_jar: Path, tmp_path: Path, capsys):
    """
    Adversarial 8.2: cmd_analyze execution:
    - Missing input JAR returns exit code 1.
    - Valid input JAR returns exit code 0 and prints distribution and classes.
    """
    # Nonexistent file
    args_bad = argparse.Namespace(jar=str(tmp_path / "nonexistent.jar"))
    code_bad = cmd_analyze(args_bad)
    assert code_bad == 1
    err = capsys.readouterr().err
    assert "not found" in err

    # Valid synthetic fixture
    args_good = argparse.Namespace(jar=str(control_flow_flattened_jar))
    code_good = cmd_analyze(args_good)
    assert code_good == 0
    out = capsys.readouterr().out
    assert "Total classes:" in out
    assert "Architectural Domain Distribution" in out


def test_adv_cli_normalize_custom_output(control_flow_flattened_jar: Path, tmp_path: Path):
    """
    Adversarial 8.3: cmd_normalize writes clean archive to explicit custom output path.
    """
    out_jar = tmp_path / "normalized_custom.jar"
    args = argparse.Namespace(jar=str(control_flow_flattened_jar), output=str(out_jar))
    ret = cmd_normalize(args)
    assert ret == 0
    assert out_jar.is_file()
    assert out_jar.stat().st_size > 0

    # Ensure saved jar can be read back cleanly
    reloaded = JarArchive.from_file(out_jar)
    assert len(reloaded.classes) >= 2


def test_adv_cli_slice_format_and_sink_variations(constant_obfuscated_jar: Path, capsys):
    """
    Adversarial 8.4: cmd_slice variations:
    - Target class JSON format prints valid JSON.
    - Target class Markdown format prints markdown headers.
    - Sink slice returns valid JSON data.
    - Neither target nor sink returns exit code 1.
    """
    # Target class JSON
    args_json = argparse.Namespace(
        jar=str(constant_obfuscated_jar),
        target_class="com/synthetic/constants/ArithmeticObfuscated",
        sink=None,
        format="json"
    )
    assert cmd_slice(args_json) == 0
    out_json = capsys.readouterr().out
    parsed_json = json.loads(out_json)
    assert parsed_json["class"] == "com/synthetic/constants/ArithmeticObfuscated"

    # Target class Markdown
    args_md = argparse.Namespace(
        jar=str(constant_obfuscated_jar),
        target_class="com/synthetic/constants/ArithmeticObfuscated",
        sink=None,
        format="markdown"
    )
    assert cmd_slice(args_md) == 0
    out_md = capsys.readouterr().out
    assert "# Class: `com/synthetic/constants/ArithmeticObfuscated`" in out_md

    # Sink slice
    args_sink = argparse.Namespace(
        jar=str(constant_obfuscated_jar),
        target_class=None,
        sink="decryptString",
        format="json"
    )
    assert cmd_slice(args_sink) == 0
    out_sink = capsys.readouterr().out
    sink_data = json.loads(out_sink)
    assert "sink_pattern" in sink_data

    # Neither specified -> error
    args_none = argparse.Namespace(
        jar=str(constant_obfuscated_jar),
        target_class=None,
        sink=None,
        format="json"
    )
    assert cmd_slice(args_none) == 1


def test_adv_cli_rename_missing_mapping_and_execution(stripped_symbols_jar: Path, tmp_path: Path):
    """
    Adversarial 8.5: cmd_rename execution:
    - Missing mapping file returns exit code 1.
    - Valid mapping executes and outputs refactored archive.
    """
    out_jar = tmp_path / "renamed_cli.jar"

    # Missing mapping
    args_no_map = argparse.Namespace(
        jar=str(stripped_symbols_jar),
        mapping=str(tmp_path / "missing_map.json"),
        output=str(out_jar)
    )
    assert cmd_rename(args_no_map) == 1

    # Valid mapping
    map_file = tmp_path / "valid_map.json"
    map_file.write_text(json.dumps({
        "classes": {"a/a": "com/app/Root"},
        "methods": {},
        "fields": {}
    }), encoding="utf-8")

    args_valid = argparse.Namespace(
        jar=str(stripped_symbols_jar),
        mapping=str(map_file),
        output=str(out_jar)
    )
    assert cmd_rename(args_valid) == 0
    assert out_jar.is_file()

    reloaded = JarArchive.from_file(out_jar)
    assert "com/app/Root" in reloaded.classes


# ============================================================================
# 9. Architectural Triage & Contextual Slicing Stress Tests
# ============================================================================

def test_adv_triage_multi_domain_hybrid_class():
    """
    Adversarial 9.1: Multi-domain hybrid class exercising Cryptography and Network APIs simultaneously.
    Verifies primary domain selected by score, confidence mathematically computed,
    and security sensitivity flag set to True.
    """
    arc = JarArchive()
    cf = make_adv_class("hybrid/SecNetClient")
    cp = cf.constant_pool

    # Inject Cryptography references
    cp.add_member_ref(ConstantTag.METHODREF, "javax/crypto/Cipher", "getInstance", "(Ljava/lang/String;)Ljavax/crypto/Cipher;")
    cp.add_member_ref(ConstantTag.METHODREF, "javax/crypto/Cipher", "doFinal", "([B)[B")
    cp.add_member_ref(ConstantTag.METHODREF, "java/security/MessageDigest", "digest", "()[B")

    # Inject Network references (fewer than crypto)
    cp.add_member_ref(ConstantTag.METHODREF, "java/net/Socket", "connect", "(Ljava/net/SocketAddress;)V")

    arc.add_class(cf)

    triager = CapabilityTriager()
    res = triager.triage_class("hybrid/SecNetClient", arc)

    assert res.primary_domain == CapabilityDomain.CRYPTOGRAPHY
    assert res.is_security_sensitive is True
    assert 0.0 < res.confidence <= 1.0
    assert CapabilityDomain.NETWORK in res.domain_scores
    assert res.domain_scores[CapabilityDomain.CRYPTOGRAPHY] > res.domain_scores[CapabilityDomain.NETWORK]


def test_adv_contextual_slicer_string_literal_cap():
    """
    Adversarial 9.2: Constant pool containing 35 string literals.
    ContextualSlicer._extract_string_literals must cap strings at 20 items to prevent LLM prompt blowup.
    """
    arc = JarArchive()
    cf = make_adv_class("strings/HeavyStrings")
    cp = cf.constant_pool
    for i in range(35):
        cp.add_string(f"literal_token_string_{i}")
    arc.add_class(cf)

    slicer = ContextualSlicer()
    strings = slicer._extract_string_literals(cf)
    assert len(strings) == 20


def test_adv_contextual_slicer_nonexistent_class():
    """
    Adversarial 9.3: ContextualSlicer on nonexistent class returns error dictionary.
    """
    arc = JarArchive()
    slicer = ContextualSlicer()
    data = slicer.generate_class_summary_dict("missing/Class", arc)
    assert "error" in data


def test_adv_contextual_slicer_sink_slice_unmatched_pattern():
    """
    Adversarial 9.4: Sink slice on pattern matching no callees in call graph returns empty candidates cleanly.
    """
    arc = JarArchive()
    cf = make_adv_class("sink/Demo")
    arc.add_class(cf)
    cg = CallGraph.build(arc)

    slicer = ContextualSlicer()
    sink_data = slicer.generate_sink_slice("UnmatchedSinkPattern", arc, cg)
    assert sink_data["sink_pattern"] == "UnmatchedSinkPattern"
    assert sink_data["direct_callers"] == []
    assert sink_data["entrypoint_candidates"] == []


# ============================================================================
# 10. Symbol Mapping Validation Edge Cases
# ============================================================================

@pytest.mark.parametrize("invalid_id", [
    "123startsWithDigit",
    "has-hyphen",
    "has space",
    "has.dot",
    "has/slash",
    ""
])
def test_adv_symbol_mapping_validation_invalid_identifiers(invalid_id: str):
    """
    Adversarial 10.1: SymbolMapping rejects invalid field and method identifiers.
    """
    mapping = SymbolMapping(
        fields={"com/test/Foo#fld": invalid_id},
        methods={"com/test/Foo#mth#()V": invalid_id}
    )
    errs = mapping.validate()
    assert len(errs) >= 1

    with pytest.raises(MappingValidationError):
        SymbolMapping.from_dict({"fields": {"com/test/Foo#fld": invalid_id}})


@pytest.mark.parametrize("invalid_class", [
    "123DigitClass",
    "/leadingSlash",
    "trailingSlash/",
    "double//slash",
    "dotted.package.Class",
    ""
])
def test_adv_symbol_mapping_validation_invalid_class_names(invalid_class: str):
    """
    Adversarial 10.2: SymbolMapping rejects invalid JVM internal class names.
    """
    mapping = SymbolMapping(classes={"com/test/Old": invalid_class})
    errs = mapping.validate()
    assert len(errs) >= 1

    with pytest.raises(MappingValidationError):
        SymbolMapping.from_dict({"classes": {"com/test/Old": invalid_class}})


@pytest.mark.parametrize("valid_name", [
    "$validDollar",
    "_validUnderscore",
    "Valid$Inner$123",
    "_$both$_"
])
def test_adv_symbol_mapping_valid_special_characters(valid_name: str):
    """
    Adversarial 10.3: Java allows $ and _ in identifiers and internal class names.
    """
    mapping = SymbolMapping(
        classes={"com/test/Old": f"pkg/{valid_name}"},
        fields={"com/test/Old#f": valid_name},
        methods={"com/test/Old#m#()V": valid_name}
    )
    assert mapping.validate() == []
