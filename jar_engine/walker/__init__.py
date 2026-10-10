"""
jar_engine.walker: Whole-archive static analysis, hierarchy DAG, call graph, and dependency slicing.
"""

from .hierarchy import (
    ClassOrigin, HierarchyNode, JDK_CORE_HIERARCHY, InheritanceHierarchy
)
from .call_graph import (
    MethodReference, FieldReference, CallEdge, FieldAccessRecord,
    InstructionScanner, CallGraph
)
from .dependency_slice import (
    DependencyKind, ClassDependencyGraph, DependencySlicer,
    extract_classes_from_descriptor
)

__all__ = [
    "ClassOrigin", "HierarchyNode", "JDK_CORE_HIERARCHY", "InheritanceHierarchy",
    "MethodReference", "FieldReference", "CallEdge", "FieldAccessRecord",
    "InstructionScanner", "CallGraph",
    "DependencyKind", "ClassDependencyGraph", "DependencySlicer",
    "extract_classes_from_descriptor",
]
