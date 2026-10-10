"""
jar_engine.walker.dependency_slice: Inter-class dependency slicing, Tarjan SCCs, and condensation DAG.
"""

from __future__ import annotations
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Optional, Tuple, Iterable, Union, Any, TYPE_CHECKING
import networkx as nx

from jar_engine.model.class_file import ClassFile
from jar_engine.model.member_info import ExceptionsAttribute, CodeAttribute
from jar_engine.model.cp_info import CONSTANT_Class_info

if TYPE_CHECKING:
    from jar_engine.model.jar_archive import JarArchive
    from jar_engine.walker.hierarchy import InheritanceHierarchy
    from jar_engine.walker.call_graph import CallGraph


class DependencyKind(str, Enum):
    INHERITANCE = "inheritance"
    METHOD_CALL = "method_call"
    FIELD_ACCESS = "field_access"
    TYPE_DESCRIPTOR = "type_descriptor"
    EXCEPTION = "exception"
    CONSTANT_POOL = "constant_pool"
    INNER_CLASS = "inner_class"


# Regex to extract all internal class names from JVM descriptors (e.g. Ljava/lang/String;)
_DESCRIPTOR_CLASS_PATTERN = re.compile(r"L([a-zA-Z0-9_$]+(?:/[a-zA-Z0-9_$]+)*);")


def extract_classes_from_descriptor(descriptor: str) -> List[str]:
    """Extracts internal class names from field or method descriptor."""
    return _DESCRIPTOR_CLASS_PATTERN.findall(descriptor)


@dataclass
class ClassDependencyGraph:
    """Directed graph where edge (C1, C2) means class C1 depends on / references C2."""
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)

    def add_dependency(self, src: str, dst: str, kind: Union[DependencyKind, str], detail: Optional[str] = None) -> None:
        """Adds a dependency edge from src to dst."""
        if not src or not dst or src == dst:
            return

        kind_str = kind.value if isinstance(kind, DependencyKind) else str(kind)
        if not self.graph.has_node(src):
            self.graph.add_node(src)
        if not self.graph.has_node(dst):
            self.graph.add_node(dst)

        if self.graph.has_edge(src, dst):
            data = self.graph.edges[src, dst]
            data["kinds"].add(kind_str)
            data["count"] += 1
            if detail and len(data["details"]) < 10:
                data["details"].append(detail)
        else:
            self.graph.add_edge(
                src,
                dst,
                kinds={kind_str},
                count=1,
                details=[detail] if detail else []
            )

    @classmethod
    def build(
        cls,
        archive: "JarArchive",
        hierarchy: Optional["InheritanceHierarchy"] = None,
        call_graph: Optional["CallGraph"] = None,
        include_external: bool = True
    ) -> "ClassDependencyGraph":
        cdg = cls()
        if archive is None:
            return cdg

        archive_classes = set(archive.classes.keys())

        # 1. Register all archive classes as nodes
        for cls_name in archive_classes:
            cdg.graph.add_node(cls_name)

        # 2. Inheritance Dependencies (superclasses & interfaces)
        for cls_name, cf in archive.classes.items():
            if cf.super_class_name:
                super_n = cf.super_class_name
                if include_external or super_n in archive_classes:
                    cdg.add_dependency(cls_name, super_n, DependencyKind.INHERITANCE, f"extends {super_n}")

            for iface in cf.interface_names:
                if include_external or iface in archive_classes:
                    cdg.add_dependency(cls_name, iface, DependencyKind.INHERITANCE, f"implements {iface}")

        # 3. Method Call Dependencies
        if call_graph is not None:
            for edge in call_graph.call_edges:
                from jar_engine.walker.call_graph import MethodReference
                try:
                    src_cls = MethodReference.from_id(edge.caller).class_name
                    dst_cls = MethodReference.from_id(edge.callee).class_name
                    if include_external or dst_cls in archive_classes:
                        cdg.add_dependency(src_cls, dst_cls, DependencyKind.METHOD_CALL, f"call {edge.callee}")
                except Exception:
                    pass

        # 4. Field Access Dependencies
        if call_graph is not None:
            for facc in call_graph.field_accesses:
                from jar_engine.walker.call_graph import MethodReference, FieldReference
                try:
                    src_cls = MethodReference.from_id(facc.caller).class_name
                    dst_cls = FieldReference.from_id(facc.field).class_name
                    if include_external or dst_cls in archive_classes:
                        cdg.add_dependency(src_cls, dst_cls, DependencyKind.FIELD_ACCESS, f"{facc.access_type} {facc.field}")
                except Exception:
                    pass

        # 5. Type Descriptor Dependencies (fields, method signatures)
        for cls_name, cf in archive.classes.items():
            # Field descriptors
            for f in cf.fields:
                f_desc = f.descriptor(cf.constant_pool)
                for ref_cls in extract_classes_from_descriptor(f_desc):
                    if include_external or ref_cls in archive_classes:
                        cdg.add_dependency(cls_name, ref_cls, DependencyKind.TYPE_DESCRIPTOR, f"field type {f.name(cf.constant_pool)}")

            # Method descriptors
            for m in cf.methods:
                m_desc = m.descriptor(cf.constant_pool)
                for ref_cls in extract_classes_from_descriptor(m_desc):
                    if include_external or ref_cls in archive_classes:
                        cdg.add_dependency(cls_name, ref_cls, DependencyKind.TYPE_DESCRIPTOR, f"method signature {m.name(cf.constant_pool)}")

        # 6. Exception Dependencies
        for cls_name, cf in archive.classes.items():
            for m in cf.methods:
                # Checked exceptions in attribute
                ex_attr = m.get_attribute("Exceptions", cf.constant_pool)
                if isinstance(ex_attr, ExceptionsAttribute):
                    for ex_name in ex_attr.exception_names(cf.constant_pool):
                        if include_external or ex_name in archive_classes:
                            cdg.add_dependency(cls_name, ex_name, DependencyKind.EXCEPTION, "throws")

                # Catch blocks in Code attribute
                code_attr = m.code_attribute(cf.constant_pool)
                if code_attr:
                    for entry in code_attr.exception_table:
                        if entry.catch_type != 0:
                            try:
                                catch_cls = cf.resolve_class_name(entry.catch_type)
                                if include_external or catch_cls in archive_classes:
                                    cdg.add_dependency(cls_name, catch_cls, DependencyKind.EXCEPTION, "catch")
                            except Exception:
                                pass

        # 7. Constant Pool Class References (e.g. checkcast, instanceof, new)
        for cls_name, cf in archive.classes.items():
            for _, entry in cf.constant_pool.entries():
                if isinstance(entry, CONSTANT_Class_info):
                    try:
                        ref_cls = cf.resolve_utf8(entry.name_index)
                        if ref_cls.startswith("["):
                            # Array class: extract element class
                            elems = extract_classes_from_descriptor(ref_cls)
                            for e_cls in elems:
                                if include_external or e_cls in archive_classes:
                                    cdg.add_dependency(cls_name, e_cls, DependencyKind.CONSTANT_POOL, "cp array class")
                        elif include_external or ref_cls in archive_classes:
                            cdg.add_dependency(cls_name, ref_cls, DependencyKind.CONSTANT_POOL, "cp class")
                    except Exception:
                        pass

        return cdg

    def forward_slice(self, roots: Union[str, Iterable[str]]) -> Set[str]:
        """
        Computes forward dependency slice: all classes reachable from roots (transitive dependencies).
        """
        if isinstance(roots, str):
            roots = [roots]
        result: Set[str] = set()
        for r in roots:
            norm_r = r.strip().replace(".", "/")
            if norm_r.startswith("L") and norm_r.endswith(";"):
                norm_r = norm_r[1:-1]
            result.add(norm_r)
            if norm_r in self.graph:
                result.update(nx.descendants(self.graph, norm_r))
        return result

    def backward_slice(self, targets: Union[str, Iterable[str]]) -> Set[str]:
        """
        Computes backward dependency slice: all classes that depend on targets (transitive dependents).
        """
        if isinstance(targets, str):
            targets = [targets]
        result: Set[str] = set()
        for t in targets:
            norm_t = t.strip().replace(".", "/")
            if norm_t.startswith("L") and norm_t.endswith(";"):
                norm_t = norm_t[1:-1]
            result.add(norm_t)
            if norm_t in self.graph:
                result.update(nx.ancestors(self.graph, norm_t))
        return result

    def get_sccs(self) -> List[Set[str]]:
        """Returns all strongly connected components evaluated via Tarjan's algorithm."""
        return [set(scc) for scc in nx.strongly_connected_components(self.graph)]

    def get_cyclic_components(self) -> List[Set[str]]:
        """Returns components with > 1 classes or self-loops."""
        cycles = []
        for scc in self.get_sccs():
            if len(scc) > 1:
                cycles.append(scc)
            elif len(scc) == 1:
                node = next(iter(scc))
                if self.graph.has_edge(node, node):
                    cycles.append(scc)
        return cycles

    def has_cycles(self) -> bool:
        """Returns True if the dependency graph contains any cycles."""
        return not nx.is_directed_acyclic_graph(self.graph)

    def condensation_graph(self) -> nx.DiGraph:
        """Returns the condensation quotient DAG contracting each SCC into a single super-node."""
        return nx.condensation(self.graph)

    def topological_sort(self, direction: str = "build") -> List[Set[str]]:
        """
        Topological order of condensation components.
        direction="build": independent prerequisites first (compilation / decompilation order).
        direction="dependency": root dependents first.
        """
        cond = self.condensation_graph()
        order = list(nx.topological_sort(cond))
        if direction.lower() in ("build", "compile", "prerequisite"):
            order.reverse()
        return [set(cond.nodes[node_id]["members"]) for node_id in order]


class DependencySlicer:
    """Facade helper for inter-class dependency slicing."""

    @staticmethod
    def slice(
        archive: "JarArchive",
        roots: Union[str, Iterable[str]],
        direction: str = "forward",
        hierarchy: Optional["InheritanceHierarchy"] = None,
        call_graph: Optional["CallGraph"] = None,
        include_external: bool = False
    ) -> Set[str]:
        cdg = ClassDependencyGraph.build(
            archive,
            hierarchy=hierarchy,
            call_graph=call_graph,
            include_external=include_external
        )
        d = direction.lower()
        if d in ("forward", "downstream", "dependencies"):
            return cdg.forward_slice(roots)
        elif d in ("backward", "upstream", "dependents"):
            return cdg.backward_slice(roots)
        else:
            raise ValueError(f"Unknown direction '{direction}'; must be 'forward' or 'backward'")
