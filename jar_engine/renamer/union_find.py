"""
jar_engine.renamer.union_find: Disjoint-Set Union-Find for method override equivalence and JDK pinning (Features F25, F26).

Ensures polymorphic methods across inheritance hierarchies and interface implementations
are renamed as a single atomic unit, preventing AbstractMethodError and linkage failures,
while pinning JDK standard library methods (java.*, javax.*) against illegal renames.
"""

from __future__ import annotations
from typing import Dict, Set, List, Optional, Tuple
from jar_engine.model.jar_archive import JarArchive
from jar_engine.walker.hierarchy import InheritanceHierarchy


OBJECT_DECLARED_METHODS: Set[str] = {
    "<init>", "<clinit>", "equals", "hashCode", "toString",
    "clone", "finalize", "getClass", "notify", "notifyAll", "wait"
}

COMMON_JDK_INTERFACE_METHODS: Dict[str, Set[str]] = {
    "java/lang/Runnable": {"run"},
    "java/lang/Comparable": {"compareTo"},
    "java/lang/AutoCloseable": {"close"},
    "java/io/Closeable": {"close"},
    "java/util/concurrent/Callable": {"call"},
    "java/util/Iterator": {"hasNext", "next", "remove"},
    "java/util/function/Function": {"apply"},
    "java/util/function/Consumer": {"accept"},
    "java/util/function/Supplier": {"get"},
    "java/util/function/Predicate": {"test"},
}


class MethodEquivalenceCluster:
    """Disjoint set data structure grouping overridden/implemented method signatures."""

    def __init__(self, hierarchy: InheritanceHierarchy):
        self.hierarchy = hierarchy
        self.parent: Dict[str, str] = {}
        self.pinned: Set[str] = set()  # Methods that cannot be renamed (JDK / external)

    def _find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
            return item
        if self.parent[item] != item:
            self.parent[item] = self._find(self.parent[item])
        return self.parent[item]

    def _union(self, item1: str, item2: str) -> None:
        root1 = self._find(item1)
        root2 = self._find(item2)
        if root1 != root2:
            self.parent[root2] = root1
            if root2 in self.pinned or root1 in self.pinned:
                self.pinned.add(root1)
                self.pinned.add(root2)

    def is_jdk_method(self, class_name: str, method_name: str) -> bool:
        """Checks if a method belongs to immutable JDK or standard runtime namespaces."""
        if method_name in OBJECT_DECLARED_METHODS or method_name == "main":
            return True
        if class_name == "java/lang/Object":
            return method_name in OBJECT_DECLARED_METHODS

        for jdk_class, methods in COMMON_JDK_INTERFACE_METHODS.items():
            if class_name == jdk_class and method_name in methods:
                return True

        if class_name.startswith(("java/", "javax/", "sun/", "jdk/")):
            # If external JDK class is supertype, check if it's Object
            if class_name == "java/lang/Object":
                return method_name in OBJECT_DECLARED_METHODS
            return True

        return False

    def build_from_archive(self, archive: JarArchive) -> None:
        """
        Builds method equivalence classes across all classes in the archive and their supertypes.
        """
        # Step 1: Register all methods and pin JDK / standard methods
        for c_name, cf in archive.classes.items():
            cp = cf.constant_pool
            for m in cf.methods:
                m_name = m.name(cp)
                m_desc = m.descriptor(cp)
                m_key = f"{c_name}#{m_name}#{m_desc}"
                root = self._find(m_key)
                if self.is_jdk_method(c_name, m_name):
                    self.pinned.add(root)

        # Step 2: Connect overrides across inheritance and interfaces
        for c_name, cf in archive.classes.items():
            cp = cf.constant_pool
            # Supertypes = superclasses + all transitive interfaces
            superclasses = self.hierarchy.get_superclasses(c_name)
            interfaces = self.hierarchy.get_interfaces(c_name)
            all_supers = list(superclasses) + list(interfaces)

            for m in cf.methods:
                m_name = m.name(cp)
                m_desc = m.descriptor(cp)
                if m.is_static or m_name in ("<init>", "<clinit>"):
                    continue

                child_key = f"{c_name}#{m_name}#{m_desc}"

                for s_name in all_supers:
                    s_cf = archive.get_class(s_name)
                    if s_cf:
                        s_m = s_cf.get_method(m_name, m_desc)
                        if s_m and not s_m.is_static:
                            parent_key = f"{s_name}#{m_name}#{m_desc}"
                            self._union(child_key, parent_key)
                    elif self.is_jdk_method(s_name, m_name):
                        # Superclass is outside the archive (e.g. java/lang/Object)
                        parent_key = f"{s_name}#{m_name}#{m_desc}"
                        self._union(child_key, parent_key)
                        self.pinned.add(self._find(child_key))

    def is_method_pinned(self, method_key: str) -> bool:
        """Returns True if this method (or any method in its override cluster) is pinned."""
        root = self._find(method_key)
        return root in self.pinned

    def get_cluster_members(self, method_key: str, all_keys: Set[str]) -> List[str]:
        """Returns all method keys belonging to the same equivalence class."""
        target_root = self._find(method_key)
        return [k for k in all_keys if self._find(k) == target_root]
