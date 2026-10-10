"""
jar_engine.walker.hierarchy: Class and interface inheritance DAG, JDK stubs, and LCA resolution.
Conforming to JVMS §4.10.1.2 and standard JVM subtyping rules.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union, Any, TYPE_CHECKING
import networkx as nx

from jar_engine.model.constants import AccessFlags
from jar_engine.model.class_file import ClassFile

if TYPE_CHECKING:
    from jar_engine.model.jar_archive import JarArchive


class ClassOrigin(Enum):
    ARCHIVE = "archive"          # Defined in target JAR
    JDK_KNOWN = "jdk_known"      # Pre-populated JDK core class/interface
    EXTERNAL_STUB = "external"   # External library or unknown class


# Pre-populated JDK core class hierarchy knowledge base
# Format: class_name -> (super_name, (interfaces,), is_interface)
JDK_CORE_HIERARCHY: Dict[str, Tuple[Optional[str], Tuple[str, ...], bool]] = {
    # java.lang fundamentals
    "java/lang/Object": (None, (), False),
    "java/lang/Cloneable": ("java/lang/Object", (), True),
    "java/io/Serializable": ("java/lang/Object", (), True),
    "java/lang/Comparable": ("java/lang/Object", (), True),
    "java/lang/CharSequence": ("java/lang/Object", (), True),
    "java/lang/AutoCloseable": ("java/lang/Object", (), True),
    "java/io/Closeable": ("java/lang/Object", ("java/lang/AutoCloseable",), True),
    "java/io/Flushable": ("java/lang/Object", (), True),
    "java/lang/Readable": ("java/lang/Object", (), True),
    "java/lang/Appendable": ("java/lang/Object", (), True),
    "java/lang/Iterable": ("java/lang/Object", (), True),
    "java/util/Iterator": ("java/lang/Object", (), True),
    "java/util/Comparator": ("java/lang/Object", (), True),
    "java/util/Collection": ("java/lang/Object", ("java/lang/Iterable",), True),
    "java/util/List": ("java/lang/Object", ("java/util/Collection",), True),
    "java/util/Set": ("java/lang/Object", ("java/util/Collection",), True),
    "java/util/Queue": ("java/lang/Object", ("java/util/Collection",), True),
    "java/util/Deque": ("java/lang/Object", ("java/util/Queue",), True),
    "java/util/Map": ("java/lang/Object", (), True),
    "java/util/Map$Entry": ("java/lang/Object", (), True),
    "java/lang/Runnable": ("java/lang/Object", (), True),
    "java/util/concurrent/Callable": ("java/lang/Object", (), True),

    # Exceptions & Errors
    "java/lang/Throwable": ("java/lang/Object", ("java/io/Serializable",), False),
    "java/lang/Exception": ("java/lang/Throwable", (), False),
    "java/lang/RuntimeException": ("java/lang/Exception", (), False),
    "java/lang/Error": ("java/lang/Throwable", (), False),
    "java/io/IOException": ("java/lang/Exception", (), False),
    "java/lang/NullPointerException": ("java/lang/RuntimeException", (), False),
    "java/lang/IllegalArgumentException": ("java/lang/RuntimeException", (), False),
    "java/lang/IllegalStateException": ("java/lang/RuntimeException", (), False),
    "java/lang/IndexOutOfBoundsException": ("java/lang/RuntimeException", (), False),
    "java/lang/ArrayIndexOutOfBoundsException": ("java/lang/IndexOutOfBoundsException", (), False),
    "java/lang/StringIndexOutOfBoundsException": ("java/lang/IndexOutOfBoundsException", (), False),
    "java/lang/ClassCastException": ("java/lang/RuntimeException", (), False),
    "java/lang/UnsupportedOperationException": ("java/lang/RuntimeException", (), False),
    "java/lang/ArithmeticException": ("java/lang/RuntimeException", (), False),
    "java/lang/SecurityException": ("java/lang/RuntimeException", (), False),

    # Number and Primitive Wrappers
    "java/lang/Number": ("java/lang/Object", ("java/io/Serializable",), False),
    "java/lang/Integer": ("java/lang/Number", ("java/lang/Comparable",), False),
    "java/lang/Long": ("java/lang/Number", ("java/lang/Comparable",), False),
    "java/lang/Double": ("java/lang/Number", ("java/lang/Comparable",), False),
    "java/lang/Float": ("java/lang/Number", ("java/lang/Comparable",), False),
    "java/lang/Short": ("java/lang/Number", ("java/lang/Comparable",), False),
    "java/lang/Byte": ("java/lang/Number", ("java/lang/Comparable",), False),
    "java/lang/Boolean": ("java/lang/Object", ("java/io/Serializable", "java/lang/Comparable"), False),
    "java/lang/Character": ("java/lang/Object", ("java/io/Serializable", "java/lang/Comparable"), False),

    # Strings & Common Classes
    "java/lang/String": ("java/lang/Object", ("java/io/Serializable", "java/lang/Comparable", "java/lang/CharSequence"), False),
    "java/lang/StringBuilder": ("java/lang/Object", ("java/io/Serializable", "java/lang/CharSequence"), False),
    "java/lang/StringBuffer": ("java/lang/Object", ("java/io/Serializable", "java/lang/CharSequence"), False),
    "java/lang/Enum": ("java/lang/Object", ("java/lang/Comparable", "java/io/Serializable"), False),
    "java/lang/Record": ("java/lang/Object", (), False),
    "java/lang/Thread": ("java/lang/Object", ("java/lang/Runnable",), False),
    "java/lang/ClassLoader": ("java/lang/Object", (), False),
    "java/lang/Class": ("java/lang/Object", ("java/io/Serializable",), False),

    # IO Streams
    "java/io/InputStream": ("java/lang/Object", ("java/io/Closeable",), False),
    "java/io/OutputStream": ("java/io/Closeable", ("java/io/Flushable",), False),
    "java/io/Reader": ("java/lang/Object", ("java/lang/Readable", "java/io/Closeable"), False),
    "java/io/Writer": ("java/lang/Object", ("java/lang/Appendable", "java/io/Closeable", "java/io/Flushable"), False),
}


@dataclass
class HierarchyNode:
    """Represents a node in the class/interface inheritance graph."""
    name: str                     # Internal binary name, e.g. "java/lang/String"
    super_name: Optional[str]     # Superclass name (None for java/lang/Object)
    interfaces: Tuple[str, ...]   # Direct interface names
    access_flags: int             # JVM access flags
    origin: ClassOrigin           # Source origin
    class_file: Optional[ClassFile] = None

    @property
    def is_interface(self) -> bool:
        return bool(self.access_flags & AccessFlags.INTERFACE)

    @property
    def is_abstract(self) -> bool:
        return bool(self.access_flags & AccessFlags.ABSTRACT)

    @property
    def is_external(self) -> bool:
        return self.origin != ClassOrigin.ARCHIVE


class InheritanceHierarchy:
    """Directed graph of class inheritance and interface implementation across an archive."""

    def __init__(self):
        self.nodes: Dict[str, HierarchyNode] = {}
        # Backing NetworkX DiGraph: edge (Child, Parent) with relation in {"extends", "implements"}
        self.graph: nx.DiGraph = nx.DiGraph()
        self._super_class: Dict[str, Optional[str]] = {}
        self._interfaces: Dict[str, List[str]] = {}
        self._sub_classes: Dict[str, Set[str]] = {}
        self._implementors: Dict[str, Set[str]] = {}

    @staticmethod
    def _normalize_class_name(name: str) -> str:
        """Normalizes class name into standard internal form ('java/lang/String')."""
        s = name.strip()
        if s.startswith("L") and s.endswith(";"):
            s = s[1:-1]
        return s.replace(".", "/")

    @classmethod
    def from_archive(cls, archive: "JarArchive", include_jdk_stubs: bool = True) -> "InheritanceHierarchy":
        return cls.from_classes(archive.classes.values(), include_jdk_stubs=include_jdk_stubs)

    @classmethod
    def from_classes(cls, classes: Iterable[ClassFile], include_jdk_stubs: bool = True) -> "InheritanceHierarchy":
        hierarchy = cls()
        class_list = list(classes)

        # 1. Register archive classes
        for cf in class_list:
            c_name = cls._normalize_class_name(cf.this_class_name)
            s_name = cls._normalize_class_name(cf.super_class_name) if cf.super_class_name else None
            ifaces = tuple(cls._normalize_class_name(i) for i in cf.interface_names)

            node = HierarchyNode(
                name=c_name,
                super_name=s_name,
                interfaces=ifaces,
                access_flags=cf.access_flags,
                origin=ClassOrigin.ARCHIVE,
                class_file=cf
            )
            hierarchy.nodes[c_name] = node
            hierarchy.graph.add_node(c_name, node=node)
            hierarchy._super_class[c_name] = s_name
            hierarchy._interfaces[c_name] = list(ifaces)
            hierarchy._sub_classes.setdefault(c_name, set())
            hierarchy._implementors.setdefault(c_name, set())

        # 2. Add JDK stubs and external references
        referenced: Set[str] = set()
        for node in list(hierarchy.nodes.values()):
            if node.super_name:
                referenced.add(node.super_name)
            for iface in node.interfaces:
                referenced.add(iface)

        while referenced:
            target = referenced.pop()
            if target in hierarchy.nodes:
                continue

            # Determine origin and metadata
            if include_jdk_stubs and target in JDK_CORE_HIERARCHY:
                super_n, ifaces_n, is_iface = JDK_CORE_HIERARCHY[target]
                flags = AccessFlags.PUBLIC
                if is_iface:
                    flags |= AccessFlags.INTERFACE | AccessFlags.ABSTRACT
                node = HierarchyNode(
                    name=target,
                    super_name=super_n,
                    interfaces=ifaces_n,
                    access_flags=flags,
                    origin=ClassOrigin.JDK_KNOWN,
                    class_file=None
                )
            else:
                # External stub fallback: assumes public class extending java/lang/Object
                super_n = "java/lang/Object" if target != "java/lang/Object" else None
                node = HierarchyNode(
                    name=target,
                    super_name=super_n,
                    interfaces=(),
                    access_flags=AccessFlags.PUBLIC,
                    origin=ClassOrigin.EXTERNAL_STUB,
                    class_file=None
                )

            hierarchy.nodes[target] = node
            hierarchy.graph.add_node(target, node=node)
            hierarchy._super_class[target] = node.super_name
            hierarchy._interfaces[target] = list(node.interfaces)
            hierarchy._sub_classes.setdefault(target, set())
            hierarchy._implementors.setdefault(target, set())

            # Recurse on superclass and interfaces of stub
            if node.super_name and node.super_name not in hierarchy.nodes:
                referenced.add(node.super_name)
            for i in node.interfaces:
                if i not in hierarchy.nodes:
                    referenced.add(i)

        # 3. Build graph edges and populate direct subtype maps
        for c_name, node in hierarchy.nodes.items():
            if node.super_name:
                s_name = node.super_name
                hierarchy.graph.add_edge(c_name, s_name, relation="extends")
                hierarchy._sub_classes.setdefault(s_name, set()).add(c_name)
            for iface in node.interfaces:
                hierarchy.graph.add_edge(c_name, iface, relation="implements")
                hierarchy._implementors.setdefault(iface, set()).add(c_name)

        return hierarchy

    # --- Query Methods ---
    def get_node(self, class_name: str) -> Optional[HierarchyNode]:
        return self.nodes.get(self._normalize_class_name(class_name))

    def get_direct_superclass(self, class_name: str) -> Optional[str]:
        norm = self._normalize_class_name(class_name)
        return self._super_class.get(norm)

    def get_superclasses(self, class_name: str) -> List[str]:
        """Returns ordered chain of superclasses upward to java/lang/Object."""
        norm = self._normalize_class_name(class_name)
        chain: List[str] = []
        curr = self.get_direct_superclass(norm)
        visited: Set[str] = {norm}
        while curr and curr not in visited:
            chain.append(curr)
            visited.add(curr)
            curr = self.get_direct_superclass(curr)
        return chain

    def get_all_superclasses(self, class_name: str) -> Set[str]:
        return set(self.get_superclasses(class_name))

    def get_direct_subclasses(self, class_name: str) -> Set[str]:
        norm = self._normalize_class_name(class_name)
        return set(self._sub_classes.get(norm, set()))

    def get_all_subclasses(self, class_name: str) -> Set[str]:
        norm = self._normalize_class_name(class_name)
        result: Set[str] = set()
        queue = list(self.get_direct_subclasses(norm))
        while queue:
            sub = queue.pop(0)
            if sub not in result:
                result.add(sub)
                queue.extend(self.get_direct_subclasses(sub))
        return result

    def get_subclasses(self, class_name: str) -> Set[str]:
        """
        Returns transitive closure of direct and indirect subclasses.
        Conforms to PROJECT.md § Interface Contracts.
        """
        return self.get_all_subclasses(class_name)

    def get_direct_interfaces(self, class_name: str) -> List[str]:
        norm = self._normalize_class_name(class_name)
        return list(self._interfaces.get(norm, []))

    def get_interfaces(self, class_name: str) -> List[str]:
        """
        Returns directly declared interfaces.
        Conforms to PROJECT.md § Interface Contracts.
        """
        return self.get_direct_interfaces(class_name)

    def get_all_interfaces(self, class_name: str) -> Set[str]:
        """Returns transitive closure of all implemented and extended interfaces."""
        norm = self._normalize_class_name(class_name)
        result: Set[str] = set()
        # Collect direct interfaces and superclasses
        nodes_to_visit = [norm] + self.get_superclasses(norm)
        visited_nodes: Set[str] = set(nodes_to_visit)
        interface_queue: List[str] = []

        for n in nodes_to_visit:
            for iface in self.get_direct_interfaces(n):
                if iface not in result:
                    result.add(iface)
                    interface_queue.append(iface)

        while interface_queue:
            iface = interface_queue.pop(0)
            for super_iface in self.get_direct_interfaces(iface):
                if super_iface not in result:
                    result.add(super_iface)
                    interface_queue.append(super_iface)

        return result

    def get_direct_implementors(self, interface_name: str) -> Set[str]:
        norm = self._normalize_class_name(interface_name)
        return set(self._implementors.get(norm, set()))

    def get_all_implementors(self, interface_name: str) -> Set[str]:
        """Returns all classes and sub-interfaces that directly or transitively implement interface."""
        norm = self._normalize_class_name(interface_name)
        result: Set[str] = set()
        queue = list(self.get_direct_implementors(norm))
        while queue:
            curr = queue.pop(0)
            if curr not in result:
                result.add(curr)
                # Subclasses of implementors also implement it
                queue.extend(self.get_direct_subclasses(curr))
                # Sub-interfaces
                queue.extend(self.get_direct_implementors(curr))
        return result

    def is_interface(self, class_name: str) -> bool:
        node = self.get_node(class_name)
        return node.is_interface if node else False

    def is_class(self, class_name: str) -> bool:
        node = self.get_node(class_name)
        return not node.is_interface if node else True

    def is_external(self, class_name: str) -> bool:
        node = self.get_node(class_name)
        return node.is_external if node else True

    def is_subclass(self, subclass: str, superclass: str) -> bool:
        sub_norm = self._normalize_class_name(subclass)
        super_norm = self._normalize_class_name(superclass)
        return super_norm in self.get_all_superclasses(sub_norm)

    # --- Lowest Common Ancestor (JVMS §4.10.1.2) ---
    def find_lowest_common_ancestor(self, class_a: str, class_b: str) -> str:
        """
        Finds the Lowest Common Ancestor adhering strictly to JVMS §4.10.1.2 StackMapTable rules.
        Crucial JVM Rules:
        - Identity: LCA(T, T) = T.
        - Interfaces merge to java/lang/Object.
        - Reference arrays of equal dimension merge covariantly: [^dim L<element_lca>;
        - Arrays of different dimensions, primitive arrays, or array vs reference merge to java/lang/Object.
        """
        a = self._normalize_class_name(class_a)
        b = self._normalize_class_name(class_b)

        # Rule 1: Identity
        if a == b:
            return a

        # Rule 2: Either is java/lang/Object
        if a == "java/lang/Object" or b == "java/lang/Object":
            return "java/lang/Object"

        # Rule 2b: Array types handling (JVMS §4.10.1.2 / §4.10.2.2)
        is_arr_a = a.startswith("[")
        is_arr_b = b.startswith("[")
        if is_arr_a or is_arr_b:
            # If one is an array and one is a reference type, LCA is java/lang/Object
            if not (is_arr_a and is_arr_b):
                return "java/lang/Object"

            dim_a = len(a) - len(a.lstrip("["))
            elem_a = a[dim_a:]
            dim_b = len(b) - len(b.lstrip("["))
            elem_b = b[dim_b:]

            is_ref_a = elem_a.startswith("L") and elem_a.endswith(";")
            is_ref_b = elem_b.startswith("L") and elem_b.endswith(";")

            # Reference arrays of equal dimension: recursively resolve element LCA
            if is_ref_a and is_ref_b and dim_a == dim_b:
                inner_a = elem_a[1:-1]
                inner_b = elem_b[1:-1]
                inner_lca = self.find_lowest_common_ancestor(inner_a, inner_b)
                return ("[" * dim_a) + "L" + inner_lca + ";"

            # Arrays of different dimensions, primitive arrays, or mixed types merge to Object
            return "java/lang/Object"

        # Rule 3: JVMS §4.10: Interfaces merge to java/lang/Object
        if self.is_interface(a) or self.is_interface(b):
            return "java/lang/Object"

        # Rule 4: Collect superclass chain for a
        chain_a: List[str] = [a] + self.get_superclasses(a)
        if "java/lang/Object" not in chain_a:
            chain_a.append("java/lang/Object")
        set_a: Set[str] = set(chain_a)

        # Rule 5: Walk b superclasses upward, first match in set_a is LCA
        curr_b: Optional[str] = b
        visited_b: Set[str] = set()
        while curr_b and curr_b not in visited_b:
            if curr_b in set_a:
                return curr_b
            visited_b.add(curr_b)
            curr_b = self.get_direct_superclass(curr_b)

        return "java/lang/Object"

    # --- Subtype Testing (Assignability) ---
    def is_assignable_from(self, target: str, source: str) -> bool:
        """
        Tests if target is assignable from source (target.isAssignableFrom(source)).
        Handles classes, interfaces, Object, and multidimensional/primitive arrays.
        """
        t = target.strip()
        s = source.strip()

        # Normalize descriptor notation if present: Lcom/foo/Bar; -> com/foo/Bar
        if t.startswith("L") and t.endswith(";"):
            t = t[1:-1]
        if s.startswith("L") and s.endswith(";"):
            s = s[1:-1]

        t = t.replace(".", "/")
        s = s.replace(".", "/")

        # Identity
        if t == s:
            return True

        # java/lang/Object can hold any reference type (class, interface, array)
        if t == "java/lang/Object":
            return True

        # Array handling
        if s.startswith("["):
            # Any array is assignable to Cloneable and Serializable
            if t in ("java/lang/Cloneable", "java/io/Serializable"):
                return True
            if not t.startswith("["):
                return False

            # Both are arrays
            s_dim = len(s) - len(s.lstrip("["))
            t_dim = len(t) - len(t.lstrip("["))
            s_elem = s[s_dim:]
            t_elem = t[t_dim:]

            if s_dim != t_dim:
                # Different dimensions: e.g. [[String to [Object, [Cloneable, [Serializable
                t_elem_clean = t_elem[1:-1] if t_elem.startswith("L") and t_elem.endswith(";") else t_elem
                if s_dim > t_dim and t_elem_clean in ("java/lang/Object", "java/lang/Cloneable", "java/io/Serializable"):
                    return True
                return False

            # Same dimensions
            # Primitive arrays are invariant
            is_s_prim = len(s_elem) == 1 and s_elem in "BCDFIJSZ"
            is_t_prim = len(t_elem) == 1 and t_elem in "BCDFIJSZ"
            if is_s_prim or is_t_prim:
                return s_elem == t_elem

            # Reference arrays are covariant: [T assignable to [U iff T assignable to U
            s_type = s_elem[1:-1] if s_elem.startswith("L") and s_elem.endswith(";") else s_elem
            t_type = t_elem[1:-1] if t_elem.startswith("L") and t_elem.endswith(";") else t_elem
            return self.is_assignable_from(t_type, s_type)

        if t.startswith("["):
            # Source is not an array, target is an array
            return False

        # Interface target
        if self.is_interface(t):
            return t in self.get_all_interfaces(s)

        # Class target
        if self.is_interface(s):
            return False

        return t in self.get_all_superclasses(s)

    # --- Cycle Detection ---
    def has_cycles(self) -> bool:
        return not nx.is_directed_acyclic_graph(self.graph)

    def get_cycles(self) -> List[List[str]]:
        try:
            return list(nx.simple_cycles(self.graph))
        except Exception:
            return []
