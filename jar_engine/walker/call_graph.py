"""
jar_engine.walker.call_graph: Whole-archive polymorphic call graph and field access indexing.
Conforms to JVMS §5.4 and §6.5 using CHA/RTA static dispatch resolution.
"""

from __future__ import annotations
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Iterator, Any, Union, TYPE_CHECKING
import networkx as nx

from jar_engine.model.constants import (
    Opcode, ConstantTag, FIXED_OPCODE_LENGTHS
)
from jar_engine.model.class_file import ClassFile
from jar_engine.model.member_info import (
    MethodInfo, FieldInfo, BootstrapMethodsAttribute
)
from jar_engine.model.cp_info import (
    CONSTANT_InvokeDynamic_info, CONSTANT_MethodHandle_info
)
from jar_engine.walker.hierarchy import InheritanceHierarchy

if TYPE_CHECKING:
    from jar_engine.model.jar_archive import JarArchive


@dataclass(frozen=True)
class MethodReference:
    """Canonical method identifier."""
    class_name: str
    name: str
    descriptor: str

    @property
    def id(self) -> str:
        return f"L{self.class_name};->{self.name}{self.descriptor}"

    @classmethod
    def from_id(cls, method_id: str) -> "MethodReference":
        if not method_id.startswith("L") or ";->" not in method_id:
            raise ValueError(f"Invalid method ID format: {method_id}")
        cls_part, rest = method_id[1:].split(";->", 1)
        paren_idx = rest.find("(")
        if paren_idx == -1:
            raise ValueError(f"Invalid method descriptor in ID: {method_id}")
        name = rest[:paren_idx]
        desc = rest[paren_idx:]
        return cls(class_name=cls_part, name=name, descriptor=desc)


@dataclass(frozen=True)
class FieldReference:
    """Canonical field identifier."""
    class_name: str
    name: str
    descriptor: str

    @property
    def id(self) -> str:
        return f"L{self.class_name};->{self.name}:{self.descriptor}"

    @classmethod
    def from_id(cls, field_id: str) -> "FieldReference":
        if not field_id.startswith("L") or ";->" not in field_id or ":" not in field_id:
            raise ValueError(f"Invalid field ID format: {field_id}")
        cls_part, rest = field_id[1:].split(";->", 1)
        name, desc = rest.split(":", 1)
        return cls(class_name=cls_part, name=name, descriptor=desc)


@dataclass
class CallEdge:
    """Represents a call edge from caller method to callee method."""
    caller: str                       # MethodID
    callee: str                       # MethodID
    opcode: str                       # "invokevirtual", "invokestatic", etc.
    opcode_num: int
    pc: int
    is_external: bool = False
    receiver_class: Optional[str] = None
    bsm_info: Optional[str] = None

    def __hash__(self) -> int:
        return hash((self.caller, self.callee, self.opcode, self.pc))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CallEdge):
            return False
        return (
            self.caller == other.caller
            and self.callee == other.callee
            and self.opcode == other.opcode
            and self.pc == other.pc
        )


@dataclass
class FieldAccessRecord:
    """Represents a field read or write instruction."""
    caller: str          # MethodID
    field: str           # FieldID
    access_type: str     # "read" | "write"
    opcode_num: int
    is_static: bool
    pc: int


class InstructionScanner:
    """Decodes bytecode instruction stream yielding (pc, opcode, operand_bytes)."""

    @staticmethod
    def get_instruction_length(code: bytes, pc: int) -> int:
        if pc >= len(code):
            return 1
        opcode = code[pc]

        if opcode == Opcode.TABLESWITCH:
            pad = (4 - (pc + 1) % 4) % 4
            if pc + 1 + pad + 12 > len(code):
                return max(1, len(code) - pc)
            low, high = struct.unpack(">ii", code[pc + 1 + pad + 4 : pc + 1 + pad + 12])
            count = max(0, high - low + 1)
            return 1 + pad + 12 + count * 4

        elif opcode == Opcode.LOOKUPSWITCH:
            pad = (4 - (pc + 1) % 4) % 4
            if pc + 1 + pad + 8 > len(code):
                return max(1, len(code) - pc)
            npairs = struct.unpack(">i", code[pc + 1 + pad + 4 : pc + 1 + pad + 8])[0]
            return 1 + pad + 8 + max(0, npairs) * 8

        elif opcode == Opcode.WIDE:
            if pc + 1 >= len(code):
                return 1
            sub_op = code[pc + 1]
            return 6 if sub_op == Opcode.IINC else 4

        return FIXED_OPCODE_LENGTHS.get(opcode, 1)

    @classmethod
    def scan(cls, code: bytes) -> Iterator[Tuple[int, int, bytes]]:
        pc = 0
        n = len(code)
        while pc < n:
            length = cls.get_instruction_length(code, pc)
            if length <= 0 or pc + length > n:
                yield (pc, code[pc], code[pc + 1 : n])
                break
            yield (pc, code[pc], code[pc + 1 : pc + length])
            pc += length


class CallGraph:
    """Whole-archive static call graph and field access index."""

    def __init__(self, mode: str = "cha"):
        self.mode = mode.lower()
        self.graph: nx.DiGraph = nx.DiGraph()
        self.call_edges: List[CallEdge] = []
        self.field_accesses: List[FieldAccessRecord] = []
        self.field_readers: Dict[str, Set[str]] = defaultdict(set)
        self.field_writers: Dict[str, Set[str]] = defaultdict(set)
        self.method_reads: Dict[str, Set[str]] = defaultdict(set)
        self.method_writes: Dict[str, Set[str]] = defaultdict(set)
        self._archive: Optional[Any] = None
        self._hierarchy: Optional[InheritanceHierarchy] = None

    @classmethod
    def build(
        cls,
        archive: "JarArchive",
        hierarchy: Optional[InheritanceHierarchy] = None,
        mode: str = "cha"
    ) -> "CallGraph":
        builder = cls(mode=mode)
        builder._archive = archive
        if hierarchy is None:
            hierarchy = archive.get_hierarchy()
        builder._hierarchy = hierarchy
        builder._construct(archive, hierarchy)
        return builder

    def _find_method_in_class_or_super(
        self,
        class_name: str,
        name: str,
        desc: str,
        archive: "JarArchive",
        hierarchy: InheritanceHierarchy
    ) -> Optional[Tuple[str, MethodInfo]]:
        """Searches class and superclasses upward for matching concrete/declared method."""
        curr = class_name
        visited: Set[str] = set()
        while curr and curr not in visited:
            visited.add(curr)
            cf = archive.get_class(curr)
            if cf:
                m = cf.get_method(name, desc)
                if m and not m.is_abstract:
                    return (curr, m)
            curr = hierarchy.get_direct_superclass(curr)
        return None

    def _find_default_method(
        self,
        class_name: str,
        name: str,
        desc: str,
        archive: "JarArchive",
        hierarchy: InheritanceHierarchy
    ) -> Optional[Tuple[str, MethodInfo]]:
        """
        Searches implemented/extended interfaces for Java 8+ default methods per JVMS §5.4.3.3.
        Filters candidate interfaces to retain only maximally-specific interfaces:
        for any two candidates A and B, if A is a superinterface of B (i.e. B extends A),
        eliminate A. Sorts remaining candidate interfaces deterministically.
        """
        candidates: List[Tuple[str, MethodInfo]] = []
        for iface in hierarchy.get_all_interfaces(class_name):
            cf = archive.get_class(iface)
            if cf:
                m = cf.get_method(name, desc)
                if m and not m.is_abstract and not m.is_static:
                    candidates.append((iface, m))

        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        # Maximally-specific filtering per JVMS §5.4.3.3:
        cand_ifaces = {c[0] for c in candidates}
        eliminated: Set[str] = set()
        for iface_a in cand_ifaces:
            for iface_b in cand_ifaces:
                if iface_a != iface_b and iface_a in hierarchy.get_all_interfaces(iface_b):
                    eliminated.add(iface_a)

        remaining = [c for c in candidates if c[0] not in eliminated]
        if remaining:
            remaining.sort(key=lambda c: c[0])
            return remaining[0]

        candidates.sort(key=lambda c: c[0])
        return candidates[0]

    def _find_field_in_interfaces(
        self,
        class_name: str,
        field_name: str,
        field_desc: str,
        archive: "JarArchive",
        hierarchy: InheritanceHierarchy,
        visited_ifaces: Optional[Set[str]] = None
    ) -> Optional[str]:
        """Recursively searches direct and indirect superinterfaces per JVMS §5.4.3.2."""
        if visited_ifaces is None:
            visited_ifaces = set()

        queue = list(hierarchy.get_direct_interfaces(class_name))
        while queue:
            iface = queue.pop(0)
            if iface in visited_ifaces:
                continue
            visited_ifaces.add(iface)

            cf = archive.get_class(iface)
            if cf and cf.get_field(field_name, field_desc):
                return iface

            queue.extend(hierarchy.get_direct_interfaces(iface))
        return None

    def _resolve_declaring_field_class(
        self,
        class_name: str,
        field_name: str,
        field_desc: str,
        archive: "JarArchive",
        hierarchy: InheritanceHierarchy
    ) -> str:
        """
        Walks hierarchy upward to find class or interface actually declaring the field.
        Strictly conforms to JVMS §5.4.3.2 (Field Resolution):
        1. Class C itself.
        2. Direct superinterfaces of C recursively.
        3. Superclass S of C recursively (including S's superinterfaces).
        """
        curr = class_name
        visited: Set[str] = set()
        visited_ifaces: Set[str] = set()

        while curr and curr not in visited:
            visited.add(curr)

            # Step 1: Check class curr itself
            cf = archive.get_class(curr)
            if cf and cf.get_field(field_name, field_desc):
                return curr

            # Step 2: Search interfaces of curr
            found_iface = self._find_field_in_interfaces(
                curr, field_name, field_desc, archive, hierarchy, visited_ifaces
            )
            if found_iface:
                return found_iface

            # Step 3: Walk to superclass
            curr = hierarchy.get_direct_superclass(curr)

        return class_name

    # Contract & dispatch alias
    _resolve_field_declaring_class = _resolve_declaring_field_class

    def _construct(self, archive: "JarArchive", hierarchy: InheritanceHierarchy) -> None:
        # Step 1: Rapid Type Analysis (RTA) scan for instantiated classes
        instantiated_classes: Set[str] = set()
        for cf in archive.classes.values():
            for m in cf.methods:
                code_attr = m.code_attribute(cf.constant_pool)
                if not code_attr:
                    continue
                for pc, op, operands in InstructionScanner.scan(code_attr.code):
                    if op == Opcode.NEW and len(operands) >= 2:
                        idx = struct.unpack(">H", operands[:2])[0]
                        try:
                            inst_cls = cf.resolve_class_name(idx)
                            instantiated_classes.add(inst_cls)
                        except Exception:
                            pass

        # Step 2: Register all declared methods as graph nodes
        for cls_name, cf in archive.classes.items():
            for m in cf.methods:
                m_ref = MethodReference(cls_name, m.name(cf.constant_pool), m.descriptor(cf.constant_pool))
                self.graph.add_node(m_ref.id, method_ref=m_ref, class_name=cls_name)

        # Step 3: Scan bytecode of each method
        for cls_name, cf in archive.classes.items():
            for m in cf.methods:
                caller_ref = MethodReference(cls_name, m.name(cf.constant_pool), m.descriptor(cf.constant_pool))
                caller_id = caller_ref.id
                code_attr = m.code_attribute(cf.constant_pool)
                if not code_attr:
                    continue

                for pc, op, operands in InstructionScanner.scan(code_attr.code):
                    # --- METHOD INVOCATIONS ---
                    if op in (Opcode.INVOKEVIRTUAL, Opcode.INVOKEINTERFACE):
                        if len(operands) < 2:
                            continue
                        cp_idx = struct.unpack(">H", operands[:2])[0]
                        try:
                            target_cls, m_name, m_desc = cf.resolve_member_ref(cp_idx)
                        except Exception:
                            continue

                        op_name = "invokevirtual" if op == Opcode.INVOKEVIRTUAL else "invokeinterface"

                        # Receiver set via CHA or RTA
                        if op == Opcode.INVOKEVIRTUAL:
                            receivers = {target_cls} | hierarchy.get_all_subclasses(target_cls)
                        else:
                            receivers = hierarchy.get_all_implementors(target_cls)
                            if not receivers:
                                receivers = {target_cls}

                        # Apply RTA filtering if requested
                        if self.mode == "rta":
                            filtered = receivers & instantiated_classes
                            if filtered:
                                receivers = filtered

                        # Resolve concrete dispatch targets
                        targets_found = False
                        seen_callees: Set[str] = set()
                        for r in receivers:
                            resolved = self._find_method_in_class_or_super(r, m_name, m_desc, archive, hierarchy)
                            if not resolved:
                                resolved = self._find_default_method(r, m_name, m_desc, archive, hierarchy)

                            if resolved:
                                decl_cls, _ = resolved
                                callee_id = MethodReference(decl_cls, m_name, m_desc).id
                                if callee_id in seen_callees:
                                    continue
                                seen_callees.add(callee_id)

                                edge = CallEdge(
                                    caller=caller_id,
                                    callee=callee_id,
                                    opcode=op_name,
                                    opcode_num=op,
                                    pc=pc,
                                    is_external=False,
                                    receiver_class=r
                                )
                                self.call_edges.append(edge)
                                self.graph.add_edge(caller_id, callee_id, opcode=op_name, pc=pc)
                                targets_found = True

                        if not targets_found:
                            # External / library target
                            callee_id = MethodReference(target_cls, m_name, m_desc).id
                            edge = CallEdge(
                                caller=caller_id,
                                callee=callee_id,
                                opcode=op_name,
                                opcode_num=op,
                                pc=pc,
                                is_external=True
                            )
                            self.call_edges.append(edge)
                            self.graph.add_edge(caller_id, callee_id, opcode=op_name, pc=pc)

                    elif op == Opcode.INVOKESTATIC:
                        if len(operands) < 2:
                            continue
                        cp_idx = struct.unpack(">H", operands[:2])[0]
                        try:
                            target_cls, m_name, m_desc = cf.resolve_member_ref(cp_idx)
                        except Exception:
                            continue

                        # Resolve static method in target class or superclasses/superinterfaces
                        decl_cls = target_cls
                        is_ext = not archive.has_class(target_cls)
                        resolved = self._find_method_in_class_or_super(target_cls, m_name, m_desc, archive, hierarchy)
                        if resolved:
                            decl_cls, _ = resolved
                            is_ext = False

                        callee_id = MethodReference(decl_cls, m_name, m_desc).id
                        edge = CallEdge(
                            caller=caller_id,
                            callee=callee_id,
                            opcode="invokestatic",
                            opcode_num=op,
                            pc=pc,
                            is_external=is_ext
                        )
                        self.call_edges.append(edge)
                        self.graph.add_edge(caller_id, callee_id, opcode="invokestatic", pc=pc)

                    elif op == Opcode.INVOKESPECIAL:
                        if len(operands) < 2:
                            continue
                        cp_idx = struct.unpack(">H", operands[:2])[0]
                        try:
                            target_cls, m_name, m_desc = cf.resolve_member_ref(cp_idx)
                        except Exception:
                            continue

                        decl_cls = target_cls
                        is_ext = not archive.has_class(target_cls)

                        if m_name == "<init>":
                            # Constructor call
                            decl_cls = target_cls
                        elif target_cls == cls_name:
                            # Private method
                            decl_cls = cls_name
                        else:
                            # Superclass call: resolve starting from direct superclass of caller
                            super_c = hierarchy.get_direct_superclass(cls_name)
                            if super_c:
                                resolved = self._find_method_in_class_or_super(super_c, m_name, m_desc, archive, hierarchy)
                                if resolved:
                                    decl_cls, _ = resolved
                                    is_ext = False

                        callee_id = MethodReference(decl_cls, m_name, m_desc).id
                        edge = CallEdge(
                            caller=caller_id,
                            callee=callee_id,
                            opcode="invokespecial",
                            opcode_num=op,
                            pc=pc,
                            is_external=is_ext
                        )
                        self.call_edges.append(edge)
                        self.graph.add_edge(caller_id, callee_id, opcode="invokespecial", pc=pc)

                    elif op == Opcode.INVOKEDYNAMIC:
                        if len(operands) < 2:
                            continue
                        cp_idx = struct.unpack(">H", operands[:2])[0]
                        try:
                            indy_entry = cf.constant_pool[cp_idx]
                        except Exception:
                            continue

                        if isinstance(indy_entry, CONSTANT_InvokeDynamic_info):
                            bsm_idx = indy_entry.bootstrap_method_attr_index
                            nat_name, nat_desc = cf.resolve_name_and_type(indy_entry.name_and_type_index)
                            bsm_attr = cf.get_attribute("BootstrapMethods")

                            callee_id = None
                            bsm_info = None

                            if isinstance(bsm_attr, BootstrapMethodsAttribute) and bsm_idx < len(bsm_attr.bootstrap_methods):
                                bm = bsm_attr.bootstrap_methods[bsm_idx]
                                try:
                                    bsm_handle = cf.constant_pool[bm.bootstrap_method_ref]
                                    if isinstance(bsm_handle, CONSTANT_MethodHandle_info):
                                        bsm_cls, bsm_name, _ = cf.resolve_member_ref(bsm_handle.reference_index)
                                        bsm_info = f"{bsm_cls}.{bsm_name}"

                                        # Lambda resolution: argument 1 is implMethod
                                        if len(bm.bootstrap_arguments) > 1:
                                            impl_handle = cf.constant_pool[bm.bootstrap_arguments[1]]
                                            if isinstance(impl_handle, CONSTANT_MethodHandle_info):
                                                l_cls, l_name, l_desc = cf.resolve_member_ref(impl_handle.reference_index)
                                                callee_id = MethodReference(l_cls, l_name, l_desc).id
                                except Exception:
                                    pass

                            if not callee_id:
                                # Fallback: dynamic call site
                                callee_id = f"L{cls_name};->indy${nat_name}{nat_desc}"

                            is_ext = False
                            try:
                                ref = MethodReference.from_id(callee_id)
                                is_ext = not archive.has_class(ref.class_name)
                            except Exception:
                                pass

                            edge = CallEdge(
                                caller=caller_id,
                                callee=callee_id,
                                opcode="invokedynamic",
                                opcode_num=op,
                                pc=pc,
                                is_external=is_ext,
                                bsm_info=bsm_info
                            )
                            self.call_edges.append(edge)
                            self.graph.add_edge(caller_id, callee_id, opcode="invokedynamic", pc=pc)

                    # --- FIELD ACCESSES ---
                    elif op in (Opcode.GETSTATIC, Opcode.PUTSTATIC, Opcode.GETFIELD, Opcode.PUTFIELD):
                        if len(operands) < 2:
                            continue
                        cp_idx = struct.unpack(">H", operands[:2])[0]
                        try:
                            f_cls, f_name, f_desc = cf.resolve_member_ref(cp_idx)
                        except Exception:
                            continue

                        # Resolve declaring class of field
                        decl_cls = self._resolve_declaring_field_class(f_cls, f_name, f_desc, archive, hierarchy)
                        field_id = FieldReference(decl_cls, f_name, f_desc).id
                        is_static = op in (Opcode.GETSTATIC, Opcode.PUTSTATIC)
                        is_read = op in (Opcode.GETSTATIC, Opcode.GETFIELD)
                        access_type = "read" if is_read else "write"

                        record = FieldAccessRecord(
                            caller=caller_id,
                            field=field_id,
                            access_type=access_type,
                            opcode_num=op,
                            is_static=is_static,
                            pc=pc
                        )
                        self.field_accesses.append(record)

                        if is_read:
                            self.field_readers[field_id].add(caller_id)
                            self.method_reads[caller_id].add(field_id)
                        else:
                            self.field_writers[field_id].add(caller_id)
                            self.method_writes[caller_id].add(field_id)

    # --- Query Interface Methods ---
    def get_callers(self, method_signature: str) -> Set[str]:
        """Returns direct caller method IDs."""
        if method_signature in self.graph:
            return set(self.graph.predecessors(method_signature))
        return set()

    def get_callees(self, method_signature: str) -> Set[str]:
        """Returns direct callee method IDs."""
        if method_signature in self.graph:
            return set(self.graph.successors(method_signature))
        return set()

    def get_field_readers(self, field_signature: str) -> Set[str]:
        """Returns all method IDs reading the given field."""
        return set(self.field_readers.get(field_signature, set()))

    def get_field_writers(self, field_signature: str) -> Set[str]:
        """Returns all method IDs writing the given field."""
        return set(self.field_writers.get(field_signature, set()))

    def get_fields_read_by(self, method_signature: str) -> Set[str]:
        """Returns all field IDs read by the given method."""
        return set(self.method_reads.get(method_signature, set()))

    def get_fields_written_by(self, method_signature: str) -> Set[str]:
        """Returns all field IDs written by the given method."""
        return set(self.method_writes.get(method_signature, set()))

    def get_transitive_callees(self, method_signature: str) -> Set[str]:
        """Returns all methods reachable from the given method."""
        if method_signature in self.graph:
            return set(nx.descendants(self.graph, method_signature))
        return set()

    def get_transitive_callers(self, method_signature: str) -> Set[str]:
        """Returns all methods that can reach the given method."""
        if method_signature in self.graph:
            return set(nx.ancestors(self.graph, method_signature))
        return set()

    def get_entry_points(self) -> Set[str]:
        """Methods with in-degree 0 (not called by any method in archive)."""
        return {n for n in self.graph.nodes() if self.graph.in_degree(n) == 0}

    def get_leaf_methods(self) -> Set[str]:
        """Methods with out-degree 0."""
        return {n for n in self.graph.nodes() if self.graph.out_degree(n) == 0}

    def compute_dependency_slice(self, root_classes: List[str], direction: str) -> Set[str]:
        """
        Computes inter-class forward or backward slice for given root classes.
        """
        from jar_engine.walker.dependency_slice import ClassDependencyGraph
        cdg = ClassDependencyGraph.build(self._archive, self._hierarchy, self)
        d = direction.lower()
        if d in ("forward", "downstream", "dependencies"):
            return cdg.forward_slice(root_classes)
        elif d in ("backward", "upstream", "dependents"):
            return cdg.backward_slice(root_classes)
        else:
            raise ValueError(f"Unknown direction '{direction}'; must be 'forward' or 'backward'")

    @property
    def adjacency(self) -> Dict[str, Set[str]]:
        """Returns adjacency map of caller_id -> set of direct callee_ids."""
        adj: Dict[str, Set[str]] = defaultdict(set)
        for e in self.call_edges:
            adj[e.caller].add(e.callee)
        return dict(adj)

    @property
    def call_edges_set(self) -> Set[CallEdge]:
        """Returns set of unique CallEdge objects."""
        return set(self.call_edges)

    def get_caller_edges(self, caller_id: str) -> Set[CallEdge]:
        """Returns unique CallEdges originating from caller_id."""
        return {e for e in self.call_edges if e.caller == caller_id}

    def get_callee_edges(self, callee_id: str) -> Set[CallEdge]:
        """Returns unique CallEdges targeting callee_id."""
        return {e for e in self.call_edges if e.callee == callee_id}
