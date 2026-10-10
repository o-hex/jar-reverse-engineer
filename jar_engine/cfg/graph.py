"""
jar_engine.cfg.graph: Control Flow Graph data structure and topological analysis.
Conforms to JVMS §4.7.3 and §6.5. Implements dominator trees, natural loops, and cyclomatic complexity.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Iterator, Any, TYPE_CHECKING
import struct
import networkx as nx

from jar_engine.model.constants import Opcode
from jar_engine.model.member_info import ExceptionTableEntry, CodeAttribute
from jar_engine.model.cp_info import ConstantPool
from .basic_block import (
    BasicBlock, Instruction, CFGEdge, CFGEdgeType, partition_basic_blocks
)

if TYPE_CHECKING:
    pass


@dataclass
class NaturalLoop:
    header_id: int
    latch_id: int                                # The tail node of the back-edge
    body: Set[int]                               # All block IDs in the loop
    exits: Set[int] = field(default_factory=set) # Blocks in loop with edges exiting the loop


class ControlFlowGraph:
    """
    Control Flow Graph representation for a single JVM method.
    Backed by NetworkX DiGraph for robust graph algorithms.
    """

    def __init__(self, method_name: str = "", method_desc: str = ""):
        self.method_name = method_name
        self.method_desc = method_desc
        self.blocks: Dict[int, BasicBlock] = {}
        self.edges: List[CFGEdge] = []
        self.entry_block_id: Optional[int] = None
        self._graph: nx.DiGraph = nx.DiGraph()

    @classmethod
    def from_code(
        cls,
        code: bytes,
        exception_table: Optional[List[ExceptionTableEntry]] = None,
        cp: Optional[ConstantPool] = None,
        method_name: str = "",
        method_desc: str = ""
    ) -> "ControlFlowGraph":
        """Constructs a complete CFG from raw bytecode and exception table."""
        if exception_table is None:
            exception_table = []

        cfg = cls(method_name=method_name, method_desc=method_desc)
        blocks = partition_basic_blocks(code, exception_table)
        if not blocks:
            return cfg

        for block in blocks:
            cfg.add_block(block)

        pc_to_block_id: Dict[int, int] = {b.start_pc: b.id for b in blocks}

        # Helper to find target block by PC
        def find_target_block_id(target_pc: int) -> Optional[int]:
            if target_pc in pc_to_block_id:
                return pc_to_block_id[target_pc]
            for b in blocks:
                if b.contains_pc(target_pc):
                    return b.id
            return None

        for i, block in enumerate(blocks):
            terminator = block.terminator
            if not terminator:
                continue

            # 1. Branch / Jump edges
            if terminator.is_conditional_branch:
                # Taken branch
                targets = terminator.get_branch_targets()
                if targets:
                    target_id = find_target_block_id(targets[0])
                    if target_id is not None:
                        cfg.add_edge(CFGEdge(block.id, target_id, CFGEdgeType.CONDITIONAL_BRANCH))

                # Fallthrough branch
                fallthrough_pc = terminator.next_pc
                fallthrough_id = find_target_block_id(fallthrough_pc)
                if fallthrough_id is not None:
                    cfg.add_edge(CFGEdge(block.id, fallthrough_id, CFGEdgeType.FALLTHROUGH))

            elif terminator.opcode in (Opcode.GOTO, Opcode.GOTO_W):
                targets = terminator.get_branch_targets()
                if targets:
                    target_id = find_target_block_id(targets[0])
                    if target_id is not None:
                        cfg.add_edge(CFGEdge(block.id, target_id, CFGEdgeType.UNCONDITIONAL_BRANCH))

            elif terminator.opcode == Opcode.TABLESWITCH:
                pad = (4 - (terminator.pc + 1) % 4) % 4
                operands = terminator.operands
                if len(operands) >= pad + 12:
                    default_off = struct.unpack(">i", operands[pad : pad + 4])[0]
                    default_id = find_target_block_id(terminator.pc + default_off)
                    if default_id is not None:
                        cfg.add_edge(CFGEdge(block.id, default_id, CFGEdgeType.SWITCH_DEFAULT))

                    low, high = struct.unpack(">ii", operands[pad + 4 : pad + 12])
                    count = max(0, high - low + 1)
                    for k in range(count):
                        off_start = pad + 12 + 4 * k
                        if off_start + 4 <= len(operands):
                            case_off = struct.unpack(">i", operands[off_start : off_start + 4])[0]
                            case_id = find_target_block_id(terminator.pc + case_off)
                            if case_id is not None:
                                cfg.add_edge(CFGEdge(block.id, case_id, CFGEdgeType.SWITCH_CASE, case_key=low + k))

            elif terminator.opcode == Opcode.LOOKUPSWITCH:
                pad = (4 - (terminator.pc + 1) % 4) % 4
                operands = terminator.operands
                if len(operands) >= pad + 8:
                    default_off = struct.unpack(">i", operands[pad : pad + 4])[0]
                    default_id = find_target_block_id(terminator.pc + default_off)
                    if default_id is not None:
                        cfg.add_edge(CFGEdge(block.id, default_id, CFGEdgeType.SWITCH_DEFAULT))

                    npairs = struct.unpack(">i", operands[pad + 4 : pad + 8])[0]
                    for k in range(max(0, npairs)):
                        off_start = pad + 8 + 8 * k
                        if off_start + 8 <= len(operands):
                            match_val, case_off = struct.unpack(">ii", operands[off_start : off_start + 8])
                            case_id = find_target_block_id(terminator.pc + case_off)
                            if case_id is not None:
                                cfg.add_edge(CFGEdge(block.id, case_id, CFGEdgeType.SWITCH_CASE, case_key=match_val))

            elif block.has_fallthrough:
                # Natural fallthrough to next sequential block
                fallthrough_pc = terminator.next_pc
                fallthrough_id = find_target_block_id(fallthrough_pc)
                if fallthrough_id is not None:
                    cfg.add_edge(CFGEdge(block.id, fallthrough_id, CFGEdgeType.FALLTHROUGH))

            # 2. Exception edges
            for ex in block.exception_handlers:
                handler_id = find_target_block_id(ex.handler_pc)
                if handler_id is not None:
                    catch_class = None
                    if cp and ex.catch_type != 0:
                        try:
                            catch_class = cp.resolve_class_name(ex.catch_type)
                        except Exception:
                            pass
                    cfg.add_edge(CFGEdge(
                        source_id=block.id,
                        target_id=handler_id,
                        edge_type=CFGEdgeType.EXCEPTION,
                        catch_type=ex.catch_type,
                        catch_class_name=catch_class
                    ))

        return cfg

    @classmethod
    def build(
        cls,
        code_attr: CodeAttribute,
        cp: Optional[ConstantPool] = None,
        method_name: str = "",
        method_desc: str = ""
    ) -> "ControlFlowGraph":
        """Factory constructor building complete CFG from CodeAttribute."""
        return cls.from_code(
            code=code_attr.code,
            exception_table=code_attr.exception_table,
            cp=cp,
            method_name=method_name,
            method_desc=method_desc
        )

    def add_block(self, block: BasicBlock) -> None:
        self.blocks[block.id] = block
        self._graph.add_node(block.id, block=block)
        if self.entry_block_id is None or block.start_pc == 0:
            self.entry_block_id = block.id

    def remove_block(self, block_id: int) -> None:
        if block_id in self.blocks:
            del self.blocks[block_id]
        self.edges = [e for e in self.edges if e.source_id != block_id and e.target_id != block_id]
        if block_id in self._graph:
            self._graph.remove_node(block_id)
        if self.entry_block_id == block_id:
            self.entry_block_id = min(self.blocks.keys()) if self.blocks else None

    def add_edge(self, edge: CFGEdge) -> None:
        # Avoid duplicate identical edges
        for existing in self.edges:
            if (existing.source_id == edge.source_id and
                existing.target_id == edge.target_id and
                existing.edge_type == edge.edge_type and
                existing.case_key == edge.case_key and
                existing.catch_type == edge.catch_type):
                return
        self.edges.append(edge)
        self._graph.add_edge(edge.source_id, edge.target_id, edge_data=edge)

    def remove_edge(self, source_id: int, target_id: int, edge_type: Optional[CFGEdgeType] = None) -> None:
        self.edges = [
            e for e in self.edges
            if not (e.source_id == source_id and e.target_id == target_id and (edge_type is None or e.edge_type == edge_type))
        ]
        if self._graph.has_edge(source_id, target_id):
            if not any(e.source_id == source_id and e.target_id == target_id for e in self.edges):
                self._graph.remove_edge(source_id, target_id)

    def get_block(self, block_id: int) -> Optional[BasicBlock]:
        return self.blocks.get(block_id)

    def get_block_at(self, pc: int) -> Optional[BasicBlock]:
        for b in self.blocks.values():
            if b.contains_pc(pc):
                return b
        return None

    def get_entry_block(self) -> Optional[BasicBlock]:
        if self.entry_block_id is not None:
            return self.blocks.get(self.entry_block_id)
        return None

    def get_exit_blocks(self) -> List[BasicBlock]:
        exits: List[BasicBlock] = []
        for b in self.blocks.values():
            if b.is_terminal:
                exits.append(b)
            elif not list(self._graph.successors(b.id)):
                exits.append(b)
        return exits

    def get_predecessors(self, block_id: int) -> Set[int]:
        if block_id in self._graph:
            return set(self._graph.predecessors(block_id))
        return set()

    def get_successors(self, block_id: int) -> Set[int]:
        if block_id in self._graph:
            return set(self._graph.successors(block_id))
        return set()

    def get_outgoing_edges(self, block_id: int) -> List[CFGEdge]:
        return [e for e in self.edges if e.source_id == block_id]

    def get_incoming_edges(self, block_id: int) -> List[CFGEdge]:
        return [e for e in self.edges if e.target_id == block_id]

    def compute_immediate_dominators(self) -> Dict[int, int]:
        """Returns mapping: block_id -> immediate_dominator_id."""
        if self.entry_block_id is None or self.entry_block_id not in self._graph:
            return {}
        reachable = nx.descendants(self._graph, self.entry_block_id) | {self.entry_block_id}
        sub = self._graph.subgraph(reachable)
        return nx.immediate_dominators(sub, self.entry_block_id)

    def compute_dominators(self) -> Dict[int, Set[int]]:
        """Returns mapping: block_id -> set of all dominator block_ids."""
        idom = self.compute_immediate_dominators()
        doms: Dict[int, Set[int]] = {}
        for node in idom:
            curr = node
            dom_set = {curr}
            visited = {curr}
            while curr in idom and idom[curr] != curr:
                curr = idom[curr]
                if curr in visited:
                    break
                visited.add(curr)
                dom_set.add(curr)
            doms[node] = dom_set
        return doms

    def find_natural_loops(self) -> List[NaturalLoop]:
        """Detects natural loops using back-edges where target dominates source."""
        doms = self.compute_dominators()
        loops: List[NaturalLoop] = []

        for edge in self.edges:
            u, v = edge.source_id, edge.target_id
            if v in doms.get(u, set()):
                # v dominates u -> back-edge (u, v), loop header is v
                body = {v, u}
                stack = [u] if u != v else []
                while stack:
                    curr = stack.pop()
                    for pred in self._graph.predecessors(curr):
                        if pred not in body:
                            body.add(pred)
                            stack.append(pred)

                exits: Set[int] = set()
                for b_id in body:
                    for succ in self._graph.successors(b_id):
                        if succ not in body:
                            exits.add(b_id)

                loops.append(NaturalLoop(header_id=v, latch_id=u, body=body, exits=exits))

        return loops

    def get_cyclomatic_complexity(self) -> int:
        """Computes McCabe cyclomatic complexity M = E - N + 2P on reachable blocks."""
        if not self.blocks or self.entry_block_id is None or self.entry_block_id not in self._graph:
            return 1

        reachable = nx.descendants(self._graph, self.entry_block_id) | {self.entry_block_id}
        sub = self._graph.subgraph(reachable)

        N = sub.number_of_nodes()
        # Count only non-exception edges for standard McCabe or count all edges in sub
        E = sub.number_of_edges()
        P = 1  # Single procedure component

        M = E - N + 2 * P
        return max(1, M)

    @property
    def cyclomatic_complexity(self) -> int:
        return self.get_cyclomatic_complexity()

    def reassemble_code(self, cp: Optional[ConstantPool] = None) -> Tuple[bytes, List[ExceptionTableEntry], int, int]:
        """Reassembles linearized bytecode from CFG blocks."""
        from .dead_code import reassemble_cfg
        return reassemble_cfg(self, cp)
