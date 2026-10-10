"""
jar_engine.cfg: Control Flow Graph (CFG) Construction, De-Flattening, and Normalization.
Provides basic block partitioning (F13), loop unflattening (F14), opaque predicate elimination (F15),
and dead code pruning / reassembly (F19).
"""

from .basic_block import (
    Instruction,
    CFGEdge,
    CFGEdgeType,
    BasicBlock,
    decode_instructions,
    find_leaders,
    partition_basic_blocks,
)
from .graph import (
    ControlFlowGraph,
    NaturalLoop,
)
from .deflattener import (
    ControlFlowDeflattener,
    DeflatteningReport,
    deflatten_control_flow,
    OpaquePredicateEliminator,
    OpaquePredicateReport,
    eliminate_opaque_predicates,
)
from .dead_code import (
    prune_unreachable_blocks,
    compact_goto_chains,
    eliminate_fallthrough_gotos,
    strip_nops_from_bytecode,
    reassemble_cfg,
    prune_dead_code_and_nops,
)

__all__ = [
    "Instruction",
    "CFGEdge",
    "CFGEdgeType",
    "BasicBlock",
    "decode_instructions",
    "find_leaders",
    "partition_basic_blocks",
    "ControlFlowGraph",
    "NaturalLoop",
    "ControlFlowDeflattener",
    "DeflatteningReport",
    "deflatten_control_flow",
    "OpaquePredicateEliminator",
    "OpaquePredicateReport",
    "eliminate_opaque_predicates",
    "prune_unreachable_blocks",
    "compact_goto_chains",
    "eliminate_fallthrough_gotos",
    "strip_nops_from_bytecode",
    "reassemble_cfg",
    "prune_dead_code_and_nops",
]
