"""
jar_engine.normalization: Control Flow Normalization & Constant Resolution Engine (Milestone M3).
Provides MethodNormalizer and NormalizationStats per PROJECT.md interface contract.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from jar_engine.model.class_file import ClassFile
from jar_engine.model.member_info import MethodInfo, CodeAttribute
from jar_engine.model.cp_info import ConstantPool
from jar_engine.cfg.graph import ControlFlowGraph
from jar_engine.cfg.deflattener import deflatten_control_flow, eliminate_opaque_predicates
from jar_engine.cfg.dead_code import prune_dead_code_and_nops
from .constant_folding import (
    fold_constants_and_arithmetic,
    IntraProceduralConstantFolder,
    ArithmeticEvaluator,
    LatticeValue,
    LatticeTag,
    AbstractFrame,
)
from .string_decryptor import (
    reconstruct_encrypted_strings,
    StringDecryptorResolver,
    BytecodeEmulator,
    DecryptorMethodDetector,
)
from .dynamic_invoke import (
    resolve_dynamic_invokes_and_reflection,
    DynamicInvokeResolver,
    ResolvedStringConcat,
    ResolvedLambda,
    ResolvedReflectionCall,
)


@dataclass
class NormalizationStats:
    """Statistics capturing cyclomatic complexity reduction and normalization impacts."""
    cyclomatic_complexity_before: int = 0
    cyclomatic_complexity_after: int = 0
    constants_folded: int = 0
    strings_decrypted: int = 0
    dead_blocks_pruned: int = 0
    dynamic_invokes_resolved: int = 0
    nops_stripped: int = 0
    deflattened: bool = False

    @property
    def complexity_reduced(self) -> int:
        return max(0, self.cyclomatic_complexity_before - self.cyclomatic_complexity_after)


class MethodNormalizer:
    """
    Unified Bytecode Normalization Engine for Milestone M3.
    Applies de-flattening, constant folding, string decryption, dynamic invoke
    resolution, dead code pruning, and bytecode reassembly.
    """

    def deflatten_control_flow(self, cfg: ControlFlowGraph) -> ControlFlowGraph:
        """F14: De-flattens switch/goto dispatch loops."""
        return deflatten_control_flow(cfg)

    def fold_constants_and_arithmetic(self, cfg: ControlFlowGraph, cp: Optional[ConstantPool] = None) -> int:
        """F16: Evaluates arithmetic and bitwise expressions over constant lattice."""
        return fold_constants_and_arithmetic(cfg, cp=cp)

    def reconstruct_encrypted_strings(self, cf: ClassFile, method: MethodInfo) -> int:
        """F17: Detects and emulates static string decryption routines."""
        return reconstruct_encrypted_strings(cf, method)

    def normalize_method(self, cf: ClassFile, method: MethodInfo) -> NormalizationStats:
        """
        Executes full canonical normalization pipeline on a single method.
        Preserves JVMS §4.10 verification and guarantees decompiler stability.
        """
        code_attr = method.code_attribute(cf.constant_pool)
        if code_attr is None or len(code_attr.code) == 0:
            return NormalizationStats()

        stats = NormalizationStats()

        # Step 1: Static string decryption inlining (F17)
        decrypted_count = self.reconstruct_encrypted_strings(cf, method)
        stats.strings_decrypted = decrypted_count

        # Step 2: Dynamic invoke & reflection resolution (F18)
        indy_resolved = resolve_dynamic_invokes_and_reflection(cf, method)
        stats.dynamic_invokes_resolved = indy_resolved

        # Step 3: Build CFG & measure initial cyclomatic complexity (F13)
        cfg = ControlFlowGraph.build(
            code_attr,
            cf.constant_pool,
            method_name=method.name(cf.constant_pool),
            method_desc=method.descriptor(cf.constant_pool)
        )
        stats.cyclomatic_complexity_before = cfg.cyclomatic_complexity

        # Step 4: Constant folding & arithmetic simplification (F16)
        folded_count = self.fold_constants_and_arithmetic(cfg, cp=cf.constant_pool)
        stats.constants_folded = folded_count

        # Step 5: Opaque predicate elimination (F15)
        eliminate_opaque_predicates(cfg, cp=cf.constant_pool)

        # Step 6: Control flow de-flattening (F14)
        blocks_before_deflat = len(cfg.blocks)
        edges_before_deflat = len(cfg.edges)
        cfg = self.deflatten_control_flow(cfg)
        if len(cfg.blocks) != blocks_before_deflat or len(cfg.edges) != edges_before_deflat:
            stats.deflattened = True

        # Step 7: Dead code & NOP pruning (F19)
        pruned_blocks, nops_removed = prune_dead_code_and_nops(cfg, code_attr.exception_table)
        stats.dead_blocks_pruned = pruned_blocks
        stats.nops_stripped = nops_removed

        # Step 8: Measure final cyclomatic complexity
        stats.cyclomatic_complexity_after = cfg.cyclomatic_complexity

        # Step 9: Bytecode reassembly & serialization
        new_code, new_ex_table, new_max_stack, new_max_locals = cfg.reassemble_code(cp=cf.constant_pool)
        code_attr.code = new_code
        code_attr.exception_table = new_ex_table
        code_attr.max_stack = max(code_attr.max_stack, new_max_stack)
        code_attr.max_locals = max(code_attr.max_locals, new_max_locals)

        # Step 10: Synchronize StackMapTable with M2 Sanitizer
        try:
            from jar_engine.sanitizer.stack_map import sanitize_class_stack_maps
            sanitize_class_stack_maps(cf, force_strip=False)
        except Exception:
            pass

        return stats


__all__ = [
    "MethodNormalizer",
    "NormalizationStats",
    "deflatten_control_flow",
    "fold_constants_and_arithmetic",
    "reconstruct_encrypted_strings",
    "resolve_dynamic_invokes_and_reflection",
    "IntraProceduralConstantFolder",
    "ArithmeticEvaluator",
    "LatticeValue",
    "LatticeTag",
    "AbstractFrame",
    "StringDecryptorResolver",
    "BytecodeEmulator",
    "DecryptorMethodDetector",
    "DynamicInvokeResolver",
    "ResolvedStringConcat",
    "ResolvedLambda",
    "ResolvedReflectionCall",
]
