"""
jar_engine.renamer: Semantic symbol restoration, reference rewriting, and repackaging engine.

F24: Symbol mapping schema and validation.
F25: Union-Find method override equivalence clustering.
F26: JDK library immutability pinning.
F27: Global constant pool and bytecode reference rewriting.
F28: Access widening for package relocations.
F29: Valid JAR repackaging and SPI provider synchronization.
"""

from __future__ import annotations
from typing import Optional
from pathlib import Path

from jar_engine.model.jar_archive import JarArchive
from jar_engine.walker.hierarchy import InheritanceHierarchy
from .schema import SymbolMapping, MappingValidationError
from .union_find import MethodEquivalenceCluster
from .ref_rewriter import ReferenceRewriter, RenamingReport, rewrite_descriptor
from .repackager import JarRepackager


class SymbolRenamer:
    """High-level symbol renaming coordinator maintaining type safety and polymorphic consistency."""

    def __init__(self, hierarchy: Optional[InheritanceHierarchy] = None):
        self.hierarchy = hierarchy

    def apply_mapping(
        self,
        archive: JarArchive,
        mapping: SymbolMapping,
        hierarchy: Optional[InheritanceHierarchy] = None
    ) -> RenamingReport:
        hier = hierarchy or self.hierarchy or archive.get_hierarchy()
        cluster = MethodEquivalenceCluster(hier)
        cluster.build_from_archive(archive)

        rewriter = ReferenceRewriter(cluster=cluster)
        report = rewriter.apply_mapping(archive, mapping)

        repackager = JarRepackager(strip_signatures=True)
        repackager.synchronize_manifest_and_services(archive, mapping)

        return report


__all__ = [
    "SymbolMapping",
    "MappingValidationError",
    "SymbolRenamer",
    "RenamingReport",
    "MethodEquivalenceCluster",
    "ReferenceRewriter",
    "JarRepackager",
    "rewrite_descriptor",
]
