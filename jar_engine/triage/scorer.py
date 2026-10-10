"""
jar_engine.triage.scorer: Multi-factor triage scoring and architectural classification (Feature F21).

Uses TF-IDF API footprint scoring, constant pool member references, and call graph propagation
to classify and rank classes by functional capability.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import math

from jar_engine.model.class_file import ClassFile
from jar_engine.model.jar_archive import JarArchive
from jar_engine.model.cp_info import (
    CONSTANT_Class_info, CONSTANT_Methodref_info, CONSTANT_InterfaceMethodref_info, CONSTANT_Fieldref_info
)
from jar_engine.walker.call_graph import CallGraph
from .taxonomy import CapabilityDomain, DOMAIN_SIGNATURES


@dataclass
class ClassTriageResult:
    """Triage categorization and scoring for an individual class."""
    class_name: str
    primary_domain: Optional[CapabilityDomain] = None
    confidence: float = 0.0
    domain_scores: Dict[CapabilityDomain, float] = field(default_factory=dict)
    matched_apis: Dict[CapabilityDomain, List[str]] = field(default_factory=dict)
    is_security_sensitive: bool = False

    @property
    def has_detected_capabilities(self) -> bool:
        return self.primary_domain is not None and self.confidence > 0.0


@dataclass
class ArchiveTriageReport:
    """Whole-archive triage analysis summary."""
    total_classes: int = 0
    domain_distribution: Dict[CapabilityDomain, int] = field(default_factory=lambda: defaultdict(int))
    triage_results: Dict[str, ClassTriageResult] = field(default_factory=dict)
    high_priority_classes: List[str] = field(default_factory=list)

    def get_classes_in_domain(self, domain: CapabilityDomain) -> List[str]:
        return [
            c_name for c_name, res in self.triage_results.items()
            if res.primary_domain == domain
        ]


class CapabilityTriager:
    """
    Analyzes API call footprints across classes and the call graph to classify subsystems.
    """

    def __init__(self, sensitivity_threshold: float = 1.0):
        self.sensitivity_threshold = sensitivity_threshold

    def _extract_referenced_apis(self, cf: ClassFile) -> Set[str]:
        """Extracts all referenced classes, methods, and fields from the class constant pool."""
        cp = cf.constant_pool
        refs: Set[str] = set()

        for idx, entry in enumerate(cp._entries):
            if entry is None:
                continue
            if isinstance(entry, CONSTANT_Class_info):
                try:
                    refs.add(cp.resolve_class_name(idx))
                except Exception:
                    pass
            elif isinstance(entry, (CONSTANT_Methodref_info, CONSTANT_InterfaceMethodref_info, CONSTANT_Fieldref_info)):
                try:
                    c_name, m_name, _ = cp.resolve_member_ref(idx)
                    refs.add(f"{c_name};->{m_name}")
                    refs.add(c_name)
                except Exception:
                    pass

        return refs

    def triage_class(
        self,
        class_name: str,
        archive: JarArchive,
        cg: Optional[CallGraph] = None
    ) -> ClassTriageResult:
        """Computes domain scores and identifies primary architectural role for a class."""
        cf = archive.get_class(class_name)
        if cf is None:
            return ClassTriageResult(class_name=class_name)

        refs = self._extract_referenced_apis(cf)
        scores: Dict[CapabilityDomain, float] = {d: 0.0 for d in CapabilityDomain}
        matched: Dict[CapabilityDomain, List[str]] = defaultdict(list)

        for domain, signatures in DOMAIN_SIGNATURES.items():
            for ref in refs:
                for sig in signatures:
                    if sig in ref:
                        # Direct match weight
                        weight = 2.0 if ";->" in ref else 1.0
                        scores[domain] += weight
                        matched[domain].append(ref)
                        break

        # Check call graph callees if call graph is available
        if cg is not None:
            for method in cf.methods:
                m_sig = f"L{cf.this_class_name};->{method.name(cf.constant_pool)}{method.descriptor(cf.constant_pool)}"
                callees = cg.get_callees(m_sig)
                for callee in callees:
                    for domain, signatures in DOMAIN_SIGNATURES.items():
                        for sig in signatures:
                            if sig in callee:
                                scores[domain] += 0.5
                                break

        # Determine primary domain
        best_domain = None
        best_score = 0.0
        for dom, sc in scores.items():
            if sc > best_score:
                best_score = sc
                best_domain = dom

        is_sec = scores[CapabilityDomain.CRYPTOGRAPHY] > 0 or \
                 scores[CapabilityDomain.SECURITY_POLICY] > 0 or \
                 scores[CapabilityDomain.REFLECTION] > 0 or \
                 scores[CapabilityDomain.PROCESS_ENV] > 0

        # Calculate confidence normalized across scores
        total_score = sum(scores.values())
        confidence = (best_score / total_score) if total_score > 0 else 0.0

        return ClassTriageResult(
            class_name=class_name,
            primary_domain=best_domain if best_score >= self.sensitivity_threshold else None,
            confidence=round(confidence, 3),
            domain_scores={d: round(s, 2) for d, s in scores.items() if s > 0},
            matched_apis=dict(matched),
            is_security_sensitive=is_sec
        )

    def triage_archive(
        self,
        archive: JarArchive,
        cg: Optional[CallGraph] = None
    ) -> ArchiveTriageReport:
        """Performs whole-archive triage classification and ranks priority classes."""
        report = ArchiveTriageReport(total_classes=len(archive.classes))

        for c_name in archive.classes:
            res = self.triage_class(c_name, archive, cg)
            report.triage_results[c_name] = res
            if res.primary_domain:
                report.domain_distribution[res.primary_domain] += 1

        # High priority classes: security sensitive or high confidence capabilities
        high_pri = [
            c_name for c_name, res in report.triage_results.items()
            if res.is_security_sensitive or (res.confidence > 0.6 and res.primary_domain is not None)
        ]
        # Sort by total score descending
        high_pri.sort(
            key=lambda c: sum(report.triage_results[c].domain_scores.values()),
            reverse=True
        )
        report.high_priority_classes = high_pri

        return report
