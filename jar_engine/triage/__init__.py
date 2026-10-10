"""
jar_engine.triage: Subsystem capability triage and LLM-centric contextual slicing.

F20: 10-domain architectural capability rules (taxonomy.py).
F21: Multi-factor triage scoring & classification (scorer.py).
F22: LLM-friendly high-density JSON/Markdown slicing (contextual_slice.py).
F23: Security sink/source backward reachability slicing.
"""

from __future__ import annotations

from .taxonomy import CapabilityDomain, DOMAIN_SIGNATURES
from .scorer import CapabilityTriager, ClassTriageResult, ArchiveTriageReport
from .contextual_slice import ContextualSlicer

__all__ = [
    "CapabilityDomain",
    "DOMAIN_SIGNATURES",
    "CapabilityTriager",
    "ClassTriageResult",
    "ArchiveTriageReport",
    "ContextualSlicer",
]
