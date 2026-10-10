"""
jar_engine.triage.contextual_slice: LLM-friendly high-density contextual slicing (Features F22, F23).

Extracts token-efficient, high-signal structural slices (JSON or Markdown) of classes,
methods, and execution sinks for direct consumption by Large Language Models.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Any
import json

from jar_engine.model.class_file import ClassFile
from jar_engine.model.jar_archive import JarArchive
from jar_engine.model.cp_info import (
    CONSTANT_String_info, CONSTANT_Methodref_info, CONSTANT_Fieldref_info
)
from jar_engine.walker.call_graph import CallGraph
from .scorer import CapabilityTriager, ClassTriageResult


class ContextualSlicer:
    """
    Generates token-budgeted, concise structural representations tailored for LLM context windows.
    """

    def __init__(self, triager: Optional[CapabilityTriager] = None):
        self.triager = triager or CapabilityTriager()

    def _extract_string_literals(self, cf: ClassFile) -> List[str]:
        """Extracts constant string literals found in the class constant pool."""
        cp = cf.constant_pool
        strings: List[str] = []
        for entry in cp._entries:
            if isinstance(entry, CONSTANT_String_info):
                try:
                    s_val = cp.resolve_utf8(entry.string_index)
                    if s_val and len(s_val.strip()) > 0:
                        strings.append(s_val)
                except Exception:
                    pass
        return strings[:20]  # Cap to prevent unbounded token growth

    def generate_class_summary_dict(
        self,
        class_name: str,
        archive: JarArchive,
        cg: Optional[CallGraph] = None
    ) -> Dict[str, Any]:
        """Builds a compact summary dictionary for an individual class."""
        cf = archive.get_class(class_name)
        if cf is None:
            return {"error": f"Class {class_name} not found"}

        cp = cf.constant_pool
        triage_info = self.triager.triage_class(class_name, archive, cg)

        fields_summary = []
        for f in cf.fields:
            fields_summary.append({
                "name": f.name(cp),
                "descriptor": f.descriptor(cp),
                "static": f.is_static,
                "final": f.is_final
            })

        methods_summary = []
        for m in cf.methods:
            m_sig = f"L{class_name};->{m.name(cp)}{m.descriptor(cp)}"
            callees = list(cg.get_callees(m_sig))[:10] if cg else []
            methods_summary.append({
                "name": m.name(cp),
                "descriptor": m.descriptor(cp),
                "static": m.is_static,
                "callees": [c.split("->")[-1] for c in callees]
            })

        return {
            "class": class_name,
            "super_class": cf.super_class_name,
            "interfaces": cf.interface_names,
            "triage": {
                "domain": triage_info.primary_domain.value if triage_info.primary_domain else "Unclassified",
                "confidence": triage_info.confidence,
                "security_sensitive": triage_info.is_security_sensitive,
            },
            "fields": fields_summary,
            "methods": methods_summary,
            "string_literals": self._extract_string_literals(cf)
        }

    def generate_llm_slice(
        self,
        class_name: str,
        archive: JarArchive,
        cg: Optional[CallGraph] = None,
        format: str = "json"
    ) -> str:
        """
        Exports a high-density, token-compressed slice formatted for an LLM prompt.
        Supported formats: "json", "markdown".
        """
        data = self.generate_class_summary_dict(class_name, archive, cg)

        if format.lower() == "json":
            return json.dumps(data, indent=2)

        # Markdown representation
        md_lines = [
            f"# Class: `{data['class']}`",
            f"- **Superclass**: `{data['super_class']}`",
            f"- **Interfaces**: {', '.join(f'`{i}`' for i in data['interfaces']) or 'None'}",
            f"- **Inferred Domain**: **{data['triage']['domain']}** (Confidence: {data['triage']['confidence']})",
            f"- **Security Sensitive**: {'Yes' if data['triage']['security_sensitive'] else 'No'}",
            "",
            "## Fields",
        ]
        for f in data["fields"]:
            flags = " ".join([k for k in ["static", "final"] if f.get(k)])
            md_lines.append(f"- `{f['name']}`: `{f['descriptor']}` {f'({flags})' if flags else ''}")

        md_lines.append("")
        md_lines.append("## Methods & Key Invocations")
        for m in data["methods"]:
            callees_str = f" -> calls: {', '.join(m['callees'])}" if m.get("callees") else ""
            md_lines.append(f"- `{m['name']}{m['descriptor']}`{callees_str}")

        if data.get("string_literals"):
            md_lines.append("")
            md_lines.append("## Embedded Strings")
            for s in data["string_literals"][:10]:
                clean_s = s.replace("\n", "\\n").replace("\r", "\\r")
                md_lines.append(f"- `\"{clean_s}\"`")

        return "\n".join(md_lines)

    def generate_sink_slice(
        self,
        sink_pattern: str,
        archive: JarArchive,
        cg: CallGraph
    ) -> Dict[str, Any]:
        """
        Isolates execution slices reaching a security sink (e.g. 'Cipher;->doFinal' or 'Socket').
        Traces callers backward to discover entry points.
        """
        matching_methods: List[str] = []
        for (caller, callee) in cg.graph.edges():
            if sink_pattern in callee:
                matching_methods.append(caller)

        root_callers = set()
        for m in matching_methods:
            callers = cg.get_callers(m)
            root_callers.update(callers or [m])

        return {
            "sink_pattern": sink_pattern,
            "direct_callers": matching_methods,
            "entrypoint_candidates": list(root_callers)
        }
