"""
jar_engine.sanitizer: Bytecode preprocessing and decompiler compatibility sanitizer.

F8: StackMapTable frame repair / stripping.
F9: Generic signature validator & cleaner (JVMS §4.7.9).
F10: Exception handler range normalizer & cycle breaker (JVMS §4.7.3, §4.7.5).
F11: Duplicate synthetic member deduplication and bridge recursion breaker.
F12: Decompiler backend compatibility verification.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

from jar_engine.model.class_file import ClassFile
from jar_engine.model.jar_archive import JarArchive
from .signatures import sanitize_class_signatures, is_valid_class_signature, is_valid_method_signature, is_valid_field_signature
from .exceptions import sanitize_class_exceptions, sanitize_code_exceptions
from .stack_map import sanitize_class_stack_maps, validate_stack_map_table
from .synthetic import sanitize_synthetic_members, deduplicate_fields, deduplicate_methods, break_recursive_bridge_methods
from .decompiler_check import run_decompiler_check, verify_sanitized_class_structural_integrity


@dataclass
class SanitizationReport:
    """Detailed summary of repairs applied to an individual ClassFile."""
    class_name: str
    signatures_stripped: int = 0
    exceptions_fixed: int = 0
    stack_maps_sanitized: int = 0
    fields_deduplicated: int = 0
    methods_deduplicated: int = 0

    @property
    def total_modifications(self) -> int:
        return (
            self.signatures_stripped
            + self.exceptions_fixed
            + self.stack_maps_sanitized
            + self.fields_deduplicated
            + self.methods_deduplicated
        )

    @property
    def was_modified(self) -> bool:
        return self.total_modifications > 0


@dataclass
class ArchiveSanitizationReport:
    """Summary of sanitization passes across an entire JarArchive."""
    total_classes: int = 0
    classes_modified: int = 0
    total_signatures_stripped: int = 0
    total_exceptions_fixed: int = 0
    total_stack_maps_sanitized: int = 0
    total_fields_deduplicated: int = 0
    total_methods_deduplicated: int = 0
    reports: Dict[str, SanitizationReport] = field(default_factory=dict)


class BytecodeSanitizer:
    """
    Orchestrates bytecode sanitization passes to ensure decompiler stability and specification compliance.
    """

    def __init__(self, force_strip_stack_maps: bool = False):
        self.force_strip_stack_maps = force_strip_stack_maps

    def sanitize_class(
        self,
        cf: ClassFile,
        hierarchy: Optional[object] = None
    ) -> SanitizationReport:
        """
        Executes all sanitization passes on a single ClassFile.
        - F9: Strip malformed generic signatures
        - F10: Repair inverted/corrupted exception handler ranges & loops
        - F8: Validate/strip corrupted StackMapTable attributes
        - F11: Deduplicate synthetic fields/methods & break bridge recursion cycles
        """
        cls_name = cf.this_class_name

        # Pass 1: Signatures
        sig_count = sanitize_class_signatures(cf)

        # Pass 2: Exception handlers
        ex_count = sanitize_class_exceptions(cf)

        # Pass 3: StackMapTable
        sm_count = sanitize_class_stack_maps(cf, force_strip=self.force_strip_stack_maps)

        # Pass 4: Synthetic deduplication
        f_dedup, m_dedup = sanitize_synthetic_members(cf)

        return SanitizationReport(
            class_name=cls_name,
            signatures_stripped=sig_count,
            exceptions_fixed=ex_count,
            stack_maps_sanitized=sm_count,
            fields_deduplicated=f_dedup,
            methods_deduplicated=m_dedup
        )

    def sanitize_archive(self, archive: JarArchive) -> ArchiveSanitizationReport:
        """
        Executes sanitization across all classes in a JarArchive.
        """
        report = ArchiveSanitizationReport(total_classes=len(archive.classes))

        for cls_name, cf in archive.classes.items():
            cls_report = self.sanitize_class(cf)
            if cls_report.was_modified:
                report.classes_modified += 1
                report.total_signatures_stripped += cls_report.signatures_stripped
                report.total_exceptions_fixed += cls_report.exceptions_fixed
                report.total_stack_maps_sanitized += cls_report.stack_maps_sanitized
                report.total_fields_deduplicated += cls_report.fields_deduplicated
                report.total_methods_deduplicated += cls_report.methods_deduplicated
            report.reports[cls_name] = cls_report

        return report


__all__ = [
    "BytecodeSanitizer",
    "SanitizationReport",
    "ArchiveSanitizationReport",
    "sanitize_class_signatures",
    "sanitize_class_exceptions",
    "sanitize_class_stack_maps",
    "sanitize_synthetic_members",
    "run_decompiler_check",
    "verify_sanitized_class_structural_integrity",
]
