"""
jar_engine.sanitizer.exceptions: Exception handler table and checked exceptions normalizer.

Repairs inverted ranges (start_pc >= end_pc), out-of-bounds PCs, self-catching recursive loops,
and invalid constant pool references in Code attribute exception tables (JVMS §4.7.3)
and method Exceptions attributes (JVMS §4.7.5).
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Set
from jar_engine.model.class_file import ClassFile
from jar_engine.model.member_info import (
    MethodInfo, CodeAttribute, ExceptionTableEntry, ExceptionsAttribute, AttributeInfo
)
from jar_engine.model.cp_info import CONSTANT_Class_info


def sanitize_code_exceptions(code_attr: CodeAttribute, cp: Optional[object] = None) -> int:
    """
    Sanitizes the exception_table of a CodeAttribute.
    Removes:
    - Out of bounds entries (start_pc < 0, end_pc > len(code), handler_pc >= len(code))
    - Inverted or empty ranges (start_pc >= end_pc)
    - Self-catching recursive loops where handler_pc falls within [start_pc, end_pc)
    - Duplicate identical entries
    Returns the count of removed or repaired exception table entries.
    """
    code_len = len(code_attr.code)
    cleaned_table: List[ExceptionTableEntry] = []
    seen_entries: Set[Tuple[int, int, int, int]] = set()
    modifications = 0

    for entry in code_attr.exception_table:
        s = entry.start_pc
        e = entry.end_pc
        h = entry.handler_pc
        c = entry.catch_type

        # 1. Bounds check
        if s < 0 or e > code_len or h < 0 or h >= code_len:
            modifications += 1
            continue

        # 2. Inverted or zero-length range
        if s >= e:
            modifications += 1
            continue

        # 3. Self-catching recursive loop: handler PC is within the protected range
        # If an exception occurs in the handler, it recursively jumps back to itself.
        # Obfuscators use this to break decompiler control-flow reconstruction.
        if s <= h < e:
            # If the handler is at the start, shrink protected range to (h+1, e) if valid,
            # or discard if trivial recursion.
            if h == s:
                # Can't shrink meaningfully if handler is first instruction of protected block
                modifications += 1
                continue
            else:
                # Truncate protected range to end right before handler_pc
                e = h
                if s >= e:
                    modifications += 1
                    continue
                entry = ExceptionTableEntry(s, e, h, c)
                modifications += 1

        # 4. Catch type validity (if constant pool provided)
        if cp is not None and c != 0:
            try:
                entry_tag = getattr(cp, "_entries", [])
                if c >= len(entry_tag) or not isinstance(entry_tag[c], CONSTANT_Class_info):
                    # Invalid class index in constant pool -> strip entry
                    modifications += 1
                    continue
            except Exception:
                modifications += 1
                continue

        # 5. Deduplicate
        key = (entry.start_pc, entry.end_pc, entry.handler_pc, entry.catch_type)
        if key in seen_entries:
            modifications += 1
            continue
        seen_entries.add(key)

        cleaned_table.append(entry)

    code_attr.exception_table = cleaned_table
    return modifications


def sanitize_method_exceptions_attribute(method: MethodInfo, cf: ClassFile) -> int:
    """
    Validates ExceptionsAttribute (throws clause). Removes invalid constant pool class indices.
    """
    modifications = 0
    cp = cf.constant_pool
    new_attrs: List[AttributeInfo] = []

    for attr in method.attributes:
        if isinstance(attr, ExceptionsAttribute):
            valid_indices: List[int] = []
            for idx in attr.exception_index_table:
                if 0 < idx < len(cp):
                    entry = cp._entries[idx]
                    if isinstance(entry, CONSTANT_Class_info):
                        valid_indices.append(idx)
                        continue
                modifications += 1
            if len(valid_indices) != len(attr.exception_index_table):
                attr.exception_index_table = valid_indices
        new_attrs.append(attr)

    method.attributes = new_attrs
    return modifications


def sanitize_class_exceptions(cf: ClassFile) -> int:
    """
    Sanitizes all exception tables across all methods in a ClassFile.
    Returns total count of fixed/removed irregular exception entries.
    """
    total_fixes = 0
    cp = cf.constant_pool

    for method in cf.methods:
        code_attr = method.code_attribute(cp)
        if code_attr is not None:
            total_fixes += sanitize_code_exceptions(code_attr, cp)
        total_fixes += sanitize_method_exceptions_attribute(method, cf)

    return total_fixes
