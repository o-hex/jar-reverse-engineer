"""
jar_engine.sanitizer.synthetic: Duplicate synthetic member deduplication and bridge cycle breaker (JVMS §4.5, §4.6).

Resolves duplicate field/method declarations (such as fields with the same name but differing descriptors,
or identical duplicate synthetic members) and breaks self-recursive synthetic bridge methods that
cause decompilers to choke or generate invalid source definitions.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict

from jar_engine.model.class_file import ClassFile
from jar_engine.model.member_info import FieldInfo, MethodInfo
from jar_engine.model.constants import AccessFlags, Opcode


def deduplicate_fields(cf: ClassFile) -> int:
    """
    Resolves conflicting field names in a class.
    Bytecode allows multiple fields with the same name if descriptors differ,
    but Java source decompilers cannot represent this without compiler/syntax errors.
    Renames duplicate secondary fields by appending a unique index suffix.
    Returns the count of deduplicated/renamed fields.
    """
    cp = cf.constant_pool
    name_groups: Dict[str, List[FieldInfo]] = defaultdict(list)
    for f in cf.fields:
        name_groups[f.name(cp)].append(f)

    renamed_count = 0
    for name, fields in name_groups.items():
        if len(fields) <= 1:
            continue

        # Sort so non-synthetic fields come first (preserved), synthetic fields are renamed
        fields_sorted = sorted(fields, key=lambda x: (x.is_synthetic, x.descriptor(cp)))
        # Keep the first field as-is; rename the subsequent duplicates
        for idx, f in enumerate(fields_sorted[1:], start=1):
            new_name = f"{name}$dup${idx}"
            new_name_idx = cp.add_utf8(new_name)
            f.name_index = new_name_idx
            renamed_count += 1

    return renamed_count


def deduplicate_methods(cf: ClassFile) -> int:
    """
    Removes redundant duplicate methods sharing identical (name, descriptor) signatures.
    If both exist, the non-synthetic implementation is preserved and the synthetic duplicate pruned.
    Returns the count of pruned duplicate methods.
    """
    cp = cf.constant_pool
    sig_groups: Dict[Tuple[str, str], List[MethodInfo]] = defaultdict(list)
    for m in cf.methods:
        key = (m.name(cp), m.descriptor(cp))
        sig_groups[key].append(m)

    pruned_count = 0
    kept_methods: List[MethodInfo] = []

    for key, methods in sig_groups.items():
        if len(methods) == 1:
            kept_methods.append(methods[0])
            continue

        # Prefer non-synthetic over synthetic; if both synthetic, prefer the one with code
        def sort_key(m: MethodInfo) -> Tuple[int, int]:
            has_code = 1 if m.code_attribute(cp) is not None else 0
            is_non_synthetic = 0 if m.is_synthetic else 1
            return (is_non_synthetic, has_code)

        best_method = max(methods, key=sort_key)
        kept_methods.append(best_method)
        pruned_count += len(methods) - 1

    cf.methods = kept_methods
    return pruned_count


def break_recursive_bridge_methods(cf: ClassFile) -> int:
    """
    Detects and breaks synthetic bridge methods that make a direct self-call with no target
    (e.g., bridge method foo() calling foo() via invokevirtual on 'this', resulting in infinite recursion).
    Strips the self-call or marks the method abstract / returns default to prevent decompiler infinite loops.
    """
    cp = cf.constant_pool
    broken_cycles = 0

    for m in cf.methods:
        if not m.is_bridge and not m.is_synthetic:
            continue

        code_attr = m.code_attribute(cp)
        if code_attr is None or len(code_attr.code) == 0:
            continue

        code = code_attr.code
        # Minimal pattern: aload_0, (optional args load), invokevirtual/invokespecial (this_class, same_name, same_desc)
        # Check if the method makes an invokevirtual to itself on this class
        m_name = m.name(cp)
        m_desc = m.descriptor(cp)

        # Scan for invoke instructions
        idx = 0
        while idx < len(code):
            op = code[idx]
            if op in (Opcode.INVOKEVIRTUAL, Opcode.INVOKESPECIAL) and idx + 2 < len(code):
                import struct
                cp_idx = struct.unpack(">H", code[idx + 1:idx + 3])[0]
                try:
                    c_name, c_mname, c_mdesc = cp.resolve_member_ref(cp_idx)
                    if c_name == cf.this_class_name and c_mname == m_name and c_mdesc == m_desc:
                        # Direct self-recursive call in a synthetic bridge method!
                        # Break recursion by replacing invoke with return or nop
                        broken_cycles += 1
                        # Return type handling for bridge cycle breaking
                        ret_desc = m_desc.split(")")[-1]
                        if ret_desc == "V":
                            code_attr.code = bytes([Opcode.RETURN])
                        elif ret_desc.startswith("L") or ret_desc.startswith("["):
                            code_attr.code = bytes([Opcode.ACONST_NULL, Opcode.ARETURN])
                        elif ret_desc in ("J", "D"):
                            code_attr.code = bytes([Opcode.LCONST_0 if ret_desc == "J" else Opcode.DCONST_0, Opcode.LRETURN if ret_desc == "J" else Opcode.DRETURN])
                        elif ret_desc == "F":
                            code_attr.code = bytes([Opcode.FCONST_0, Opcode.FRETURN])
                        else:
                            code_attr.code = bytes([Opcode.ICONST_0, Opcode.IRETURN])
                        break
                except Exception:
                    pass
            idx += 1

    return broken_cycles


def sanitize_synthetic_members(cf: ClassFile) -> Tuple[int, int]:
    """
    Runs field deduplication, method deduplication, and recursive bridge cycle breaking.
    Returns (fields_modified, methods_modified).
    """
    f_count = deduplicate_fields(cf)
    m_count = deduplicate_methods(cf) + break_recursive_bridge_methods(cf)
    return (f_count, m_count)
