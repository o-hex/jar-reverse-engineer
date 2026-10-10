"""
jar_engine.sanitizer.stack_map: StackMapTable verification frame validator and normalizer (JVMS §4.7.4).

Validates verification frame deltas, verification_type_info invariants, and constant pool
class references. Strips corrupted or malformed StackMapTable attributes that cause
decompiler crashes in CFR, Vineflower, and Procyon.
"""

from __future__ import annotations
from typing import Optional, List, Tuple
import struct
import io

from jar_engine.model.class_file import ClassFile
from jar_engine.model.member_info import (
    MethodInfo, CodeAttribute, StackMapTableAttribute, AttributeInfo
)
from jar_engine.model.cp_info import ConstantPool, CONSTANT_Class_info


def validate_verification_type_info(stream: io.BytesIO, code_len: int, cp: Optional[ConstantPool]) -> bool:
    """Validates an individual verification_type_info entry per JVMS §4.7.4."""
    tag_bytes = stream.read(1)
    if not tag_bytes:
        return False
    tag = tag_bytes[0]
    if 0 <= tag <= 6:
        # Top, Integer, Float, Double, Long, Null, UninitializedThis
        return True
    elif tag == 7:
        # Object_variable_info: u2 cpool_index
        idx_bytes = stream.read(2)
        if len(idx_bytes) < 2:
            return False
        cpool_idx = struct.unpack(">H", idx_bytes)[0]
        if cp is not None:
            if cpool_idx <= 0 or cpool_idx >= len(cp):
                return False
            if not isinstance(cp._entries[cpool_idx], CONSTANT_Class_info):
                return False
        return True
    elif tag == 8:
        # Uninitialized_variable_info: u2 offset
        off_bytes = stream.read(2)
        if len(off_bytes) < 2:
            return False
        offset = struct.unpack(">H", off_bytes)[0]
        return offset < code_len
    return False


def validate_stack_map_table(entries_data: bytes, code_len: int, cp: Optional[ConstantPool]) -> bool:
    """
    Parses and validates a StackMapTable attribute's binary payload per JVMS §4.7.4.
    Returns True if fully valid and compliant, False if corrupted.
    """
    if len(entries_data) < 2:
        return False

    stream = io.BytesIO(entries_data)
    num_entries = struct.unpack(">H", stream.read(2))[0]
    current_pc = -1

    try:
        for _ in range(num_entries):
            frame_type_b = stream.read(1)
            if not frame_type_b:
                return False
            ft = frame_type_b[0]

            if ft <= 63:
                # same_frame
                offset_delta = ft
            elif ft <= 127:
                # same_locals_1_stack_item_frame
                offset_delta = ft - 64
                if not validate_verification_type_info(stream, code_len, cp):
                    return False
            elif 128 <= ft <= 246:
                # Reserved frame tags
                return False
            elif ft == 247:
                # same_locals_1_stack_item_frame_extended
                delta_b = stream.read(2)
                if len(delta_b) < 2:
                    return False
                offset_delta = struct.unpack(">H", delta_b)[0]
                if not validate_verification_type_info(stream, code_len, cp):
                    return False
            elif 248 <= ft <= 250:
                # chop_frame
                delta_b = stream.read(2)
                if len(delta_b) < 2:
                    return False
                offset_delta = struct.unpack(">H", delta_b)[0]
            elif ft == 251:
                # same_frame_extended
                delta_b = stream.read(2)
                if len(delta_b) < 2:
                    return False
                offset_delta = struct.unpack(">H", delta_b)[0]
            elif 252 <= ft <= 254:
                # append_frame
                delta_b = stream.read(2)
                if len(delta_b) < 2:
                    return False
                offset_delta = struct.unpack(">H", delta_b)[0]
                num_locals = ft - 251
                for _ in range(num_locals):
                    if not validate_verification_type_info(stream, code_len, cp):
                        return False
            elif ft == 255:
                # full_frame
                delta_b = stream.read(2)
                if len(delta_b) < 2:
                    return False
                offset_delta = struct.unpack(">H", delta_b)[0]
                # locals
                loc_count_b = stream.read(2)
                if len(loc_count_b) < 2:
                    return False
                num_loc = struct.unpack(">H", loc_count_b)[0]
                for _ in range(num_loc):
                    if not validate_verification_type_info(stream, code_len, cp):
                        return False
                # stack
                stk_count_b = stream.read(2)
                if len(stk_count_b) < 2:
                    return False
                num_stk = struct.unpack(">H", stk_count_b)[0]
                for _ in range(num_stk):
                    if not validate_verification_type_info(stream, code_len, cp):
                        return False
            else:
                return False

            if current_pc == -1:
                current_pc = offset_delta
            else:
                current_pc += offset_delta + 1

            if current_pc >= code_len and code_len > 0:
                return False

        # Must have consumed entire payload (no trailing junk)
        if stream.tell() != len(entries_data):
            return False

        return True
    except Exception:
        return False


def sanitize_class_stack_maps(cf: ClassFile, force_strip: bool = False) -> int:
    """
    Sanitizes StackMapTable attributes across all methods in a ClassFile.
    - For Java <= 6 (major_version <= 50): StackMapTable is stripped (legacy classes).
    - For Java >= 7: validates frame data; strips corrupted frames that break decompilers.
    Returns total count of stripped or repaired StackMapTable attributes.
    """
    stripped_count = 0
    cp = cf.constant_pool
    is_legacy_java = cf.major_version <= 50

    for method in cf.methods:
        code_attr = method.code_attribute(cp)
        if code_attr is None:
            continue

        new_code_attrs: List[AttributeInfo] = []
        for attr in code_attr.attributes:
            is_stack_map = False
            entries_bytes = b""

            if isinstance(attr, StackMapTableAttribute):
                is_stack_map = True
                entries_bytes = attr.entries_data
            elif hasattr(attr, "name") and attr.name(cp) == "StackMapTable":
                is_stack_map = True
                entries_bytes = getattr(attr, "info", getattr(attr, "entries_data", b""))

            if is_stack_map:
                if force_strip or is_legacy_java:
                    stripped_count += 1
                    continue

                # Validate frame binary structure
                if not validate_stack_map_table(entries_bytes, len(code_attr.code), cp):
                    stripped_count += 1
                    continue

            new_code_attrs.append(attr)
        code_attr.attributes = new_code_attrs

    return stripped_count
