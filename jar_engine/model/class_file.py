"""
jar_engine.model.class_file: JVMS §4 class binary parser, serializer, and inspection model.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, BinaryIO, Union
import struct
import io

from .constants import JAVA_MAGIC, ConstantTag, AccessFlags
from .cp_info import (
    ConstantPool, CpEntry, CpPlaceholder,
    CONSTANT_Utf8_info, CONSTANT_Integer_info, CONSTANT_Float_info,
    CONSTANT_Long_info, CONSTANT_Double_info, CONSTANT_Class_info,
    CONSTANT_String_info, CONSTANT_Fieldref_info, CONSTANT_Methodref_info,
    CONSTANT_InterfaceMethodref_info, CONSTANT_NameAndType_info,
    CONSTANT_MethodHandle_info, CONSTANT_MethodType_info,
    CONSTANT_Dynamic_info, CONSTANT_InvokeDynamic_info,
    CONSTANT_Module_info, CONSTANT_Package_info,
    decode_modified_utf8
)
from .member_info import (
    FieldInfo, MethodInfo, AttributeInfo, parse_attribute
)


class ClassFormatError(ValueError):
    """Raised when bytecode data does not conform to JVMS §4 class file structure."""
    pass


@dataclass
class ClassFile:
    """Represents a compiled JVM class file conforming to JVMS §4.1."""
    magic: int = JAVA_MAGIC
    minor_version: int = 0
    major_version: int = 52
    constant_pool: ConstantPool = field(default_factory=ConstantPool)
    access_flags: int = 0
    this_class: int = 0
    super_class: int = 0
    interfaces: List[int] = field(default_factory=list)
    fields: List[FieldInfo] = field(default_factory=list)
    methods: List[MethodInfo] = field(default_factory=list)
    attributes: List[AttributeInfo] = field(default_factory=list)

    # --- High-Level Identification Properties ---
    @property
    def this_class_name(self) -> str:
        return self.constant_pool.resolve_class_name(self.this_class)

    @property
    def super_class_name(self) -> Optional[str]:
        if self.super_class == 0:
            return None
        return self.constant_pool.resolve_class_name(self.super_class)

    @property
    def interface_names(self) -> List[str]:
        return [self.constant_pool.resolve_class_name(idx) for idx in self.interfaces]

    @property
    def is_public(self) -> bool:
        return bool(self.access_flags & AccessFlags.PUBLIC)

    @property
    def is_final(self) -> bool:
        return bool(self.access_flags & AccessFlags.FINAL)

    @property
    def is_interface(self) -> bool:
        return bool(self.access_flags & AccessFlags.INTERFACE)

    @property
    def is_abstract(self) -> bool:
        return bool(self.access_flags & AccessFlags.ABSTRACT)

    @property
    def is_enum(self) -> bool:
        return bool(self.access_flags & AccessFlags.ENUM)

    @property
    def is_synthetic(self) -> bool:
        return bool(self.access_flags & AccessFlags.SYNTHETIC)

    @property
    def is_annotation(self) -> bool:
        return bool(self.access_flags & AccessFlags.ANNOTATION)

    @property
    def is_module(self) -> bool:
        return bool(self.access_flags & AccessFlags.MODULE)

    # --- Lookup Helpers ---
    def get_field(self, name: str, descriptor: Optional[str] = None) -> Optional[FieldInfo]:
        for f in self.fields:
            if f.name(self.constant_pool) == name:
                if descriptor is None or f.descriptor(self.constant_pool) == descriptor:
                    return f
        return None

    def get_method(self, name: str, descriptor: Optional[str] = None) -> Optional[MethodInfo]:
        for m in self.methods:
            if m.name(self.constant_pool) == name:
                if descriptor is None or m.descriptor(self.constant_pool) == descriptor:
                    return m
        return None

    def find_methods(self, name: str) -> List[MethodInfo]:
        return [m for m in self.methods if m.name(self.constant_pool) == name]

    def get_attribute(self, name: str) -> Optional[AttributeInfo]:
        for attr in self.attributes:
            try:
                if attr.name(self.constant_pool) == name:
                    return attr
            except Exception:
                continue
        return None

    # --- Constant Pool Shortcuts ---
    def resolve_utf8(self, index: int) -> str:
        return self.constant_pool.resolve_utf8(index)

    def resolve_class_name(self, index: int) -> str:
        return self.constant_pool.resolve_class_name(index)

    def resolve_string(self, index: int) -> str:
        return self.constant_pool.resolve_string(index)

    def resolve_name_and_type(self, index: int) -> Tuple[str, str]:
        return self.constant_pool.resolve_name_and_type(index)

    def resolve_member_ref(self, index: int) -> Tuple[str, str, str]:
        return self.constant_pool.resolve_member_ref(index)

    def add_utf8(self, value: str) -> int:
        return self.constant_pool.add_utf8(value)

    def add_class(self, class_name: str) -> int:
        return self.constant_pool.add_class(class_name)

    def add_name_and_type(self, name: str, descriptor: str) -> int:
        return self.constant_pool.add_name_and_type(name, descriptor)

    def add_member_ref(self, tag: ConstantTag, class_name: str, name: str, descriptor: str) -> int:
        return self.constant_pool.add_member_ref(tag, class_name, name, descriptor)

    def add_string(self, value: str) -> int:
        return self.constant_pool.add_string(value)

    def update_utf8(self, index: int, new_value: str) -> None:
        self.constant_pool.update_utf8(index, new_value)

    # --- Binary Parser ---
    @classmethod
    def parse(cls, data: bytes) -> "ClassFile":
        stream = io.BytesIO(data)
        header = stream.read(10)
        if len(header) < 10:
            raise ClassFormatError(f"File too short for JVM class header: {len(data)} bytes")
        magic, minor, major, cp_count = struct.unpack(">IHHH", header)
        if magic != JAVA_MAGIC:
            raise ClassFormatError(f"Invalid magic: {hex(magic)}, expected 0xCAFEBABE")

        # Parse Constant Pool (1-indexed, size cp_count)
        cp_entries: List[Optional[CpEntry]] = [None] * cp_count
        idx = 1
        while idx < cp_count:
            tag_b = stream.read(1)
            if not tag_b:
                raise ClassFormatError(f"Unexpected EOF reading CP tag at index {idx}")
            tag = tag_b[0]

            if tag == ConstantTag.UTF8:
                len_bytes = stream.read(2)
                if len(len_bytes) < 2:
                    raise ClassFormatError(f"Truncated UTF8 length at CP index {idx}")
                l = struct.unpack(">H", len_bytes)[0]
                raw = stream.read(l)
                if len(raw) < l:
                    raise ClassFormatError(f"Truncated UTF8 payload at CP index {idx}")
                val = decode_modified_utf8(raw)
                cp_entries[idx] = CONSTANT_Utf8_info(tag=tag, value=val, raw_bytes=raw)
                idx += 1
            elif tag == ConstantTag.INTEGER:
                b = stream.read(4)
                if len(b) < 4:
                    raise ClassFormatError(f"Truncated Integer at CP index {idx}")
                val = struct.unpack(">i", b)[0]
                cp_entries[idx] = CONSTANT_Integer_info(tag=tag, value=val)
                idx += 1
            elif tag == ConstantTag.FLOAT:
                b = stream.read(4)
                if len(b) < 4:
                    raise ClassFormatError(f"Truncated Float at CP index {idx}")
                raw_bits = struct.unpack(">I", b)[0]
                f_val = struct.unpack(">f", b)[0]
                cp_entries[idx] = CONSTANT_Float_info(tag=tag, value=f_val, raw_bits=raw_bits)
                idx += 1
            elif tag == ConstantTag.LONG:
                b = stream.read(8)
                if len(b) < 8:
                    raise ClassFormatError(f"Truncated Long at CP index {idx}")
                val = struct.unpack(">q", b)[0]
                cp_entries[idx] = CONSTANT_Long_info(tag=tag, value=val)
                if idx + 1 < cp_count:
                    cp_entries[idx + 1] = CpPlaceholder()
                idx += 2
            elif tag == ConstantTag.DOUBLE:
                b = stream.read(8)
                if len(b) < 8:
                    raise ClassFormatError(f"Truncated Double at CP index {idx}")
                raw_bits = struct.unpack(">Q", b)[0]
                d_val = struct.unpack(">d", b)[0]
                cp_entries[idx] = CONSTANT_Double_info(tag=tag, value=d_val, raw_bits=raw_bits)
                if idx + 1 < cp_count:
                    cp_entries[idx + 1] = CpPlaceholder()
                idx += 2
            elif tag == ConstantTag.CLASS:
                b = stream.read(2)
                if len(b) < 2:
                    raise ClassFormatError(f"Truncated Class at CP index {idx}")
                n_idx = struct.unpack(">H", b)[0]
                cp_entries[idx] = CONSTANT_Class_info(tag=tag, name_index=n_idx)
                idx += 1
            elif tag == ConstantTag.STRING:
                b = stream.read(2)
                if len(b) < 2:
                    raise ClassFormatError(f"Truncated String at CP index {idx}")
                s_idx = struct.unpack(">H", b)[0]
                cp_entries[idx] = CONSTANT_String_info(tag=tag, string_index=s_idx)
                idx += 1
            elif tag in (ConstantTag.FIELDREF, ConstantTag.METHODREF, ConstantTag.INTERFACE_METHODREF):
                b = stream.read(4)
                if len(b) < 4:
                    raise ClassFormatError(f"Truncated MemberRef at CP index {idx}")
                c_idx, nat_idx = struct.unpack(">HH", b)
                cls_map = {
                    ConstantTag.FIELDREF: CONSTANT_Fieldref_info,
                    ConstantTag.METHODREF: CONSTANT_Methodref_info,
                    ConstantTag.INTERFACE_METHODREF: CONSTANT_InterfaceMethodref_info,
                }
                cp_entries[idx] = cls_map[tag](tag=tag, class_index=c_idx, name_and_type_index=nat_idx)
                idx += 1
            elif tag == ConstantTag.NAME_AND_TYPE:
                b = stream.read(4)
                if len(b) < 4:
                    raise ClassFormatError(f"Truncated NameAndType at CP index {idx}")
                n_idx, d_idx = struct.unpack(">HH", b)
                cp_entries[idx] = CONSTANT_NameAndType_info(tag=tag, name_index=n_idx, descriptor_index=d_idx)
                idx += 1
            elif tag == ConstantTag.METHOD_HANDLE:
                b = stream.read(3)
                if len(b) < 3:
                    raise ClassFormatError(f"Truncated MethodHandle at CP index {idx}")
                rk, ref_idx = struct.unpack(">BH", b)
                cp_entries[idx] = CONSTANT_MethodHandle_info(tag=tag, reference_kind=rk, reference_index=ref_idx)
                idx += 1
            elif tag == ConstantTag.METHOD_TYPE:
                b = stream.read(2)
                if len(b) < 2:
                    raise ClassFormatError(f"Truncated MethodType at CP index {idx}")
                d_idx = struct.unpack(">H", b)[0]
                cp_entries[idx] = CONSTANT_MethodType_info(tag=tag, descriptor_index=d_idx)
                idx += 1
            elif tag in (ConstantTag.DYNAMIC, ConstantTag.INVOKE_DYNAMIC):
                b = stream.read(4)
                if len(b) < 4:
                    raise ClassFormatError(f"Truncated Dynamic at CP index {idx}")
                bsm_idx, nat_idx = struct.unpack(">HH", b)
                entry_cls = CONSTANT_Dynamic_info if tag == ConstantTag.DYNAMIC else CONSTANT_InvokeDynamic_info
                cp_entries[idx] = entry_cls(tag=tag, bootstrap_method_attr_index=bsm_idx, name_and_type_index=nat_idx)
                idx += 1
            elif tag == ConstantTag.MODULE:
                b = stream.read(2)
                if len(b) < 2:
                    raise ClassFormatError(f"Truncated Module at CP index {idx}")
                n_idx = struct.unpack(">H", b)[0]
                cp_entries[idx] = CONSTANT_Module_info(tag=tag, name_index=n_idx)
                idx += 1
            elif tag == ConstantTag.PACKAGE:
                b = stream.read(2)
                if len(b) < 2:
                    raise ClassFormatError(f"Truncated Package at CP index {idx}")
                n_idx = struct.unpack(">H", b)[0]
                cp_entries[idx] = CONSTANT_Package_info(tag=tag, name_index=n_idx)
                idx += 1
            else:
                raise ClassFormatError(f"Invalid constant pool tag: {tag} at CP index {idx}")

        # Construct ConstantPool object
        # _entries in ConstantPool starts at index 0. cp_entries already has None at index 0.
        cp = ConstantPool()
        cp._entries = cp_entries

        # Class Body
        body_hdr = stream.read(8)
        if len(body_hdr) < 8:
            raise ClassFormatError("Truncated class declaration header")
        acc, this_c, super_c, iface_count = struct.unpack(">HHHH", body_hdr)

        interfaces: List[int] = []
        for _ in range(iface_count):
            iface_bytes = stream.read(2)
            if len(iface_bytes) < 2:
                raise ClassFormatError("Truncated interfaces table")
            interfaces.append(struct.unpack(">H", iface_bytes)[0])

        # Fields
        f_count_bytes = stream.read(2)
        if len(f_count_bytes) < 2:
            raise ClassFormatError("Truncated fields count")
        f_count = struct.unpack(">H", f_count_bytes)[0]
        fields: List[FieldInfo] = []
        for _ in range(f_count):
            f_hdr = stream.read(8)
            if len(f_hdr) < 8:
                raise ClassFormatError("Truncated field header")
            f_acc, f_name, f_desc, f_ac = struct.unpack(">HHHH", f_hdr)
            try:
                f_attrs = [parse_attribute(stream, cp) for _ in range(f_ac)]
            except EOFError as e:
                raise ClassFormatError(f"Truncated field attribute: {e}") from e
            fields.append(FieldInfo(f_acc, f_name, f_desc, f_attrs))

        # Methods
        m_count_bytes = stream.read(2)
        if len(m_count_bytes) < 2:
            raise ClassFormatError("Truncated methods count")
        m_count = struct.unpack(">H", m_count_bytes)[0]
        methods: List[MethodInfo] = []
        for _ in range(m_count):
            m_hdr = stream.read(8)
            if len(m_hdr) < 8:
                raise ClassFormatError("Truncated method header")
            m_acc, m_name, m_desc, m_ac = struct.unpack(">HHHH", m_hdr)
            try:
                m_attrs = [parse_attribute(stream, cp) for _ in range(m_ac)]
            except EOFError as e:
                raise ClassFormatError(f"Truncated method attribute: {e}") from e
            methods.append(MethodInfo(m_acc, m_name, m_desc, m_attrs))

        # Attributes
        c_ac_bytes = stream.read(2)
        if len(c_ac_bytes) < 2:
            raise ClassFormatError("Truncated class attributes count")
        c_ac = struct.unpack(">H", c_ac_bytes)[0]
        try:
            c_attrs = [parse_attribute(stream, cp) for _ in range(c_ac)]
        except EOFError as e:
            raise ClassFormatError(f"Truncated class attribute: {e}") from e

        return cls(
            magic=magic,
            minor_version=minor,
            major_version=major,
            constant_pool=cp,
            access_flags=acc,
            this_class=this_c,
            super_class=super_c,
            interfaces=interfaces,
            fields=fields,
            methods=methods,
            attributes=c_attrs,
        )

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "ClassFile":
        with open(path, "rb") as f:
            return cls.parse(f.read())

    # --- Binary Serializer ---
    def to_bytes(self) -> bytes:
        out = io.BytesIO()
        out.write(struct.pack(">IHHH", self.magic, self.minor_version, self.major_version, len(self.constant_pool)))
        out.write(self.constant_pool.to_bytes())
        out.write(struct.pack(">HHHH", self.access_flags, self.this_class, self.super_class, len(self.interfaces)))
        for iface in self.interfaces:
            out.write(struct.pack(">H", iface))

        out.write(struct.pack(">H", len(self.fields)))
        for f in self.fields:
            out.write(f.to_bytes(self.constant_pool))

        out.write(struct.pack(">H", len(self.methods)))
        for m in self.methods:
            out.write(m.to_bytes(self.constant_pool))

        out.write(struct.pack(">H", len(self.attributes)))
        for attr in self.attributes:
            out.write(attr.to_bytes(self.constant_pool))

        return out.getvalue()

    def save(self, path: Union[str, Path]) -> None:
        with open(path, "wb") as f:
            f.write(self.to_bytes())
