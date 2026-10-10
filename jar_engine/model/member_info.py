"""
jar_engine.model.member_info: Field, Method, and Attribute models conforming to JVMS §4.5, §4.6, §4.7.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, BinaryIO, Union
import struct
import io

from .constants import AccessFlags
from .cp_info import ConstantPool


@dataclass
class AttributeInfo:
    """Base class for JVM attributes (JVMS §4.7)."""
    attribute_name_index: int

    def name(self, cp: ConstantPool) -> str:
        return cp.resolve_utf8(self.attribute_name_index)

    def to_bytes(self, cp: Optional[ConstantPool] = None) -> bytes:
        raise NotImplementedError


@dataclass
class RawAttribute(AttributeInfo):
    """Fallback attribute preserving verbatim uninterpreted bytes."""
    info: bytes = b""

    def to_bytes(self, cp: Optional[ConstantPool] = None) -> bytes:
        return struct.pack(">HI", self.attribute_name_index, len(self.info)) + self.info


@dataclass
class ExceptionTableEntry:
    """Entry in a Code attribute exception_table (JVMS §4.7.3)."""
    start_pc: int
    end_pc: int
    handler_pc: int
    catch_type: int  # 0 for any/finally, or constant pool class index


@dataclass
class CodeAttribute(AttributeInfo):
    """Code attribute containing method bytecode instructions (JVMS §4.7.3)."""
    max_stack: int = 0
    max_locals: int = 0
    code: bytes = b""
    exception_table: List[ExceptionTableEntry] = field(default_factory=list)
    attributes: List[AttributeInfo] = field(default_factory=list)

    def get_attribute(self, name: str, cp: ConstantPool) -> Optional[AttributeInfo]:
        for attr in self.attributes:
            try:
                if attr.name(cp) == name:
                    return attr
            except Exception:
                continue
        return None

    def to_bytes(self, cp: Optional[ConstantPool] = None) -> bytes:
        body = io.BytesIO()
        body.write(struct.pack(">HHI", self.max_stack, self.max_locals, len(self.code)))
        body.write(self.code)
        body.write(struct.pack(">H", len(self.exception_table)))
        for ex in self.exception_table:
            body.write(struct.pack(">HHHH", ex.start_pc, ex.end_pc, ex.handler_pc, ex.catch_type))
        body.write(struct.pack(">H", len(self.attributes)))
        for attr in self.attributes:
            body.write(attr.to_bytes(cp))
        payload = body.getvalue()
        return struct.pack(">HI", self.attribute_name_index, len(payload)) + payload


@dataclass
class SignatureAttribute(AttributeInfo):
    """Generic type signature attribute (JVMS §4.7.9)."""
    signature_index: int = 0

    def signature(self, cp: ConstantPool) -> str:
        return cp.resolve_utf8(self.signature_index)

    def to_bytes(self, cp: Optional[ConstantPool] = None) -> bytes:
        return struct.pack(">HIH", self.attribute_name_index, 2, self.signature_index)


@dataclass
class SourceFileAttribute(AttributeInfo):
    """Source file name attribute (JVMS §4.7.10)."""
    sourcefile_index: int = 0

    def sourcefile(self, cp: ConstantPool) -> str:
        return cp.resolve_utf8(self.sourcefile_index)

    def to_bytes(self, cp: Optional[ConstantPool] = None) -> bytes:
        return struct.pack(">HIH", self.attribute_name_index, 2, self.sourcefile_index)


@dataclass
class ConstantValueAttribute(AttributeInfo):
    """Constant value attribute for static final fields (JVMS §4.7.2)."""
    constantvalue_index: int = 0

    def to_bytes(self, cp: Optional[ConstantPool] = None) -> bytes:
        return struct.pack(">HIH", self.attribute_name_index, 2, self.constantvalue_index)


@dataclass
class ExceptionsAttribute(AttributeInfo):
    """Checked exceptions declared by a method (JVMS §4.7.5)."""
    exception_index_table: List[int] = field(default_factory=list)

    def exception_names(self, cp: ConstantPool) -> List[str]:
        return [cp.resolve_class_name(idx) for idx in self.exception_index_table]

    def to_bytes(self, cp: Optional[ConstantPool] = None) -> bytes:
        payload = struct.pack(">H", len(self.exception_index_table)) + b"".join(
            struct.pack(">H", idx) for idx in self.exception_index_table
        )
        return struct.pack(">HI", self.attribute_name_index, len(payload)) + payload


@dataclass
class StackMapTableAttribute(AttributeInfo):
    """StackMapTable verification frames for Java 7+ bytecode (JVMS §4.7.4)."""
    entries_data: bytes = b""

    def to_bytes(self, cp: Optional[ConstantPool] = None) -> bytes:
        return struct.pack(">HI", self.attribute_name_index, len(self.entries_data)) + self.entries_data


@dataclass
class BootstrapMethod:
    """Individual entry in BootstrapMethods attribute (JVMS §4.7.23)."""
    bootstrap_method_ref: int
    bootstrap_arguments: List[int] = field(default_factory=list)


@dataclass
class BootstrapMethodsAttribute(AttributeInfo):
    """Bootstrap methods table for invokedynamic / CONSTANT_Dynamic (JVMS §4.7.23)."""
    bootstrap_methods: List[BootstrapMethod] = field(default_factory=list)

    def to_bytes(self, cp: Optional[ConstantPool] = None) -> bytes:
        buf = io.BytesIO()
        buf.write(struct.pack(">H", len(self.bootstrap_methods)))
        for bm in self.bootstrap_methods:
            buf.write(struct.pack(">HH", bm.bootstrap_method_ref, len(bm.bootstrap_arguments)))
            for arg in bm.bootstrap_arguments:
                buf.write(struct.pack(">H", arg))
        payload = buf.getvalue()
        return struct.pack(">HI", self.attribute_name_index, len(payload)) + payload


@dataclass
class MemberInfo:
    """Base class for fields and methods (JVMS §4.5, §4.6)."""
    access_flags: int
    name_index: int
    descriptor_index: int
    attributes: List[AttributeInfo] = field(default_factory=list)

    def name(self, cp: ConstantPool) -> str:
        return cp.resolve_utf8(self.name_index)

    def descriptor(self, cp: ConstantPool) -> str:
        return cp.resolve_utf8(self.descriptor_index)

    def get_attribute(self, name: str, cp: ConstantPool) -> Optional[AttributeInfo]:
        for attr in self.attributes:
            try:
                if attr.name(cp) == name:
                    return attr
            except Exception:
                continue
        return None

    def signature(self, cp: ConstantPool) -> Optional[str]:
        attr = self.get_attribute("Signature", cp)
        if isinstance(attr, SignatureAttribute):
            return attr.signature(cp)
        return None

    def to_bytes(self, cp: Optional[ConstantPool] = None) -> bytes:
        out = io.BytesIO()
        out.write(struct.pack(">HHH", self.access_flags, self.name_index, self.descriptor_index))
        out.write(struct.pack(">H", len(self.attributes)))
        for attr in self.attributes:
            out.write(attr.to_bytes(cp))
        return out.getvalue()


@dataclass
class FieldInfo(MemberInfo):
    """JVM field model (JVMS §4.5)."""
    @property
    def is_static(self) -> bool:
        return bool(self.access_flags & AccessFlags.STATIC)

    @property
    def is_final(self) -> bool:
        return bool(self.access_flags & AccessFlags.FINAL)

    @property
    def is_synthetic(self) -> bool:
        return bool(self.access_flags & AccessFlags.SYNTHETIC)

    @property
    def is_volatile(self) -> bool:
        return bool(self.access_flags & AccessFlags.VOLATILE)

    @property
    def is_transient(self) -> bool:
        return bool(self.access_flags & AccessFlags.TRANSIENT)


@dataclass
class MethodInfo(MemberInfo):
    """JVM method model (JVMS §4.6)."""
    def code_attribute(self, cp: ConstantPool) -> Optional[CodeAttribute]:
        attr = self.get_attribute("Code", cp)
        if isinstance(attr, CodeAttribute):
            return attr
        return None

    @property
    def is_static(self) -> bool:
        return bool(self.access_flags & AccessFlags.STATIC)

    @property
    def is_native(self) -> bool:
        return bool(self.access_flags & AccessFlags.NATIVE)

    @property
    def is_abstract(self) -> bool:
        return bool(self.access_flags & AccessFlags.ABSTRACT)

    @property
    def is_synthetic(self) -> bool:
        return bool(self.access_flags & AccessFlags.SYNTHETIC)

    @property
    def is_bridge(self) -> bool:
        return bool(self.access_flags & AccessFlags.BRIDGE)

    @property
    def is_synchronized(self) -> bool:
        return bool(self.access_flags & AccessFlags.SYNCHRONIZED)

    @property
    def is_varargs(self) -> bool:
        return bool(self.access_flags & AccessFlags.VARARGS)

    def is_constructor(self, cp: ConstantPool) -> bool:
        return self.name(cp) == "<init>"

    def is_class_initializer(self, cp: ConstantPool) -> bool:
        return self.name(cp) == "<clinit>"


def parse_attribute(stream: BinaryIO, cp: ConstantPool) -> AttributeInfo:
    """
    Parses an attribute from a binary stream. Returns strongly-typed AttributeInfo or RawAttribute fallback.
    """
    header = stream.read(6)
    if len(header) < 6:
        raise EOFError("Truncated attribute header")
    name_idx, length = struct.unpack(">HI", header)
    payload = stream.read(length)
    if len(payload) < length:
        raise EOFError(f"Truncated attribute payload: expected {length}, got {len(payload)}")

    try:
        attr_name = cp.resolve_utf8(name_idx)
    except Exception:
        # If constant pool index is invalid, return RawAttribute
        return RawAttribute(name_idx, payload)

    pstream = io.BytesIO(payload)

    try:
        if attr_name == "Code":
            max_stack, max_locals, code_len = struct.unpack(">HHI", pstream.read(8))
            code = pstream.read(code_len)
            ex_len = struct.unpack(">H", pstream.read(2))[0]
            ex_table: List[ExceptionTableEntry] = []
            for _ in range(ex_len):
                s, e, h, c = struct.unpack(">HHHH", pstream.read(8))
                ex_table.append(ExceptionTableEntry(s, e, h, c))
            sub_attr_count = struct.unpack(">H", pstream.read(2))[0]
            sub_attrs = [parse_attribute(pstream, cp) for _ in range(sub_attr_count)]
            return CodeAttribute(name_idx, max_stack, max_locals, code, ex_table, sub_attrs)
        elif attr_name == "Signature" and length == 2:
            sig_idx = struct.unpack(">H", pstream.read(2))[0]
            return SignatureAttribute(name_idx, sig_idx)
        elif attr_name == "SourceFile" and length == 2:
            sf_idx = struct.unpack(">H", pstream.read(2))[0]
            return SourceFileAttribute(name_idx, sf_idx)
        elif attr_name == "ConstantValue" and length == 2:
            cv_idx = struct.unpack(">H", pstream.read(2))[0]
            return ConstantValueAttribute(name_idx, cv_idx)
        elif attr_name == "Exceptions":
            num_ex = struct.unpack(">H", pstream.read(2))[0]
            ex_indices = [struct.unpack(">H", pstream.read(2))[0] for _ in range(num_ex)]
            return ExceptionsAttribute(name_idx, ex_indices)
        elif attr_name == "StackMapTable":
            return StackMapTableAttribute(name_idx, payload)
        elif attr_name == "BootstrapMethods":
            num_bm = struct.unpack(">H", pstream.read(2))[0]
            bms: List[BootstrapMethod] = []
            for _ in range(num_bm):
                mref, num_args = struct.unpack(">HH", pstream.read(4))
                args = [struct.unpack(">H", pstream.read(2))[0] for _ in range(num_args)]
                bms.append(BootstrapMethod(mref, args))
            return BootstrapMethodsAttribute(name_idx, bms)
        else:
            return RawAttribute(name_idx, payload)
    except Exception:
        # Fallback to RawAttribute on any parse discrepancy to guarantee zero data loss
        return RawAttribute(name_idx, payload)
