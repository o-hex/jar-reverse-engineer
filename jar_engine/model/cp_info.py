"""
jar_engine.model.cp_info: Constant pool entry models, Modified UTF-8 codec, and ConstantPool table.
Conforming to JVMS §4.4.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, BinaryIO, Iterator, Union
import struct
import io
import math

from .constants import ConstantTag, ReferenceKind


def decode_modified_utf8(data: bytes) -> str:
    """
    Decodes JVM Modified UTF-8 bytes to a Python string per JVMS §4.4.7.
    - \u0000 is encoded as 0xC0 0x80
    - 1-byte: 0x01..0x7F
    - 2-byte: 0xC0..0xDF 0x80..0xBF
    - 3-byte: 0xE0..0xEF 0x80..0xBF 0x80..0xBF
    - Characters > 0xFFFF are encoded as 6-byte surrogate pairs.
    """
    res: List[str] = []
    i = 0
    n = len(data)
    while i < n:
        b1 = data[i]
        if (b1 & 0x80) == 0:
            res.append(chr(b1))
            i += 1
        elif (b1 & 0xE0) == 0xC0:
            if i + 1 >= n:
                # Truncated, fallback
                return data.decode("utf-8", errors="replace")
            b2 = data[i + 1]
            cp = ((b1 & 0x1F) << 6) | (b2 & 0x3F)
            res.append(chr(cp))
            i += 2
        elif (b1 & 0xF0) == 0xE0:
            if i + 2 >= n:
                return data.decode("utf-8", errors="replace")
            b2 = data[i + 1]
            b3 = data[i + 2]
            cp = ((b1 & 0x0F) << 12) | ((b2 & 0x3F) << 6) | (b3 & 0x3F)
            res.append(chr(cp))
            i += 3
        else:
            # Irregular / standard 4-byte UTF-8 fallback
            return data.decode("utf-8", errors="replace")

    s = "".join(res)
    # Merge UTF-16 surrogate pairs if astral plane characters were encoded
    try:
        s = s.encode("utf-16", "surrogatepass").decode("utf-16")
    except Exception:
        pass
    return s


def encode_modified_utf8(s: str) -> bytes:
    """
    Encodes a Python string into JVM Modified UTF-8 bytes per JVMS §4.4.7.
    """
    res = bytearray()
    for ch in s:
        cp = ord(ch)
        if 1 <= cp <= 0x7F:
            res.append(cp)
        elif cp == 0 or (0x80 <= cp <= 0x7FF):
            res.append(0xC0 | ((cp >> 6) & 0x1F))
            res.append(0x80 | (cp & 0x3F))
        elif 0x800 <= cp <= 0xFFFF:
            res.append(0xE0 | ((cp >> 12) & 0x0F))
            res.append(0x80 | ((cp >> 6) & 0x3F))
            res.append(0x80 | (cp & 0x3F))
        else:
            # Code point > 0xFFFF: encode as UTF-16 surrogate pair (each 3 bytes, total 6 bytes)
            high = 0xD800 + ((cp - 0x10000) >> 10)
            low = 0xDC00 + ((cp - 0x10000) & 0x3FF)
            for surr in (high, low):
                res.append(0xE0 | ((surr >> 12) & 0x0F))
                res.append(0x80 | ((surr >> 6) & 0x3F))
                res.append(0x80 | (surr & 0x3F))
    return bytes(res)


@dataclass
class CpEntry:
    tag: int
    is_wide: bool = False

    def to_bytes(self) -> bytes:
        raise NotImplementedError


@dataclass
class CpPlaceholder(CpEntry):
    """Occupies index n+1 after CONSTANT_Long and CONSTANT_Double entries."""
    tag: int = 0
    is_wide: bool = False

    def to_bytes(self) -> bytes:
        return b""


@dataclass
class CONSTANT_Utf8_info(CpEntry):
    tag: int = ConstantTag.UTF8
    value: str = ""
    raw_bytes: Optional[bytes] = None

    def to_bytes(self) -> bytes:
        # If raw_bytes is preserved and string hasn't changed, reuse raw_bytes for 100% byte fidelity
        if self.raw_bytes is not None:
            try:
                if decode_modified_utf8(self.raw_bytes) == self.value:
                    return struct.pack(">BH", ConstantTag.UTF8, len(self.raw_bytes)) + self.raw_bytes
            except Exception:
                pass
        encoded = encode_modified_utf8(self.value)
        return struct.pack(">BH", ConstantTag.UTF8, len(encoded)) + encoded


@dataclass
class CONSTANT_Integer_info(CpEntry):
    tag: int = ConstantTag.INTEGER
    value: int = 0  # 32-bit signed int

    def to_bytes(self) -> bytes:
        return struct.pack(">Bi", ConstantTag.INTEGER, self.value)


@dataclass
class CONSTANT_Float_info(CpEntry):
    tag: int = ConstantTag.FLOAT
    value: float = 0.0
    raw_bits: int = 0  # Preserves exact NaN payloads

    def __post_init__(self) -> None:
        if self.raw_bits == 0 and (self.value != 0.0 or math.copysign(1.0, self.value) < 0):
            self.raw_bits = struct.unpack(">I", struct.pack(">f", self.value))[0]
        elif self.raw_bits != 0 and self.value == 0.0:
            self.value = struct.unpack(">f", struct.pack(">I", self.raw_bits))[0]

    def to_bytes(self) -> bytes:
        bits = self.raw_bits
        if bits == 0 and (self.value != 0.0 or math.copysign(1.0, self.value) < 0):
            bits = struct.unpack(">I", struct.pack(">f", self.value))[0]
        return struct.pack(">BI", ConstantTag.FLOAT, bits)


@dataclass
class CONSTANT_Long_info(CpEntry):
    tag: int = ConstantTag.LONG
    is_wide: bool = True
    value: int = 0  # 64-bit signed int

    def to_bytes(self) -> bytes:
        return struct.pack(">Bq", ConstantTag.LONG, self.value)


@dataclass
class CONSTANT_Double_info(CpEntry):
    tag: int = ConstantTag.DOUBLE
    is_wide: bool = True
    value: float = 0.0
    raw_bits: int = 0  # Preserves exact NaN payloads

    def __post_init__(self) -> None:
        if self.raw_bits == 0 and (self.value != 0.0 or math.copysign(1.0, self.value) < 0):
            self.raw_bits = struct.unpack(">Q", struct.pack(">d", self.value))[0]
        elif self.raw_bits != 0 and self.value == 0.0:
            self.value = struct.unpack(">d", struct.pack(">Q", self.raw_bits))[0]

    def to_bytes(self) -> bytes:
        bits = self.raw_bits
        if bits == 0 and (self.value != 0.0 or math.copysign(1.0, self.value) < 0):
            bits = struct.unpack(">Q", struct.pack(">d", self.value))[0]
        return struct.pack(">BQ", ConstantTag.DOUBLE, bits)


@dataclass
class CONSTANT_Class_info(CpEntry):
    tag: int = ConstantTag.CLASS
    name_index: int = 0

    def to_bytes(self) -> bytes:
        return struct.pack(">BH", ConstantTag.CLASS, self.name_index)


@dataclass
class CONSTANT_String_info(CpEntry):
    tag: int = ConstantTag.STRING
    string_index: int = 0

    def to_bytes(self) -> bytes:
        return struct.pack(">BH", ConstantTag.STRING, self.string_index)


@dataclass
class CONSTANT_Fieldref_info(CpEntry):
    tag: int = ConstantTag.FIELDREF
    class_index: int = 0
    name_and_type_index: int = 0

    def to_bytes(self) -> bytes:
        return struct.pack(">BHH", ConstantTag.FIELDREF, self.class_index, self.name_and_type_index)


@dataclass
class CONSTANT_Methodref_info(CpEntry):
    tag: int = ConstantTag.METHODREF
    class_index: int = 0
    name_and_type_index: int = 0

    def to_bytes(self) -> bytes:
        return struct.pack(">BHH", ConstantTag.METHODREF, self.class_index, self.name_and_type_index)


@dataclass
class CONSTANT_InterfaceMethodref_info(CpEntry):
    tag: int = ConstantTag.INTERFACE_METHODREF
    class_index: int = 0
    name_and_type_index: int = 0

    def to_bytes(self) -> bytes:
        return struct.pack(">BHH", ConstantTag.INTERFACE_METHODREF, self.class_index, self.name_and_type_index)


@dataclass
class CONSTANT_NameAndType_info(CpEntry):
    tag: int = ConstantTag.NAME_AND_TYPE
    name_index: int = 0
    descriptor_index: int = 0

    def to_bytes(self) -> bytes:
        return struct.pack(">BHH", ConstantTag.NAME_AND_TYPE, self.name_index, self.descriptor_index)


@dataclass
class CONSTANT_MethodHandle_info(CpEntry):
    tag: int = ConstantTag.METHOD_HANDLE
    reference_kind: int = 0
    reference_index: int = 0

    def to_bytes(self) -> bytes:
        return struct.pack(">BBH", ConstantTag.METHOD_HANDLE, self.reference_kind, self.reference_index)


@dataclass
class CONSTANT_MethodType_info(CpEntry):
    tag: int = ConstantTag.METHOD_TYPE
    descriptor_index: int = 0

    def to_bytes(self) -> bytes:
        return struct.pack(">BH", ConstantTag.METHOD_TYPE, self.descriptor_index)


@dataclass
class CONSTANT_Dynamic_info(CpEntry):
    tag: int = ConstantTag.DYNAMIC
    bootstrap_method_attr_index: int = 0
    name_and_type_index: int = 0

    def to_bytes(self) -> bytes:
        return struct.pack(">BHH", ConstantTag.DYNAMIC, self.bootstrap_method_attr_index, self.name_and_type_index)


@dataclass
class CONSTANT_InvokeDynamic_info(CpEntry):
    tag: int = ConstantTag.INVOKE_DYNAMIC
    bootstrap_method_attr_index: int = 0
    name_and_type_index: int = 0

    def to_bytes(self) -> bytes:
        return struct.pack(">BHH", ConstantTag.INVOKE_DYNAMIC, self.bootstrap_method_attr_index, self.name_and_type_index)


@dataclass
class CONSTANT_Module_info(CpEntry):
    tag: int = ConstantTag.MODULE
    name_index: int = 0

    def to_bytes(self) -> bytes:
        return struct.pack(">BH", ConstantTag.MODULE, self.name_index)


@dataclass
class CONSTANT_Package_info(CpEntry):
    tag: int = ConstantTag.PACKAGE
    name_index: int = 0

    def to_bytes(self) -> bytes:
        return struct.pack(">BH", ConstantTag.PACKAGE, self.name_index)


class ConstantPool:
    """
    1-indexed Constant Pool table managing JVM constants, lookup, deduplication, and rewriting.
    """
    def __init__(self, entries: Optional[List[Optional[CpEntry]]] = None):
        # entries[0] is None, 1-indexed.
        if entries is not None:
            self._entries: List[Optional[CpEntry]] = list(entries)
            if not self._entries or self._entries[0] is not None:
                self._entries.insert(0, None)
        else:
            self._entries: List[Optional[CpEntry]] = [None]

    def __len__(self) -> int:
        """Returns constant_pool_count (i.e. length of internal 1-indexed list)."""
        return len(self._entries)

    def __getitem__(self, index: int) -> CpEntry:
        if index <= 0 or index >= len(self._entries):
            raise IndexError(f"Constant pool index out of bounds: {index} (size: {len(self._entries)})")
        entry = self._entries[index]
        if entry is None or isinstance(entry, CpPlaceholder):
            raise ValueError(f"Constant pool index {index} is an unusable 2nd slot or empty entry")
        return entry

    def __setitem__(self, index: int, entry: CpEntry) -> None:
        if index <= 0 or index >= len(self._entries):
            raise IndexError(f"Constant pool index out of bounds: {index}")
        self._entries[index] = entry

    def get(self, index: int) -> Optional[CpEntry]:
        if index <= 0 or index >= len(self._entries):
            return None
        entry = self._entries[index]
        if entry is None or isinstance(entry, CpPlaceholder):
            return None
        return entry

    def entries(self) -> Iterator[Tuple[int, CpEntry]]:
        """Yields (index, entry) for all non-placeholder, valid entries."""
        for i in range(1, len(self._entries)):
            entry = self._entries[i]
            if entry is not None and not isinstance(entry, CpPlaceholder):
                yield i, entry

    # --- Resolution Helpers ---
    def resolve_utf8(self, index: int) -> str:
        entry = self[index]
        if not isinstance(entry, CONSTANT_Utf8_info):
            raise TypeError(f"CP entry at {index} is {type(entry).__name__}, expected CONSTANT_Utf8_info")
        return entry.value

    def resolve_class_name(self, index: int) -> str:
        entry = self[index]
        if not isinstance(entry, CONSTANT_Class_info):
            raise TypeError(f"CP entry at {index} is {type(entry).__name__}, expected CONSTANT_Class_info")
        return self.resolve_utf8(entry.name_index)

    def resolve_string(self, index: int) -> str:
        entry = self[index]
        if not isinstance(entry, CONSTANT_String_info):
            raise TypeError(f"CP entry at {index} is {type(entry).__name__}, expected CONSTANT_String_info")
        return self.resolve_utf8(entry.string_index)

    def resolve_name_and_type(self, index: int) -> Tuple[str, str]:
        entry = self[index]
        if not isinstance(entry, CONSTANT_NameAndType_info):
            raise TypeError(f"CP entry at {index} is {type(entry).__name__}, expected CONSTANT_NameAndType_info")
        return self.resolve_utf8(entry.name_index), self.resolve_utf8(entry.descriptor_index)

    def resolve_member_ref(self, index: int) -> Tuple[str, str, str]:
        entry = self[index]
        if not isinstance(entry, (CONSTANT_Fieldref_info, CONSTANT_Methodref_info, CONSTANT_InterfaceMethodref_info)):
            raise TypeError(f"CP entry at {index} is {type(entry).__name__}, expected member ref")
        cls_name = self.resolve_class_name(entry.class_index)
        name, desc = self.resolve_name_and_type(entry.name_and_type_index)
        return cls_name, name, desc

    # --- Insertion & Deduplication Helpers ---
    def add_utf8(self, value: str) -> int:
        for idx, entry in self.entries():
            if isinstance(entry, CONSTANT_Utf8_info) and entry.value == value:
                return idx
        idx = len(self._entries)
        raw = encode_modified_utf8(value)
        self._entries.append(CONSTANT_Utf8_info(tag=ConstantTag.UTF8, value=value, raw_bytes=raw))
        return idx

    def add_class(self, class_name: str) -> int:
        utf_idx = self.add_utf8(class_name)
        for idx, entry in self.entries():
            if isinstance(entry, CONSTANT_Class_info) and entry.name_index == utf_idx:
                return idx
        idx = len(self._entries)
        self._entries.append(CONSTANT_Class_info(name_index=utf_idx))
        return idx

    def add_name_and_type(self, name: str, descriptor: str) -> int:
        n_idx = self.add_utf8(name)
        d_idx = self.add_utf8(descriptor)
        for idx, entry in self.entries():
            if isinstance(entry, CONSTANT_NameAndType_info) and entry.name_index == n_idx and entry.descriptor_index == d_idx:
                return idx
        idx = len(self._entries)
        self._entries.append(CONSTANT_NameAndType_info(name_index=n_idx, descriptor_index=d_idx))
        return idx

    def add_member_ref(self, tag: ConstantTag, class_name: str, name: str, descriptor: str) -> int:
        cls_idx = self.add_class(class_name)
        nat_idx = self.add_name_and_type(name, descriptor)
        for idx, entry in self.entries():
            if entry.tag == tag and getattr(entry, "class_index", None) == cls_idx and getattr(entry, "name_and_type_index", None) == nat_idx:
                return idx
        idx = len(self._entries)
        cls_map = {
            ConstantTag.FIELDREF: CONSTANT_Fieldref_info,
            ConstantTag.METHODREF: CONSTANT_Methodref_info,
            ConstantTag.INTERFACE_METHODREF: CONSTANT_InterfaceMethodref_info,
        }
        cls_type = cls_map[tag]
        self._entries.append(cls_type(class_index=cls_idx, name_and_type_index=nat_idx))
        return idx

    def add_string(self, value: str) -> int:
        utf_idx = self.add_utf8(value)
        for idx, entry in self.entries():
            if isinstance(entry, CONSTANT_String_info) and entry.string_index == utf_idx:
                return idx
        idx = len(self._entries)
        self._entries.append(CONSTANT_String_info(string_index=utf_idx))
        return idx

    def add_integer(self, value: int) -> int:
        for idx, entry in self.entries():
            if isinstance(entry, CONSTANT_Integer_info) and entry.value == value:
                return idx
        idx = len(self._entries)
        self._entries.append(CONSTANT_Integer_info(value=value))
        return idx

    def add_long(self, value: int) -> int:
        for idx, entry in self.entries():
            if isinstance(entry, CONSTANT_Long_info) and entry.value == value:
                return idx
        idx = len(self._entries)
        self._entries.append(CONSTANT_Long_info(value=value))
        self._entries.append(CpPlaceholder())
        return idx

    def add_float(self, value: float) -> int:
        raw_bits = struct.unpack(">I", struct.pack(">f", value))[0]
        for idx, entry in self.entries():
            if isinstance(entry, CONSTANT_Float_info) and entry.raw_bits == raw_bits:
                return idx
        idx = len(self._entries)
        self._entries.append(CONSTANT_Float_info(value=value, raw_bits=raw_bits))
        return idx

    def add_double(self, value: float) -> int:
        raw_bits = struct.unpack(">Q", struct.pack(">d", value))[0]
        for idx, entry in self.entries():
            if isinstance(entry, CONSTANT_Double_info) and entry.raw_bits == raw_bits:
                return idx
        idx = len(self._entries)
        self._entries.append(CONSTANT_Double_info(value=value, raw_bits=raw_bits))
        self._entries.append(CpPlaceholder())
        return idx

    def update_utf8(self, index: int, new_value: str) -> None:
        """In-place update of CONSTANT_Utf8 entry. Re-encodes modified UTF-8 value."""
        entry = self[index]
        if not isinstance(entry, CONSTANT_Utf8_info):
            raise TypeError(f"CP entry at {index} is {type(entry).__name__}, expected CONSTANT_Utf8_info")
        entry.value = new_value
        entry.raw_bytes = encode_modified_utf8(new_value)

    # --- Serialization ---
    def to_bytes(self) -> bytes:
        out = io.BytesIO()
        idx = 1
        n = len(self._entries)
        while idx < n:
            entry = self._entries[idx]
            if entry is None or isinstance(entry, CpPlaceholder):
                idx += 1
                continue
            out.write(entry.to_bytes())
            if entry.is_wide:
                idx += 2
            else:
                idx += 1
        return out.getvalue()
