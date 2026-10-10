"""
jar_engine.model.jar_archive: JAR/ZIP archive reader, writer, manifest handler, and symbol indexer.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, BinaryIO, Generator, TYPE_CHECKING
import zipfile
import re

from .class_file import ClassFile, ClassFormatError
from .member_info import MethodInfo, FieldInfo

if TYPE_CHECKING:
    from jar_engine.walker.hierarchy import InheritanceHierarchy


def wrap_manifest_line(key: str, value: str, max_bytes: int = 72) -> bytes:
    """Wraps a manifest header line per RFC 822 (max 72 UTF-8 bytes per line)."""
    line_str = f"{key}: {value}"
    raw_bytes = line_str.encode("utf-8")
    if len(raw_bytes) <= max_bytes:
        return raw_bytes + b"\r\n"

    chunks: List[bytes] = []
    # First chunk: up to 72 bytes
    chunks.append(raw_bytes[:max_bytes])
    remainder = raw_bytes[max_bytes:]
    # Continuation chunks: 1 space byte (0x20) + up to 71 bytes
    while remainder:
        chunk = remainder[:max_bytes - 1]
        chunks.append(b" " + chunk)
        remainder = remainder[max_bytes - 1:]
    return b"\r\n".join(chunks) + b"\r\n"


class Manifest:
    """Represents a JAR META-INF/MANIFEST.MF conforming to RFC 822."""

    def __init__(
        self,
        main_attributes: Optional[Dict[str, str]] = None,
        sections: Optional[Dict[str, Dict[str, str]]] = None
    ):
        self.main_attributes: Dict[str, str] = dict(main_attributes or {})
        self.sections: Dict[str, Dict[str, str]] = dict(sections or {})

    @classmethod
    def parse(cls, data: Union[bytes, str]) -> "Manifest":
        if isinstance(data, str):
            raw = data.encode("utf-8")
        else:
            raw = data

        # Unfold RFC 822 continuation lines in byte domain prior to UTF-8 decoding.
        # This prevents multi-byte UTF-8 sequences straddling the 72-byte line wrap
        # boundary from being corrupted into replacement characters (\ufffd).
        raw_unfolded = re.sub(rb"\r\n[ \t]|\n[ \t]|\r[ \t]", b"", raw)
        text = raw_unfolded.decode("utf-8", errors="replace")
        lines = text.splitlines()

        # Step 1: Unfold RFC 822 continuation lines
        unfolded: List[str] = []
        for line in lines:
            if not line:
                unfolded.append("")
            elif line.startswith(" ") or line.startswith("\t"):
                if unfolded:
                    unfolded[-1] += line[1:]
                else:
                    unfolded.append(line.lstrip())
            else:
                unfolded.append(line)

        # Step 2: Split into sections delimited by blank lines
        sections_raw: List[List[str]] = []
        current: List[str] = []
        for line in unfolded:
            if not line.strip():
                if current:
                    sections_raw.append(current)
                    current = []
            else:
                current.append(line)
        if current:
            sections_raw.append(current)

        main_attrs: Dict[str, str] = {}
        named_sections: Dict[str, Dict[str, str]] = {}

        if sections_raw:
            # Parse main section
            for line in sections_raw[0]:
                if ":" in line:
                    k, v = line.split(":", 1)
                    main_attrs[k.strip()] = v.strip()

            # Parse subsequent named sections
            for sec_lines in sections_raw[1:]:
                sec_dict: Dict[str, str] = {}
                sec_name: Optional[str] = None
                for line in sec_lines:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        k_s = k.strip()
                        v_s = v.strip()
                        if k_s.lower() == "name" and sec_name is None:
                            sec_name = v_s
                        else:
                            sec_dict[k_s] = v_s
                if sec_name:
                    named_sections[sec_name] = sec_dict

        return cls(main_attributes=main_attrs, sections=named_sections)

    def to_bytes(self) -> bytes:
        out = BytesIO()

        # Write Main Section: Manifest-Version first if present
        mv_key = None
        for k in self.main_attributes:
            if k.lower() == "manifest-version":
                mv_key = k
                break

        if mv_key:
            out.write(wrap_manifest_line(mv_key, self.main_attributes[mv_key]))
        elif not self.main_attributes:
            # Empty manifest: default Manifest-Version: 1.0
            out.write(wrap_manifest_line("Manifest-Version", "1.0"))

        for k, v in self.main_attributes.items():
            if k == mv_key:
                continue
            out.write(wrap_manifest_line(k, v))

        out.write(b"\r\n")

        # Write Individual Sections
        for sec_name, attrs in self.sections.items():
            out.write(wrap_manifest_line("Name", sec_name))
            for k, v in attrs.items():
                if k.lower() == "name":
                    continue
                out.write(wrap_manifest_line(k, v))
            out.write(b"\r\n")

        return out.getvalue()

    def get_attribute(self, key: str, section: Optional[str] = None) -> Optional[str]:
        target = self.main_attributes if section is None else self.sections.get(section, {})
        key_lower = key.lower()
        for k, v in target.items():
            if k.lower() == key_lower:
                return v
        return None

    def set_attribute(self, key: str, value: str, section: Optional[str] = None) -> None:
        target = self.main_attributes if section is None else self.sections.setdefault(section, {})
        key_lower = key.lower()
        found_key = None
        for k in target:
            if k.lower() == key_lower:
                found_key = k
                break
        if found_key:
            target[found_key] = value
        else:
            target[key] = value

    def remove_attribute(self, key: str, section: Optional[str] = None) -> Optional[str]:
        target = self.main_attributes if section is None else self.sections.get(section, {})
        key_lower = key.lower()
        found_key = None
        for k in target:
            if k.lower() == key_lower:
                found_key = k
                break
        if found_key:
            return target.pop(found_key)
        return None

    def get_main_class(self) -> Optional[str]:
        return self.get_attribute("Main-Class")

    def set_main_class(self, class_name: str) -> None:
        # Convert internal name "com/example/Main" to dot name "com.example.Main" if needed
        dot_name = class_name.replace("/", ".")
        self.set_attribute("Main-Class", dot_name)


@dataclass
class JarStats:
    class_count: int
    method_count: int
    field_count: int
    resource_count: int
    total_uncompressed_bytes: int
    packages: Set[str] = field(default_factory=set)


class JarArchive:
    """In-memory representation of a compiled Java JAR archive."""

    def __init__(self):
        self.classes: Dict[str, ClassFile] = {}
        self.resources: Dict[str, bytes] = {}
        self.manifest: Optional[Manifest] = None
        self.directories: Set[str] = set()
        self.versioned_classes: Dict[int, Dict[str, ClassFile]] = {}
        self.path: Optional[Path] = None
        self._hierarchy_cache: Optional[InheritanceHierarchy] = None

    @classmethod
    def from_file(cls, path: Union[str, Path], strict: bool = False) -> "JarArchive":
        p = Path(path)
        with open(p, "rb") as f:
            archive = cls.from_stream(f, strict=strict)
            archive.path = p
            return archive

    @classmethod
    def from_bytes(cls, data: bytes, strict: bool = False) -> "JarArchive":
        return cls.from_stream(BytesIO(data), strict=strict)

    @classmethod
    def from_stream(cls, stream: BinaryIO, strict: bool = False) -> "JarArchive":
        archive = cls()
        mr_pattern = re.compile(r"^META-INF/versions/(\d+)/(.*\.class)$")

        with zipfile.ZipFile(stream, "r") as zf:
            for entry in zf.infolist():
                name = entry.filename

                if entry.is_dir() or name.endswith("/"):
                    archive.directories.add(name)
                    continue

                raw_bytes = zf.read(name)

                if name == "META-INF/MANIFEST.MF":
                    archive.manifest = Manifest.parse(raw_bytes)
                elif name.endswith(".class"):
                    mr_match = mr_pattern.match(name)
                    if mr_match:
                        version = int(mr_match.group(1))
                        try:
                            cf = ClassFile.parse(raw_bytes)
                            if version not in archive.versioned_classes:
                                archive.versioned_classes[version] = {}
                            archive.versioned_classes[version][cf.this_class_name] = cf
                        except Exception as e:
                            if strict:
                                raise
                            archive.resources[name] = raw_bytes
                    else:
                        try:
                            cf = ClassFile.parse(raw_bytes)
                            archive.classes[cf.this_class_name] = cf
                        except Exception as e:
                            if strict:
                                raise
                            archive.resources[name] = raw_bytes
                else:
                    archive.resources[name] = raw_bytes

        return archive

    def save(
        self,
        path: Union[str, Path],
        compression: int = zipfile.ZIP_DEFLATED,
        strip_signatures: bool = False
    ) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_bytes(compression=compression, strip_signatures=strip_signatures)
        with open(p, "wb") as f:
            f.write(data)

    def to_bytes(
        self,
        compression: int = zipfile.ZIP_DEFLATED,
        strip_signatures: bool = False
    ) -> bytes:
        if strip_signatures:
            self.strip_signatures()

        out = BytesIO()
        with zipfile.ZipFile(out, "w", compression=compression) as zf:
            # 1. Directory entries
            for d in sorted(self.directories):
                if not d.endswith("/"):
                    d += "/"
                zf.writestr(d, b"")

            # 2. Manifest
            if self.manifest is not None:
                zf.writestr("META-INF/MANIFEST.MF", self.manifest.to_bytes())

            # 3. Resources
            sig_pattern = re.compile(r"^META-INF/(.*\.(SF|DSA|RSA)|SIG-.*)$", re.IGNORECASE)
            for r_name, r_bytes in self.resources.items():
                if strip_signatures and sig_pattern.match(r_name):
                    continue
                zf.writestr(r_name, r_bytes)

            # 4. Standard Classes
            for class_name, cf in sorted(self.classes.items(), key=lambda x: x[0]):
                entry_name = f"{class_name}.class"
                zf.writestr(entry_name, cf.to_bytes())

            # 5. Multi-Release Classes
            for ver, v_classes in sorted(self.versioned_classes.items(), key=lambda x: x[0]):
                for c_name, cf in sorted(v_classes.items(), key=lambda x: x[0]):
                    entry_name = f"META-INF/versions/{ver}/{c_name}.class"
                    zf.writestr(entry_name, cf.to_bytes())

        return out.getvalue()

    # --- Class Operations ---
    def get_class(self, name: str) -> Optional[ClassFile]:
        norm = name.replace(".", "/")
        if norm.startswith("L") and norm.endswith(";"):
            norm = norm[1:-1]
        return self.classes.get(norm)

    def add_class(self, cf: ClassFile) -> None:
        self.classes[cf.this_class_name] = cf
        self._hierarchy_cache = None

    def remove_class(self, name: str) -> Optional[ClassFile]:
        norm = name.replace(".", "/")
        if norm.startswith("L") and norm.endswith(";"):
            norm = norm[1:-1]
        cf = self.classes.pop(norm, None)
        if cf is not None:
            self._hierarchy_cache = None
        return cf

    def has_class(self, name: str) -> bool:
        norm = name.replace(".", "/")
        if norm.startswith("L") and norm.endswith(";"):
            norm = norm[1:-1]
        return norm in self.classes

    @property
    def class_names(self) -> List[str]:
        return sorted(self.classes.keys())

    @property
    def packages(self) -> Set[str]:
        pkgs = set()
        for name in self.classes:
            if "/" in name:
                pkgs.add(name.rsplit("/", 1)[0])
        return pkgs

    # --- Resource Operations ---
    def get_resource(self, path: str) -> Optional[bytes]:
        return self.resources.get(path)

    def set_resource(self, path: str, data: bytes) -> None:
        self.resources[path] = data

    def remove_resource(self, path: str) -> Optional[bytes]:
        return self.resources.pop(path, None)

    def has_resource(self, path: str) -> bool:
        return path in self.resources

    @property
    def resource_names(self) -> List[str]:
        return sorted(self.resources.keys())

    def strip_signatures(self) -> int:
        """Removes code signing files (META-INF/*.SF, *.DSA, *.RSA, SIG-*) and manifest digests."""
        sig_pattern = re.compile(r"^META-INF/(.*\.(SF|DSA|RSA)|SIG-.*)$", re.IGNORECASE)
        removed_count = 0
        to_remove = [k for k in self.resources if sig_pattern.match(k)]
        for k in to_remove:
            del self.resources[k]
            removed_count += 1

        if self.manifest is not None:
            # Strip digest attributes from manifest sections
            digest_attrs = ["SHA-256-Digest", "SHA1-Digest", "MD5-Digest", "SHA-384-Digest", "SHA-512-Digest"]
            for sec_name, attrs in list(self.manifest.sections.items()):
                for d in digest_attrs:
                    for k in list(attrs.keys()):
                        if k.lower() == d.lower():
                            del attrs[k]
                if not attrs:
                    del self.manifest.sections[sec_name]

        return removed_count

    # --- Symbol Indexing ---
    def find_method(self, class_name: str, name: str, descriptor: str) -> Optional[MethodInfo]:
        cf = self.get_class(class_name)
        if cf is None:
            return None
        return cf.get_method(name, descriptor)

    def find_field(self, class_name: str, name: str, descriptor: Optional[str] = None) -> Optional[FieldInfo]:
        cf = self.get_class(class_name)
        if cf is None:
            return None
        return cf.get_field(name, descriptor)

    def all_methods(self) -> Generator[Tuple[str, MethodInfo], None, None]:
        for cls_name, cf in self.classes.items():
            for m in cf.methods:
                yield (cls_name, m)

    def all_fields(self) -> Generator[Tuple[str, FieldInfo], None, None]:
        for cls_name, cf in self.classes.items():
            for f in cf.fields:
                yield (cls_name, f)

    def get_stats(self) -> JarStats:
        total_methods = sum(len(cf.methods) for cf in self.classes.values())
        total_fields = sum(len(cf.fields) for cf in self.classes.values())
        total_bytes = sum(len(cf.to_bytes()) for cf in self.classes.values()) + sum(len(b) for b in self.resources.values())
        return JarStats(
            class_count=len(self.classes),
            method_count=total_methods,
            field_count=total_fields,
            resource_count=len(self.resources),
            total_uncompressed_bytes=total_bytes,
            packages=self.packages
        )

    def get_hierarchy(self) -> "InheritanceHierarchy":
        if self._hierarchy_cache is None:
            from jar_engine.walker.hierarchy import InheritanceHierarchy
            self._hierarchy_cache = InheritanceHierarchy.from_archive(self)
        return self._hierarchy_cache
