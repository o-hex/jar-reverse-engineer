"""
jar_engine: Java Bytecode Analysis and Program Comprehension Engine.
"""

__version__ = "0.1.0"

from .model import (
    JAVA_MAGIC, JavaMajorVersion, ConstantTag, ReferenceKind,
    AccessFlags, VerificationTypeTag, Opcode, FIXED_OPCODE_LENGTHS,
    decode_modified_utf8, encode_modified_utf8,
    CpEntry, CpPlaceholder,
    ConstantPool,
    ClassFile, ClassFormatError,
    JarArchive, Manifest, JarStats,
)

from .walker import (
    InheritanceHierarchy, CallGraph, ClassDependencyGraph,
    DependencySlicer, DependencyKind,
    MethodReference, FieldReference, CallEdge, FieldAccessRecord
)

__all__ = [
    "__version__",
    "JAVA_MAGIC", "JavaMajorVersion", "ConstantTag", "ReferenceKind",
    "AccessFlags", "VerificationTypeTag", "Opcode", "FIXED_OPCODE_LENGTHS",
    "decode_modified_utf8", "encode_modified_utf8",
    "CpEntry", "CpPlaceholder",
    "ConstantPool",
    "ClassFile", "ClassFormatError",
    "JarArchive", "Manifest", "JarStats",
    "InheritanceHierarchy", "CallGraph", "ClassDependencyGraph",
    "DependencySlicer", "DependencyKind",
    "MethodReference", "FieldReference", "CallEdge", "FieldAccessRecord",
]
