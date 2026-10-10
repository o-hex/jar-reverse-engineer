"""
jar_engine.model: Core JVM bytecode binary model and archive handler.
"""

from .constants import (
    JAVA_MAGIC, JavaMajorVersion, ConstantTag, ReferenceKind,
    AccessFlags, VerificationTypeTag, Opcode, FIXED_OPCODE_LENGTHS
)
from .cp_info import (
    decode_modified_utf8, encode_modified_utf8,
    CpEntry, CpPlaceholder,
    CONSTANT_Utf8_info, CONSTANT_Integer_info, CONSTANT_Float_info,
    CONSTANT_Long_info, CONSTANT_Double_info, CONSTANT_Class_info,
    CONSTANT_String_info, CONSTANT_Fieldref_info, CONSTANT_Methodref_info,
    CONSTANT_InterfaceMethodref_info, CONSTANT_NameAndType_info,
    CONSTANT_MethodHandle_info, CONSTANT_MethodType_info,
    CONSTANT_Dynamic_info, CONSTANT_InvokeDynamic_info,
    CONSTANT_Module_info, CONSTANT_Package_info,
    ConstantPool
)
from .member_info import (
    AttributeInfo, RawAttribute, ExceptionTableEntry, CodeAttribute,
    SignatureAttribute, SourceFileAttribute, ConstantValueAttribute,
    ExceptionsAttribute, StackMapTableAttribute, BootstrapMethod,
    BootstrapMethodsAttribute, MemberInfo, FieldInfo, MethodInfo,
    parse_attribute
)
from .class_file import ClassFile, ClassFormatError
from .jar_archive import JarArchive, Manifest, JarStats, wrap_manifest_line

__all__ = [
    "JAVA_MAGIC", "JavaMajorVersion", "ConstantTag", "ReferenceKind",
    "AccessFlags", "VerificationTypeTag", "Opcode", "FIXED_OPCODE_LENGTHS",
    "decode_modified_utf8", "encode_modified_utf8",
    "CpEntry", "CpPlaceholder",
    "CONSTANT_Utf8_info", "CONSTANT_Integer_info", "CONSTANT_Float_info",
    "CONSTANT_Long_info", "CONSTANT_Double_info", "CONSTANT_Class_info",
    "CONSTANT_String_info", "CONSTANT_Fieldref_info", "CONSTANT_Methodref_info",
    "CONSTANT_InterfaceMethodref_info", "CONSTANT_NameAndType_info",
    "CONSTANT_MethodHandle_info", "CONSTANT_MethodType_info",
    "CONSTANT_Dynamic_info", "CONSTANT_InvokeDynamic_info",
    "CONSTANT_Module_info", "CONSTANT_Package_info",
    "ConstantPool",
    "AttributeInfo", "RawAttribute", "ExceptionTableEntry", "CodeAttribute",
    "SignatureAttribute", "SourceFileAttribute", "ConstantValueAttribute",
    "ExceptionsAttribute", "StackMapTableAttribute", "BootstrapMethod",
    "BootstrapMethodsAttribute", "MemberInfo", "FieldInfo", "MethodInfo",
    "parse_attribute",
    "ClassFile", "ClassFormatError",
    "JarArchive", "Manifest", "JarStats", "wrap_manifest_line",
]

