"""
tests/tier4_application/test_workload_1_full_obfuscated_jar_recovery.py: Real-world Workload 1.
Full End-to-End Recovery Pipeline on heavily obfuscated JAR archive.
Exercises:
- M1 Core Model & Walker: F1-F7
- M2 Sanitizer: F8-F12
- M3 CFG & Normalizer: F13-F19
- M4 Capability Triage: F20-F23
- M5 Symbol Renamer & Repackager: F24-F30, F34
"""

import json
import struct
from pathlib import Path
import pytest

from jar_engine.model.constants import Opcode, AccessFlags, ConstantTag
from jar_engine.model.class_file import ClassFile
from jar_engine.model.cp_info import (
    ConstantPool, CONSTANT_String_info, CONSTANT_Class_info
)
from jar_engine.model.member_info import (
    MethodInfo, FieldInfo, CodeAttribute, SignatureAttribute,
    ExceptionTableEntry
)
from jar_engine.model.jar_archive import JarArchive, Manifest
from jar_engine.walker.hierarchy import InheritanceHierarchy
from jar_engine.walker.call_graph import CallGraph
from jar_engine.walker.dependency_slice import DependencySlicer
from jar_engine.sanitizer import (
    BytecodeSanitizer,
    verify_sanitized_class_structural_integrity,
)
from jar_engine.normalization import MethodNormalizer
from jar_engine.triage import CapabilityDomain, CapabilityTriager, ContextualSlicer
from jar_engine.renamer import (
    SymbolRenamer, SymbolMapping, JarRepackager
)


def create_full_obfuscated_archive() -> JarArchive:
    """
    Constructs a rich synthetic archive combining all major obfuscation and corruption patterns:
    1. a/a: Interface with malformed generic signature, stripped name, final field.
    2. a/b: Diamond-join class with flattened switch/goto dispatch loop,
            Caesar decryptor routine, inverted exception table, and pinned toString override.
    3. a/c: Entry point class invoking methods, reading fields, and registered as Main-Class.
    4. META-INF/services/a.a: SPI registration pointing to a.b.
    """
    archive = JarArchive()

    # 1. Interface a/a
    cf_a = ClassFile(
        magic=0xCAFEBABE, minor_version=0, major_version=52, constant_pool=ConstantPool(),
        access_flags=AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT,
        this_class=0, super_class=0, interfaces=[], fields=[], methods=[], attributes=[]
    )
    cp_a = cf_a.constant_pool
    cf_a.this_class = cp_a.add_class("a/a")
    cf_a.super_class = cp_a.add_class("java/lang/Object")
    # Corrupted generic signature violating JVMS §4.7.9
    cf_a.attributes.append(SignatureAttribute(cp_a.add_utf8("Signature"), cp_a.add_utf8("<T:unclosed")))
    cf_a.fields.append(FieldInfo(
        AccessFlags.PUBLIC | AccessFlags.STATIC | AccessFlags.FINAL,
        cp_a.add_utf8("TAG"), cp_a.add_utf8("Ljava/lang/String;"), []
    ))
    cf_a.methods.append(MethodInfo(
        AccessFlags.PUBLIC | AccessFlags.ABSTRACT,
        cp_a.add_utf8("execute"), cp_a.add_utf8("(I)I"), []
    ))
    archive.add_class(cf_a)

    # 2. Class a/b implements a/a
    cf_b = ClassFile(
        magic=0xCAFEBABE, minor_version=0, major_version=52, constant_pool=ConstantPool(),
        access_flags=AccessFlags.PUBLIC, this_class=0, super_class=0, interfaces=[],
        fields=[], methods=[], attributes=[]
    )
    cp_b = cf_b.constant_pool
    cf_b.this_class = cp_b.add_class("a/b")
    cf_b.super_class = cp_b.add_class("java/lang/Object")
    cf_b.interfaces = [cp_b.add_class("a/a")]

    # Crypto API references for architectural triage
    cp_b.add_class("javax/crypto/Cipher")
    cp_b.add_member_ref(ConstantTag.METHODREF, "javax/crypto/Cipher", "getInstance", "(Ljava/lang/String;)Ljavax/crypto/Cipher;")
    cp_b.add_string("AES/CBC/PKCS5Padding")

    # Constructor
    obj_init = cp_b.add_member_ref(ConstantTag.METHODREF, "java/lang/Object", "<init>", "()V")
    cf_b.methods.append(MethodInfo(
        AccessFlags.PUBLIC, cp_b.add_utf8("<init>"), cp_b.add_utf8("()V"),
        [CodeAttribute(cp_b.add_utf8("Code"), 1, 1, bytes([Opcode.ALOAD_0, Opcode.INVOKESPECIAL]) + struct.pack(">H", obj_init) + bytes([Opcode.RETURN]))]
    ))

    # Caesar Decryptor: dec(String) -> String
    str_cls = cp_b.add_class("java/lang/String")
    toCharArray_ref = cp_b.add_member_ref(ConstantTag.METHODREF, "java/lang/String", "toCharArray", "()[C")
    str_init_ref = cp_b.add_member_ref(ConstantTag.METHODREF, "java/lang/String", "<init>", "([C)V")
    dec_code = (
        bytes([Opcode.ALOAD_0, Opcode.INVOKEVIRTUAL]) + struct.pack(">H", toCharArray_ref) +
        bytes([Opcode.ASTORE_1, Opcode.ICONST_0, Opcode.ISTORE_2, Opcode.ILOAD_2, Opcode.ALOAD_1, Opcode.ARRAYLENGTH, Opcode.IF_ICMPGE]) +
        struct.pack(">h", 18) +
        bytes([Opcode.ALOAD_1, Opcode.ILOAD_2, Opcode.ALOAD_1, Opcode.ILOAD_2, Opcode.CALOAD, Opcode.ICONST_1, Opcode.ISUB, Opcode.I2C, Opcode.CASTORE, Opcode.IINC, 2, 1, Opcode.GOTO]) +
        struct.pack(">h", -18) +
        bytes([Opcode.NEW]) + struct.pack(">H", str_cls) +
        bytes([Opcode.DUP, Opcode.ALOAD_1, Opcode.INVOKESPECIAL]) + struct.pack(">H", str_init_ref) +
        bytes([Opcode.ARETURN])
    )
    cf_b.methods.append(MethodInfo(
        AccessFlags.PUBLIC | AccessFlags.STATIC, cp_b.add_utf8("dec"), cp_b.add_utf8("(Ljava/lang/String;)Ljava/lang/String;"),
        [CodeAttribute(cp_b.add_utf8("Code"), 4, 3, dec_code)]
    ))

    # Method execute(I)I with flattened dispatch loop and corrupt exception table
    disp_code = (
        bytes([Opcode.ICONST_0, Opcode.ISTORE_1, Opcode.GOTO]) + struct.pack(">h", 3) +
        bytes([Opcode.ILOAD_1, Opcode.TABLESWITCH, 0]) +
        struct.pack(">iiiiiii", 59, 0, 3, 30, 40, 49, 59) +
        bytes([Opcode.ILOAD_0, Opcode.BIPUSH, 10, Opcode.IADD, Opcode.ISTORE_0, Opcode.ICONST_1, Opcode.ISTORE_1, Opcode.GOTO]) + struct.pack(">h", -38) +
        bytes([Opcode.ILOAD_0, Opcode.ICONST_2, Opcode.IMUL, Opcode.ISTORE_0, Opcode.ICONST_2, Opcode.ISTORE_1, Opcode.GOTO]) + struct.pack(">h", -47) +
        bytes([Opcode.ILOAD_0, Opcode.BIPUSH, 5, Opcode.ISUB, Opcode.ISTORE_0, Opcode.ICONST_3, Opcode.ISTORE_1, Opcode.GOTO]) + struct.pack(">h", -57) +
        bytes([Opcode.ILOAD_0, Opcode.IRETURN])
    )
    # Corrupt exception entries: inverted and self-catching
    ex_entries = [
        ExceptionTableEntry(start_pc=30, end_pc=10, handler_pc=5, catch_type=0),   # Inverted
        ExceptionTableEntry(start_pc=0, end_pc=20, handler_pc=10, catch_type=0),    # Self-catching
        ExceptionTableEntry(start_pc=0, end_pc=15, handler_pc=30, catch_type=0),    # Valid
    ]
    cf_b.methods.append(MethodInfo(
        AccessFlags.PUBLIC, cp_b.add_utf8("execute"), cp_b.add_utf8("(I)I"),
        [CodeAttribute(cp_b.add_utf8("Code"), 3, 2, disp_code, exception_table=ex_entries)]
    ))

    # Encrypted string call site
    enc_idx = cp_b.add_string("IFMMP")  # Decodes to "HELLO"
    dec_ref = cp_b.add_member_ref(ConstantTag.METHODREF, "a/b", "dec", "(Ljava/lang/String;)Ljava/lang/String;")
    secret_code = bytes([Opcode.LDC, enc_idx, Opcode.INVOKESTATIC]) + struct.pack(">H", dec_ref) + bytes([Opcode.ARETURN])
    cf_b.methods.append(MethodInfo(
        AccessFlags.PUBLIC | AccessFlags.STATIC, cp_b.add_utf8("getSecretMessage"), cp_b.add_utf8("()Ljava/lang/String;"),
        [CodeAttribute(cp_b.add_utf8("Code"), 2, 0, secret_code)]
    ))

    # Pinned method toString()
    d_str = cp_b.add_string("InstanceB")
    cf_b.methods.append(MethodInfo(
        AccessFlags.PUBLIC, cp_b.add_utf8("toString"), cp_b.add_utf8("()Ljava/lang/String;"),
        [CodeAttribute(cp_b.add_utf8("Code"), 1, 1, bytes([Opcode.LDC, d_str, Opcode.ARETURN]))]
    ))
    archive.add_class(cf_b)

    # 3. Class a/c (Caller and Main class)
    cf_c = ClassFile(
        magic=0xCAFEBABE, minor_version=0, major_version=52, constant_pool=ConstantPool(),
        access_flags=AccessFlags.PUBLIC, this_class=0, super_class=0, interfaces=[],
        fields=[], methods=[], attributes=[]
    )
    cp_c = cf_c.constant_pool
    cf_c.this_class = cp_c.add_class("a/c")
    cf_c.super_class = cp_c.add_class("java/lang/Object")

    b_cls_ref = cp_c.add_class("a/b")
    b_init_ref = cp_c.add_member_ref(ConstantTag.METHODREF, "a/b", "<init>", "()V")
    b_exec_ref = cp_c.add_member_ref(ConstantTag.METHODREF, "a/b", "execute", "(I)I")
    main_code = (
        bytes([Opcode.NEW]) + struct.pack(">H", b_cls_ref) +
        bytes([Opcode.DUP, Opcode.INVOKESPECIAL]) + struct.pack(">H", b_init_ref) +
        bytes([Opcode.ASTORE_1, Opcode.ALOAD_1, Opcode.BIPUSH, 7, Opcode.INVOKEVIRTUAL]) + struct.pack(">H", b_exec_ref) +
        bytes([Opcode.POP, Opcode.RETURN])
    )
    cf_c.methods.append(MethodInfo(
        AccessFlags.PUBLIC | AccessFlags.STATIC, cp_c.add_utf8("main"), cp_c.add_utf8("([Ljava/lang/String;)V"),
        [CodeAttribute(cp_c.add_utf8("Code"), 2, 2, main_code)]
    ))
    archive.add_class(cf_c)

    # Manifest and SPI registration
    manifest = Manifest()
    manifest.set_main_class("a.c")
    archive.manifest = manifest
    archive.set_resource("META-INF/services/a.a", b"a.b\n")

    return archive


def test_workload_1_full_obfuscated_jar_recovery(temp_workspace: Path):
    """
    Workload 1: Full E2E recovery of an obfuscated JAR archive.
    1. Parse and index whole archive with call graph & inheritance DAG.
    2. Preprocess & sanitize non-standard bytecode (corrupt signatures, inverted exceptions).
    3. Normalize control flow (de-flatten dispatch loops, decrypt strings, fold constants).
    4. Architectural capability triage & security audit.
    5. Semantic symbol refactoring across diamond hierarchy preserving JDK pins.
    6. Repackage into valid JAR with synchronized RFC 822 Manifest and SPI services.
    7. Verify 100% decompiler structural integrity across all recovered classes.
    """
    # Step 1: Save initial obfuscated archive to disk and load via JarArchive
    initial_jar = temp_workspace / "initial_obfuscated.jar"
    raw_archive = create_full_obfuscated_archive()
    raw_archive.save(initial_jar)
    archive = JarArchive.from_file(initial_jar)
    assert len(archive.classes) == 3

    # Verify initial defects exist before recovery
    is_valid_a, issues_a = verify_sanitized_class_structural_integrity(archive.classes["a/a"])
    assert is_valid_a is False
    assert len(issues_a) >= 1  # Malformed generic signature detected

    is_valid_b, issues_b = verify_sanitized_class_structural_integrity(archive.classes["a/b"])
    assert is_valid_b is False
    assert len(issues_b) >= 1  # Inverted exception handler detected

    # Step 2: Sanitization pass (M2)
    sanitizer = BytecodeSanitizer()
    san_report = sanitizer.sanitize_archive(archive)
    assert san_report.total_signatures_stripped >= 1
    assert san_report.total_exceptions_fixed >= 1

    # Step 3: Normalization pass (M3)
    normalizer = MethodNormalizer()
    total_complexity_reduced = 0
    total_strings_decrypted = 0

    for c_name, cf in archive.classes.items():
        for m in cf.methods:
            stats = normalizer.normalize_method(cf, m)
            total_complexity_reduced += stats.complexity_reduced
            total_strings_decrypted += stats.strings_decrypted

    # Assert cyclomatic complexity demonstrably dropped and strings decrypted
    assert total_complexity_reduced >= 1
    assert total_strings_decrypted >= 1
    # Plaintext "HELLO" reconstructed in constant pool of a/b
    cp_b = archive.classes["a/b"].constant_pool
    all_strings = [cp_b.resolve_utf8(e.string_index) for _, e in cp_b.entries() if isinstance(e, CONSTANT_String_info)]
    assert "HELLO" in all_strings

    # Step 4: Architectural capability triage (M4)
    triager = CapabilityTriager()
    triage_report = triager.triage_archive(archive)
    assert triage_report.domain_distribution[CapabilityDomain.CRYPTOGRAPHY] >= 1
    assert "a/b" in triage_report.high_priority_classes

    # Step 5: Semantic symbol refactoring (M5)
    mapping_data = {
        "classes": {
            "a/a": "com/enterprise/core/ServiceApi",
            "a/b": "com/enterprise/crypto/SecureEngineImpl",
            "a/c": "com/enterprise/app/ApplicationLauncher"
        },
        "methods": {
            "a/a#execute#(I)I": "processTransaction",
            "a/b#toString#()Ljava/lang/String;": "customToString"  # Pinned method: should be skipped!
        },
        "fields": {
            "a/a#TAG": "IDENTIFIER_TAG"
        }
    }
    mapping = SymbolMapping.from_dict(mapping_data)
    renamer = SymbolRenamer()
    ren_report = renamer.apply_mapping(archive, mapping)

    assert ren_report.classes_renamed == 3
    assert ren_report.methods_renamed >= 2  # Interface execute + Class execute updated
    assert ren_report.pinned_skipped >= 1   # toString preserved

    # Step 6: Valid JAR repackaging & SPI synchronization (M5)
    recovered_jar = temp_workspace / "recovered_engine.jar"
    repackager = JarRepackager()
    repackager.repackage_to_file(archive, recovered_jar, mapping=mapping)
    assert recovered_jar.is_file() and recovered_jar.stat().st_size > 0

    # Step 7: Final Structural Integrity & Decompiler Export Verification
    final_archive = JarArchive.from_file(recovered_jar)
    assert "com/enterprise/core/ServiceApi" in final_archive.classes
    assert "com/enterprise/crypto/SecureEngineImpl" in final_archive.classes
    assert "com/enterprise/app/ApplicationLauncher" in final_archive.classes

    # Verify RFC 822 Manifest synchronized
    assert final_archive.manifest is not None
    assert final_archive.manifest.get_main_class() == "com.enterprise.app.ApplicationLauncher"

    # Verify SPI service provider synchronized
    assert "META-INF/services/com.enterprise.core.ServiceApi" in final_archive.resources
    spi_content = final_archive.resources["META-INF/services/com.enterprise.core.ServiceApi"].decode("utf-8")
    assert "com.enterprise.crypto.SecureEngineImpl" in spi_content

    # Verify 100% of classes pass structural integrity with zero issues
    for c_name, cf in final_archive.classes.items():
        ok, issues = verify_sanitized_class_structural_integrity(cf)
        assert ok is True, f"Class {c_name} failed structural verification: {issues}"
        assert len(issues) == 0
