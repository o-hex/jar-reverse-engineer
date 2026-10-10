"""
tests.fixtures.make_fixtures: Automated Deterministic Synthetic JAR Fixture Generator.
Builds the 4 synthetic test JAR archives in tests/fixtures/generated_jars/ in pure Python
using jar_engine.model, conforming strictly to PROJECT.md and TEST_INFRA.md specifications.
"""

from __future__ import annotations
import struct
from pathlib import Path
from typing import Dict, Optional, List

from jar_engine.model.constants import (
    Opcode, AccessFlags, ConstantTag, ReferenceKind
)
from jar_engine.model.cp_info import (
    ConstantPool,
    CONSTANT_MethodHandle_info,
    CONSTANT_InvokeDynamic_info,
    CONSTANT_MethodType_info,
)
from jar_engine.model.member_info import (
    MethodInfo,
    FieldInfo,
    CodeAttribute,
    SignatureAttribute,
    StackMapTableAttribute,
    ExceptionTableEntry,
    BootstrapMethodsAttribute,
    BootstrapMethod,
)
from jar_engine.model.class_file import ClassFile
from jar_engine.model.jar_archive import JarArchive, Manifest


def create_base_class(
    this_class_name: str,
    super_class_name: Optional[str] = "java/lang/Object",
    interfaces: Optional[List[str]] = None,
    access_flags: int = AccessFlags.PUBLIC,
    major_version: int = 52,
    minor_version: int = 0
) -> ClassFile:
    """Helper to initialize a ClassFile instance with basic constant pool entries."""
    cp = ConstantPool()
    this_c = cp.add_class(this_class_name)
    super_c = cp.add_class(super_class_name) if super_class_name else 0
    ifaces = [cp.add_class(i) for i in (interfaces or [])]
    return ClassFile(
        magic=0xCAFEBABE,
        minor_version=minor_version,
        major_version=major_version,
        constant_pool=cp,
        access_flags=access_flags,
        this_class=this_c,
        super_class=super_c,
        interfaces=ifaces,
        fields=[],
        methods=[],
        attributes=[]
    )


# ============================================================================
# JAR 1: control_flow_flattened.jar
# ============================================================================

def build_control_flow_flattened_jar() -> JarArchive:
    """
    Builds control_flow_flattened.jar:
    - DispatcherService: state-variable tableswitch/goto dispatch loop,
      algebraic opaque predicates ((x^x)==0, (x-x)==0), dead blocks, and NOPs.
    - ComplexFlow: lookupswitch with sparse keys.
    """
    archive = JarArchive()

    # 1. DispatcherService
    cf = create_base_class("com/synthetic/flattened/DispatcherService")
    cp = cf.constant_pool
    c_name = cp.add_utf8("Code")

    # Method 1: dispatchAlgorithm(I)I
    # State-variable switch dispatch loop:
    # PC 0..4: iconst_0, istore_1, goto PC 5
    # PC 5..7: iload_1, tableswitch (pad 1 byte)
    # PC 8..35: default_off=59, low=0, high=3, case0=30, case1=40, case2=49, case3=59
    # PC 36..45 (State 0): iload_0, bipush 10, iadd, istore_0, iconst_1, istore_1, goto PC 5 (offset -38)
    # PC 46..54 (State 1): iload_0, iconst_2, imul, istore_0, iconst_2, istore_1, goto PC 5 (offset -47)
    # PC 55..64 (State 2): iload_0, bipush 5, isub, istore_0, iconst_3, istore_1, goto PC 5 (offset -57)
    # PC 65..66 (State 3/Exit): iload_0, ireturn
    disp_code = (
        bytes([Opcode.ICONST_0, Opcode.ISTORE_1, Opcode.GOTO]) + struct.pack(">h", 3) +
        bytes([Opcode.ILOAD_1, Opcode.TABLESWITCH, 0]) +
        struct.pack(">iiiiiii", 59, 0, 3, 30, 40, 49, 59) +
        bytes([Opcode.ILOAD_0, Opcode.BIPUSH, 10, Opcode.IADD, Opcode.ISTORE_0, Opcode.ICONST_1, Opcode.ISTORE_1, Opcode.GOTO]) + struct.pack(">h", -38) +
        bytes([Opcode.ILOAD_0, Opcode.ICONST_2, Opcode.IMUL, Opcode.ISTORE_0, Opcode.ICONST_2, Opcode.ISTORE_1, Opcode.GOTO]) + struct.pack(">h", -47) +
        bytes([Opcode.ILOAD_0, Opcode.BIPUSH, 5, Opcode.ISUB, Opcode.ISTORE_0, Opcode.ICONST_3, Opcode.ISTORE_1, Opcode.GOTO]) + struct.pack(">h", -57) +
        bytes([Opcode.ILOAD_0, Opcode.IRETURN])
    )
    m_disp = MethodInfo(
        access_flags=AccessFlags.PUBLIC | AccessFlags.STATIC,
        name_index=cp.add_utf8("dispatchAlgorithm"),
        descriptor_index=cp.add_utf8("(I)I"),
        attributes=[CodeAttribute(c_name, 3, 2, disp_code)]
    )
    cf.methods.append(m_disp)

    # Method 2: opaqueBranchAlgorithm(I)I
    # Opaque predicates: (x^x)==0 and (x-x)==0
    # PC 0..5: iload_0, iload_0, ixor, ifeq (pc 3 + 9 = 12)
    # PC 6..11: bipush 99, istore_0, goto (pc 9 + 19 = 28)
    # PC 12..17: iload_0, iload_0, isub, ifeq (pc 15 + 9 = 24)
    # PC 18..23: bipush 42, istore_0, goto (pc 21 + 7 = 28)
    # PC 24..27: iload_0, iconst_1, iadd, istore_0
    # PC 28..29: iload_0, ireturn
    opaque_code = (
        bytes([Opcode.ILOAD_0, Opcode.ILOAD_0, Opcode.IXOR, Opcode.IFEQ]) + struct.pack(">h", 9) +
        bytes([Opcode.BIPUSH, 99, Opcode.ISTORE_0, Opcode.GOTO]) + struct.pack(">h", 19) +
        bytes([Opcode.ILOAD_0, Opcode.ILOAD_0, Opcode.ISUB, Opcode.IFEQ]) + struct.pack(">h", 9) +
        bytes([Opcode.BIPUSH, 42, Opcode.ISTORE_0, Opcode.GOTO]) + struct.pack(">h", 7) +
        bytes([Opcode.ILOAD_0, Opcode.ICONST_1, Opcode.IADD, Opcode.ISTORE_0]) +
        bytes([Opcode.ILOAD_0, Opcode.IRETURN])
    )
    m_opaque = MethodInfo(
        access_flags=AccessFlags.PUBLIC | AccessFlags.STATIC,
        name_index=cp.add_utf8("opaqueBranchAlgorithm"),
        descriptor_index=cp.add_utf8("(I)I"),
        attributes=[CodeAttribute(c_name, 2, 1, opaque_code)]
    )
    cf.methods.append(m_opaque)

    # Method 3: redundantChainsAndNops()V
    # 4 NOPs, compacted gotos, dead code trailing unconditional return
    redundant_code = (
        bytes([Opcode.NOP, Opcode.NOP, Opcode.NOP, Opcode.NOP]) +
        bytes([Opcode.GOTO]) + struct.pack(">h", 3) +
        bytes([Opcode.GOTO]) + struct.pack(">h", 3) +
        bytes([Opcode.RETURN]) +
        bytes([Opcode.BIPUSH, 123, Opcode.ISTORE_0, Opcode.RETURN])
    )
    m_redundant = MethodInfo(
        access_flags=AccessFlags.PUBLIC | AccessFlags.STATIC,
        name_index=cp.add_utf8("redundantChainsAndNops"),
        descriptor_index=cp.add_utf8("()V"),
        attributes=[CodeAttribute(c_name, 1, 1, redundant_code)]
    )
    cf.methods.append(m_redundant)

    # Method 4: loopDispatch(I)I
    # State machine with back-edge loop
    # PC 0..4: iconst_0, istore_1, goto PC 5
    # PC 5..7: iload_1, tableswitch (pad 1 byte)
    # PC 8..27: default_off=34 (PC 41), low=0, high=1, case0=21 (PC 28), case1=27 (PC 34)
    # PC 28..33 (State 0): iload_0, ifeq PC 41 (offset 41 - 29 = 12), iconst_1, istore_1, goto PC 5 (offset -28)
    # PC 34..40 (State 1): iload_0, iconst_1, isub, istore_0, iconst_0, istore_1, goto PC 5 (offset -35)
    # PC 41..42 (Exit): iload_0, ireturn
    # PC 5: iload_1(1), tableswitch(1), pad(1), default(4), low(4), high(4), case0(4), case1(4) = 23 bytes (PCs 5..27)
    loop_code = (
        bytes([Opcode.ICONST_0, Opcode.ISTORE_1, Opcode.GOTO]) + struct.pack(">h", 3) +
        bytes([Opcode.ILOAD_1, Opcode.TABLESWITCH, 0]) +
        struct.pack(">iiiii", 36, 0, 1, 23, 29) +
        bytes([Opcode.ILOAD_0, Opcode.IFEQ]) + struct.pack(">h", 16) + bytes([Opcode.ICONST_1, Opcode.ISTORE_1, Opcode.GOTO]) + struct.pack(">h", -31) +
        bytes([Opcode.ILOAD_0, Opcode.ICONST_1, Opcode.ISUB, Opcode.ISTORE_0, Opcode.ICONST_0, Opcode.ISTORE_1, Opcode.GOTO]) + struct.pack(">h", -40) +
        bytes([Opcode.ILOAD_0, Opcode.IRETURN])
    )
    m_loop = MethodInfo(
        access_flags=AccessFlags.PUBLIC | AccessFlags.STATIC,
        name_index=cp.add_utf8("loopDispatch"),
        descriptor_index=cp.add_utf8("(I)I"),
        attributes=[CodeAttribute(c_name, 2, 2, loop_code)]
    )
    cf.methods.append(m_loop)
    archive.add_class(cf)

    # 2. ComplexFlow (LOOKUPSWITCH)
    cf_complex = create_base_class("com/synthetic/flattened/ComplexFlow")
    cp_c = cf_complex.constant_pool
    lookup_code = (
        bytes([Opcode.ILOAD_0, Opcode.LOOKUPSWITCH, 0, 0]) +
        struct.pack(">ii", 44, 3) +
        struct.pack(">ii", 100, 35) +
        struct.pack(">ii", 500, 38) +
        struct.pack(">ii", 1000, 41) +
        bytes([Opcode.BIPUSH, 1, Opcode.IRETURN]) +
        bytes([Opcode.BIPUSH, 2, Opcode.IRETURN]) +
        bytes([Opcode.BIPUSH, 3, Opcode.IRETURN]) +
        bytes([Opcode.ICONST_0, Opcode.IRETURN])
    )
    cf_complex.methods.append(MethodInfo(
        access_flags=AccessFlags.PUBLIC | AccessFlags.STATIC,
        name_index=cp_c.add_utf8("switchAlgorithm"),
        descriptor_index=cp_c.add_utf8("(I)I"),
        attributes=[CodeAttribute(cp_c.add_utf8("Code"), 2, 1, lookup_code)]
    ))
    archive.add_class(cf_complex)

    manifest = Manifest()
    manifest.set_main_class("com.synthetic.flattened.DispatcherService")
    archive.manifest = manifest
    return archive


# ============================================================================
# JAR 2: constant_obfuscated.jar
# ============================================================================

def build_constant_obfuscated_jar() -> JarArchive:
    """
    Builds constant_obfuscated.jar:
    - ArithmeticObfuscated: arithmetic/bitwise/shift chains, 2-slot long constants.
    - StringObfuscated: Caesar/XOR static decryptor loop, call sites decoding
      to "HELLO" and "SECRET".
    - DynamicInvoker: invokedynamic bootstrap methods (StringConcatFactory, LambdaMetafactory).
    """
    archive = JarArchive()

    # 1. ArithmeticObfuscated
    cf_arith = create_base_class("com/synthetic/constants/ArithmeticObfuscated")
    cp_a = cf_arith.constant_pool
    c_name_a = cp_a.add_utf8("Code")

    # Method: evaluateArithmetic()I -> (10 + 20) * 2 - 15 = 45
    code_arith = bytes([
        Opcode.BIPUSH, 10,
        Opcode.BIPUSH, 20,
        Opcode.IADD,
        Opcode.BIPUSH, 2,
        Opcode.IMUL,
        Opcode.BIPUSH, 15,
        Opcode.ISUB,
        Opcode.IRETURN
    ])
    cf_arith.methods.append(MethodInfo(
        access_flags=AccessFlags.PUBLIC | AccessFlags.STATIC,
        name_index=cp_a.add_utf8("evaluateArithmetic"),
        descriptor_index=cp_a.add_utf8("()I"),
        attributes=[CodeAttribute(c_name_a, 2, 0, code_arith)]
    ))

    # Method: evaluateBitwise()I -> (15 | 16) & 5 = 5
    code_bit = bytes([
        Opcode.BIPUSH, 15,
        Opcode.BIPUSH, 16,
        Opcode.IOR,
        Opcode.BIPUSH, 5,
        Opcode.IAND,
        Opcode.IRETURN
    ])
    cf_arith.methods.append(MethodInfo(
        access_flags=AccessFlags.PUBLIC | AccessFlags.STATIC,
        name_index=cp_a.add_utf8("evaluateBitwise"),
        descriptor_index=cp_a.add_utf8("()I"),
        attributes=[CodeAttribute(c_name_a, 2, 0, code_bit)]
    ))

    # Method: evaluateShift()I -> 1 << 4 = 16
    code_shift = bytes([
        Opcode.ICONST_1,
        Opcode.ICONST_4,
        Opcode.ISHL,
        Opcode.IRETURN
    ])
    cf_arith.methods.append(MethodInfo(
        access_flags=AccessFlags.PUBLIC | AccessFlags.STATIC,
        name_index=cp_a.add_utf8("evaluateShift"),
        descriptor_index=cp_a.add_utf8("()I"),
        attributes=[CodeAttribute(c_name_a, 2, 0, code_shift)]
    ))

    # Method: evaluateLongArithmetic()J -> 1000L + 2000L = 3000L (2-slot CP)
    l1 = cp_a.add_long(1000)
    l2 = cp_a.add_long(2000)
    code_long = (
        bytes([Opcode.LDC2_W]) + struct.pack(">H", l1) +
        bytes([Opcode.LDC2_W]) + struct.pack(">H", l2) +
        bytes([Opcode.LADD, Opcode.LRETURN])
    )
    cf_arith.methods.append(MethodInfo(
        access_flags=AccessFlags.PUBLIC | AccessFlags.STATIC,
        name_index=cp_a.add_utf8("evaluateLongArithmetic"),
        descriptor_index=cp_a.add_utf8("()J"),
        attributes=[CodeAttribute(c_name_a, 4, 0, code_long)]
    ))

    # Method: evaluateFloatArithmetic()F -> 3.5f - 1.5f = 2.0f
    f1 = cp_a.add_float(3.5)
    f2 = cp_a.add_float(1.5)
    code_float = bytes([
        Opcode.LDC, f1,
        Opcode.LDC, f2,
        Opcode.FSUB,
        Opcode.FRETURN
    ])
    cf_arith.methods.append(MethodInfo(
        access_flags=AccessFlags.PUBLIC | AccessFlags.STATIC,
        name_index=cp_a.add_utf8("evaluateFloatArithmetic"),
        descriptor_index=cp_a.add_utf8("()F"),
        attributes=[CodeAttribute(c_name_a, 2, 0, code_float)]
    ))
    archive.add_class(cf_arith)

    # 2. StringObfuscated
    cf_str = create_base_class("com/synthetic/constants/StringObfuscated")
    cp_s = cf_str.constant_pool
    c_name_s = cp_s.add_utf8("Code")
    str_cls = cp_s.add_class("java/lang/String")
    toCharArray_ref = cp_s.add_member_ref(ConstantTag.METHODREF, "java/lang/String", "toCharArray", "()[C")
    str_init_ref = cp_s.add_member_ref(ConstantTag.METHODREF, "java/lang/String", "<init>", "([C)V")

    # Decryptor method: dec(String s) -> String
    # Loops over characters subtracting 1 from char code.
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
    cf_str.methods.append(MethodInfo(
        access_flags=AccessFlags.PUBLIC | AccessFlags.STATIC,
        name_index=cp_s.add_utf8("dec"),
        descriptor_index=cp_s.add_utf8("(Ljava/lang/String;)Ljava/lang/String;"),
        attributes=[CodeAttribute(c_name_s, 4, 3, dec_code)]
    ))

    # Call site 1: getSecretMessage() -> dec("IFMMP") -> "HELLO"
    enc_idx1 = cp_s.add_string("IFMMP")
    dec_ref = cp_s.add_member_ref(ConstantTag.METHODREF, "com/synthetic/constants/StringObfuscated", "dec", "(Ljava/lang/String;)Ljava/lang/String;")
    call1_code = bytes([Opcode.LDC, enc_idx1, Opcode.INVOKESTATIC]) + struct.pack(">H", dec_ref) + bytes([Opcode.ARETURN])
    cf_str.methods.append(MethodInfo(
        access_flags=AccessFlags.PUBLIC | AccessFlags.STATIC,
        name_index=cp_s.add_utf8("getSecretMessage"),
        descriptor_index=cp_s.add_utf8("()Ljava/lang/String;"),
        attributes=[CodeAttribute(c_name_s, 2, 0, call1_code)]
    ))

    # Call site 2: getDatabaseKey() -> dec("TFDSFU") -> "SECRET"
    enc_idx2 = cp_s.add_string("TFDSFU")
    call2_code = bytes([Opcode.LDC, enc_idx2, Opcode.INVOKESTATIC]) + struct.pack(">H", dec_ref) + bytes([Opcode.ARETURN])
    cf_str.methods.append(MethodInfo(
        access_flags=AccessFlags.PUBLIC | AccessFlags.STATIC,
        name_index=cp_s.add_utf8("getDatabaseKey"),
        descriptor_index=cp_s.add_utf8("()Ljava/lang/String;"),
        attributes=[CodeAttribute(c_name_s, 2, 0, call2_code)]
    ))
    archive.add_class(cf_str)

    # 3. DynamicInvoker
    cf_indy = create_base_class("com/synthetic/constants/DynamicInvoker")
    cp_d = cf_indy.constant_pool
    c_name_d = cp_d.add_utf8("Code")

    # BSM 0: StringConcatFactory.makeConcatWithConstants
    bsm0_mh_idx = cp_d.add_member_ref(
        ConstantTag.METHODREF,
        "java/lang/invoke/StringConcatFactory",
        "makeConcatWithConstants",
        "(Ljava/lang/invoke/MethodHandles$Lookup;Ljava/lang/String;Ljava/lang/invoke/MethodType;Ljava/lang/String;[Ljava/lang/Object;)Ljava/lang/invoke/CallSite;"
    )
    mh0_idx = len(cp_d._entries)
    cp_d._entries.append(CONSTANT_MethodHandle_info(reference_kind=ReferenceKind.REF_invokeStatic, reference_index=bsm0_mh_idx))
    recipe_idx = cp_d.add_string("Secret: \u0002")
    arg_idx = cp_d.add_string("42")
    bm0 = BootstrapMethod(bootstrap_method_ref=mh0_idx, bootstrap_arguments=[recipe_idx, arg_idx])

    # BSM 1: LambdaMetafactory.metafactory
    bsm1_mh_idx = cp_d.add_member_ref(
        ConstantTag.METHODREF,
        "java/lang/invoke/LambdaMetafactory",
        "metafactory",
        "(Ljava/lang/invoke/MethodHandles$Lookup;Ljava/lang/String;Ljava/lang/invoke/MethodType;Ljava/lang/invoke/MethodType;Ljava/lang/invoke/MethodHandle;Ljava/lang/invoke/MethodType;)Ljava/lang/invoke/CallSite;"
    )
    mh1_idx = len(cp_d._entries)
    cp_d._entries.append(CONSTANT_MethodHandle_info(reference_kind=ReferenceKind.REF_invokeStatic, reference_index=bsm1_mh_idx))
    sam_type_idx = len(cp_d._entries)
    cp_d._entries.append(CONSTANT_MethodType_info(descriptor_index=cp_d.add_utf8("()V")))
    impl_m_idx = cp_d.add_member_ref(ConstantTag.METHODREF, "com/synthetic/constants/DynamicInvoker", "workerAction", "()V")
    impl_mh_idx = len(cp_d._entries)
    cp_d._entries.append(CONSTANT_MethodHandle_info(reference_kind=ReferenceKind.REF_invokeStatic, reference_index=impl_m_idx))
    inst_type_idx = len(cp_d._entries)
    cp_d._entries.append(CONSTANT_MethodType_info(descriptor_index=cp_d.add_utf8("()V")))
    bm1 = BootstrapMethod(bootstrap_method_ref=mh1_idx, bootstrap_arguments=[sam_type_idx, impl_mh_idx, inst_type_idx])

    bsm_attr = BootstrapMethodsAttribute(
        attribute_name_index=cp_d.add_utf8("BootstrapMethods"),
        bootstrap_methods=[bm0, bm1]
    )
    cf_indy.attributes.append(bsm_attr)

    # Method 1: concatSecret()Ljava/lang/String;
    nat0_idx = cp_d.add_name_and_type("makeConcatWithConstants", "()Ljava/lang/String;")
    indy0_idx = len(cp_d._entries)
    cp_d._entries.append(CONSTANT_InvokeDynamic_info(bootstrap_method_attr_index=0, name_and_type_index=nat0_idx))
    code_indy0 = bytes([Opcode.INVOKEDYNAMIC]) + struct.pack(">H", indy0_idx) + bytes([0, 0, Opcode.ARETURN])
    cf_indy.methods.append(MethodInfo(
        access_flags=AccessFlags.PUBLIC | AccessFlags.STATIC,
        name_index=cp_d.add_utf8("concatSecret"),
        descriptor_index=cp_d.add_utf8("()Ljava/lang/String;"),
        attributes=[CodeAttribute(c_name_d, 1, 0, code_indy0)]
    ))

    # Method 2: getWorker()Ljava/lang/Runnable;
    nat1_idx = cp_d.add_name_and_type("run", "()Ljava/lang/Runnable;")
    indy1_idx = len(cp_d._entries)
    cp_d._entries.append(CONSTANT_InvokeDynamic_info(bootstrap_method_attr_index=1, name_and_type_index=nat1_idx))
    code_indy1 = bytes([Opcode.INVOKEDYNAMIC]) + struct.pack(">H", indy1_idx) + bytes([0, 0, Opcode.ARETURN])
    cf_indy.methods.append(MethodInfo(
        access_flags=AccessFlags.PUBLIC | AccessFlags.STATIC,
        name_index=cp_d.add_utf8("getWorker"),
        descriptor_index=cp_d.add_utf8("()Ljava/lang/Runnable;"),
        attributes=[CodeAttribute(c_name_d, 1, 0, code_indy1)]
    ))

    # Method 3: workerAction()V
    cf_indy.methods.append(MethodInfo(
        access_flags=AccessFlags.PUBLIC | AccessFlags.STATIC,
        name_index=cp_d.add_utf8("workerAction"),
        descriptor_index=cp_d.add_utf8("()V"),
        attributes=[CodeAttribute(c_name_d, 1, 0, bytes([Opcode.RETURN]))]
    ))
    archive.add_class(cf_indy)

    manifest = Manifest()
    manifest.set_main_class("com.synthetic.constants.ArithmeticObfuscated")
    archive.manifest = manifest
    return archive


# ============================================================================
# JAR 3: stripped_symbols.jar
# ============================================================================

def build_stripped_symbols_jar() -> JarArchive:
    """
    Builds stripped_symbols.jar:
    - a/a: interface with field a: I and method a()V
    - a/b: interface extends a/a with field a: String and methods a()V, b()V
    - a/c: interface extends a/a with field a: boolean and methods a()V, c()V
    - a/d: class implements a/b, a/c (diamond hierarchy) with fields a: I, b: String,
           methods <init>, a()V, b()V, c()V, a(int), and pinned toString()
    - a/e: class extends a/d, overrides a()V
    - a/f: caller/entry point invoking methods and accessing fields
    - META-INF/services/a.a provider registration file.
    """
    archive = JarArchive()

    # 1. a/a (interface)
    cf_a = create_base_class("a/a", access_flags=AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT)
    cp_a = cf_a.constant_pool
    cf_a.fields.append(FieldInfo(
        AccessFlags.PUBLIC | AccessFlags.STATIC | AccessFlags.FINAL,
        cp_a.add_utf8("a"), cp_a.add_utf8("I"), []
    ))
    cf_a.methods.append(MethodInfo(
        AccessFlags.PUBLIC | AccessFlags.ABSTRACT,
        cp_a.add_utf8("a"), cp_a.add_utf8("()V"), []
    ))
    archive.add_class(cf_a)

    # 2. a/b (interface extends a/a)
    cf_b = create_base_class("a/b", interfaces=["a/a"], access_flags=AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT)
    cp_b = cf_b.constant_pool
    cf_b.fields.append(FieldInfo(
        AccessFlags.PUBLIC | AccessFlags.STATIC | AccessFlags.FINAL,
        cp_b.add_utf8("a"), cp_b.add_utf8("Ljava/lang/String;"), []
    ))
    cf_b.methods.append(MethodInfo(
        AccessFlags.PUBLIC | AccessFlags.ABSTRACT,
        cp_b.add_utf8("a"), cp_b.add_utf8("()V"), []
    ))
    cf_b.methods.append(MethodInfo(
        AccessFlags.PUBLIC | AccessFlags.ABSTRACT,
        cp_b.add_utf8("b"), cp_b.add_utf8("()V"), []
    ))
    archive.add_class(cf_b)

    # 3. a/c (interface extends a/a)
    cf_c = create_base_class("a/c", interfaces=["a/a"], access_flags=AccessFlags.PUBLIC | AccessFlags.INTERFACE | AccessFlags.ABSTRACT)
    cp_c = cf_c.constant_pool
    cf_c.fields.append(FieldInfo(
        AccessFlags.PUBLIC | AccessFlags.STATIC | AccessFlags.FINAL,
        cp_c.add_utf8("a"), cp_c.add_utf8("Z"), []
    ))
    cf_c.methods.append(MethodInfo(
        AccessFlags.PUBLIC | AccessFlags.ABSTRACT,
        cp_c.add_utf8("a"), cp_c.add_utf8("()V"), []
    ))
    cf_c.methods.append(MethodInfo(
        AccessFlags.PUBLIC | AccessFlags.ABSTRACT,
        cp_c.add_utf8("c"), cp_c.add_utf8("()V"), []
    ))
    archive.add_class(cf_c)

    # 4. a/d (implements a/b, a/c)
    cf_d = create_base_class("a/d", interfaces=["a/b", "a/c"])
    cp_d = cf_d.constant_pool
    c_name_d = cp_d.add_utf8("Code")

    cf_d.fields.append(FieldInfo(AccessFlags.PRIVATE, cp_d.add_utf8("a"), cp_d.add_utf8("I"), []))
    cf_d.fields.append(FieldInfo(AccessFlags.PUBLIC, cp_d.add_utf8("b"), cp_d.add_utf8("Ljava/lang/String;"), []))

    obj_init_ref = cp_d.add_member_ref(ConstantTag.METHODREF, "java/lang/Object", "<init>", "()V")
    code_init = bytes([Opcode.ALOAD_0, Opcode.INVOKESPECIAL]) + struct.pack(">H", obj_init_ref) + bytes([Opcode.RETURN])
    cf_d.methods.append(MethodInfo(
        AccessFlags.PUBLIC, cp_d.add_utf8("<init>"), cp_d.add_utf8("()V"),
        [CodeAttribute(c_name_d, 1, 1, code_init)]
    ))
    cf_d.methods.append(MethodInfo(
        AccessFlags.PUBLIC, cp_d.add_utf8("a"), cp_d.add_utf8("()V"),
        [CodeAttribute(c_name_d, 1, 1, bytes([Opcode.RETURN]))]
    ))
    cf_d.methods.append(MethodInfo(
        AccessFlags.PUBLIC, cp_d.add_utf8("b"), cp_d.add_utf8("()V"),
        [CodeAttribute(c_name_d, 1, 1, bytes([Opcode.RETURN]))]
    ))
    cf_d.methods.append(MethodInfo(
        AccessFlags.PUBLIC, cp_d.add_utf8("c"), cp_d.add_utf8("()V"),
        [CodeAttribute(c_name_d, 1, 1, bytes([Opcode.RETURN]))]
    ))
    # Overload on name 'a': a(int) -> int
    cf_d.methods.append(MethodInfo(
        AccessFlags.PUBLIC, cp_d.add_utf8("a"), cp_d.add_utf8("(I)I"),
        [CodeAttribute(c_name_d, 1, 2, bytes([Opcode.ILOAD_1, Opcode.IRETURN]))]
    ))
    # Pinned JDK method: toString() -> String
    d_str_ref = cp_d.add_string("d_instance")
    cf_d.methods.append(MethodInfo(
        AccessFlags.PUBLIC, cp_d.add_utf8("toString"), cp_d.add_utf8("()Ljava/lang/String;"),
        [CodeAttribute(c_name_d, 1, 1, bytes([Opcode.LDC, d_str_ref, Opcode.ARETURN]))]
    ))
    archive.add_class(cf_d)

    # 5. a/e (extends a/d)
    cf_e = create_base_class("a/e", super_class_name="a/d")
    cp_e = cf_e.constant_pool
    c_name_e = cp_e.add_utf8("Code")
    d_init_ref = cp_e.add_member_ref(ConstantTag.METHODREF, "a/d", "<init>", "()V")
    code_e_init = bytes([Opcode.ALOAD_0, Opcode.INVOKESPECIAL]) + struct.pack(">H", d_init_ref) + bytes([Opcode.RETURN])
    cf_e.methods.append(MethodInfo(
        AccessFlags.PUBLIC, cp_e.add_utf8("<init>"), cp_e.add_utf8("()V"),
        [CodeAttribute(c_name_e, 1, 1, code_e_init)]
    ))
    cf_e.methods.append(MethodInfo(
        AccessFlags.PUBLIC, cp_e.add_utf8("a"), cp_e.add_utf8("()V"),
        [CodeAttribute(c_name_e, 1, 1, bytes([Opcode.RETURN]))]
    ))
    archive.add_class(cf_e)

    # 6. a/f (caller / entry point)
    cf_f = create_base_class("a/f")
    cp_f = cf_f.constant_pool
    c_name_f = cp_f.add_utf8("Code")
    d_cls_ref = cp_f.add_class("a/d")
    d_init_f = cp_f.add_member_ref(ConstantTag.METHODREF, "a/d", "<init>", "()V")
    d_a_f = cp_f.add_member_ref(ConstantTag.METHODREF, "a/d", "a", "()V")
    b_b_f = cp_f.add_member_ref(ConstantTag.INTERFACE_METHODREF, "a/b", "b", "()V")
    c_c_f = cp_f.add_member_ref(ConstantTag.INTERFACE_METHODREF, "a/c", "c", "()V")
    d_fld_b = cp_f.add_member_ref(ConstantTag.FIELDREF, "a/d", "b", "Ljava/lang/String;")

    code_main = (
        bytes([Opcode.NEW]) + struct.pack(">H", d_cls_ref) +
        bytes([Opcode.DUP, Opcode.INVOKESPECIAL]) + struct.pack(">H", d_init_f) +
        bytes([Opcode.ASTORE_1, Opcode.ALOAD_1, Opcode.INVOKEVIRTUAL]) + struct.pack(">H", d_a_f) +
        bytes([Opcode.ALOAD_1, Opcode.INVOKEINTERFACE]) + struct.pack(">H", b_b_f) + bytes([1, 0]) +
        bytes([Opcode.ALOAD_1, Opcode.INVOKEINTERFACE]) + struct.pack(">H", c_c_f) + bytes([1, 0]) +
        bytes([Opcode.ALOAD_1, Opcode.GETFIELD]) + struct.pack(">H", d_fld_b) +
        bytes([Opcode.POP, Opcode.RETURN])
    )
    cf_f.methods.append(MethodInfo(
        AccessFlags.PUBLIC | AccessFlags.STATIC,
        cp_f.add_utf8("main"), cp_f.add_utf8("([Ljava/lang/String;)V"),
        [CodeAttribute(c_name_f, 2, 2, code_main)]
    ))
    archive.add_class(cf_f)

    # Manifest and SPI registration
    manifest = Manifest()
    manifest.set_main_class("a.f")
    archive.manifest = manifest
    archive.set_resource("META-INF/services/a.a", b"a.d\n")
    return archive


# ============================================================================
# JAR 4: corrupted_structures.jar
# ============================================================================

def build_corrupted_structures_jar() -> JarArchive:
    """
    Builds corrupted_structures.jar:
    - CorruptSignatures: malformed generic signatures violating JVMS §4.7.9
    - CorruptExceptions: inverted, zero-length, out-of-bounds, and self-catching handlers
    - CorruptStackMaps: illegal StackMapTable frame tags (e.g., reserved tag 220)
    - DuplicateSynthetics: duplicate synthetic fields/methods, self-referential bridge loops
    """
    archive = JarArchive()

    # 1. CorruptSignatures
    cf1 = create_base_class("com/synthetic/corrupted/CorruptSignatures")
    cp1 = cf1.constant_pool
    sig_name = cp1.add_utf8("Signature")
    c_name1 = cp1.add_utf8("Code")

    # Class signature: unclosed parameter <T:bad
    cf1.attributes.append(SignatureAttribute(sig_name, cp1.add_utf8("<T:bad")))

    # Field signature: missing terminal semicolon Lbroken/Field
    cf1.fields.append(FieldInfo(
        AccessFlags.PRIVATE, cp1.add_utf8("corruptField"), cp1.add_utf8("Ljava/lang/Object;"),
        [SignatureAttribute(sig_name, cp1.add_utf8("Lbroken/Field"))]
    ))

    # Method signature: missing return type (Ljava/lang/String;)
    cf1.methods.append(MethodInfo(
        AccessFlags.PUBLIC, cp1.add_utf8("corruptMethod"), cp1.add_utf8("(Ljava/lang/String;)V"),
        [SignatureAttribute(sig_name, cp1.add_utf8("(Ljava/lang/String;)"))]
    ))

    # Valid method signature: ()V
    cf1.methods.append(MethodInfo(
        AccessFlags.PUBLIC, cp1.add_utf8("validMethod"), cp1.add_utf8("()V"),
        [
            SignatureAttribute(sig_name, cp1.add_utf8("()V")),
            CodeAttribute(c_name1, 1, 1, bytes([Opcode.RETURN]))
        ]
    ))
    archive.add_class(cf1)

    # 2. CorruptExceptions
    cf2 = create_base_class("com/synthetic/corrupted/CorruptExceptions")
    cp2 = cf2.constant_pool
    code2 = bytes([Opcode.NOP] * 39 + [Opcode.RETURN])
    ex_entries = [
        ExceptionTableEntry(start_pc=25, end_pc=10, handler_pc=5, catch_type=0),   # Inverted range
        ExceptionTableEntry(start_pc=5, end_pc=5, handler_pc=10, catch_type=0),     # Zero-length range
        ExceptionTableEntry(start_pc=0, end_pc=80, handler_pc=10, catch_type=0),    # Out of bounds (> 40)
        ExceptionTableEntry(start_pc=0, end_pc=20, handler_pc=10, catch_type=0),    # Self-catching loop
        ExceptionTableEntry(start_pc=0, end_pc=15, handler_pc=30, catch_type=0),    # Valid handler
        ExceptionTableEntry(start_pc=0, end_pc=15, handler_pc=30, catch_type=0),    # Duplicate handler
    ]
    cf2.methods.append(MethodInfo(
        AccessFlags.PUBLIC, cp2.add_utf8("corruptExceptionTable"), cp2.add_utf8("()V"),
        [CodeAttribute(cp2.add_utf8("Code"), 2, 1, code2, exception_table=ex_entries)]
    ))
    archive.add_class(cf2)

    # 3. CorruptStackMaps
    cf3 = create_base_class("com/synthetic/corrupted/CorruptStackMaps")
    cp3 = cf3.constant_pool
    sm_name = cp3.add_utf8("StackMapTable")
    bad_sm_code = CodeAttribute(
        cp3.add_utf8("Code"), 1, 1, bytes([Opcode.RETURN]),
        attributes=[StackMapTableAttribute(sm_name, struct.pack(">HB", 1, 220))]  # Reserved tag 220
    )
    cf3.methods.append(MethodInfo(
        AccessFlags.PUBLIC, cp3.add_utf8("badStackMapFrame"), cp3.add_utf8("()V"),
        [bad_sm_code]
    ))
    archive.add_class(cf3)

    # 4. DuplicateSynthetics
    cf4 = create_base_class("com/synthetic/corrupted/DuplicateSynthetics")
    cp4 = cf4.constant_pool
    c_name4 = cp4.add_utf8("Code")

    # Duplicate fields: regular int shadowVal vs synthetic String shadowVal
    cf4.fields.append(FieldInfo(AccessFlags.PUBLIC, cp4.add_utf8("shadowVal"), cp4.add_utf8("I"), []))
    cf4.fields.append(FieldInfo(AccessFlags.PUBLIC | AccessFlags.SYNTHETIC, cp4.add_utf8("shadowVal"), cp4.add_utf8("Ljava/lang/String;"), []))

    # Duplicate methods: regular doWork vs synthetic doWork
    cf4.methods.append(MethodInfo(
        AccessFlags.PUBLIC, cp4.add_utf8("doWork"), cp4.add_utf8("()V"),
        [CodeAttribute(c_name4, 1, 1, bytes([Opcode.RETURN]))]
    ))
    cf4.methods.append(MethodInfo(
        AccessFlags.PUBLIC | AccessFlags.SYNTHETIC, cp4.add_utf8("doWork"), cp4.add_utf8("()V"),
        []
    ))

    # Recursive bridge method calling itself
    bridge_ref = cp4.add_member_ref(ConstantTag.METHODREF, "com/synthetic/corrupted/DuplicateSynthetics", "bridgeCall", "()V")
    bridge_code = bytes([Opcode.INVOKEVIRTUAL]) + struct.pack(">H", bridge_ref) + bytes([Opcode.RETURN])
    cf4.methods.append(MethodInfo(
        AccessFlags.PUBLIC | AccessFlags.BRIDGE, cp4.add_utf8("bridgeCall"), cp4.add_utf8("()V"),
        [CodeAttribute(c_name4, 1, 1, bridge_code)]
    ))
    archive.add_class(cf4)

    manifest = Manifest()
    manifest.set_main_class("com.synthetic.corrupted.CorruptSignatures")
    archive.manifest = manifest
    return archive


# ============================================================================
# Main Fixture Management API
# ============================================================================

def ensure_fixtures(output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    Generates and saves all 4 deterministic synthetic JARs if missing.
    Returns dictionary of paths keyed by fixture identifier.
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "generated_jars"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    builders = {
        "control_flow_flattened": ("control_flow_flattened.jar", build_control_flow_flattened_jar),
        "constant_obfuscated": ("constant_obfuscated.jar", build_constant_obfuscated_jar),
        "stripped_symbols": ("stripped_symbols.jar", build_stripped_symbols_jar),
        "corrupted_structures": ("corrupted_structures.jar", build_corrupted_structures_jar),
    }

    result: Dict[str, Path] = {}
    for key, (filename, builder_fn) in builders.items():
        jar_path = output_dir / filename
        if not jar_path.is_file() or jar_path.stat().st_size == 0:
            archive = builder_fn()
            archive.save(jar_path)
        result[key] = jar_path
        # Also provide convenient alias keys
        result[filename] = jar_path
        result[f"{key}_jar"] = jar_path

    return result


def make_all_fixtures(output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Alias for ensure_fixtures matching explorer handoff convention."""
    return ensure_fixtures(output_dir)


if __name__ == "__main__":
    paths = ensure_fixtures()
    print("[+] Synthetic fixture generation completed:")
    for name, p in paths.items():
        if name.endswith(".jar"):
            print(f"    - {name}: {p.stat().st_size} bytes ({p})")
