"""
jar_engine.model.constants: JVM binary format constants, opcodes, access flags, and tags.
Conforming to The Java Virtual Machine Specification (Java SE 21 Edition).
"""

from enum import IntEnum, IntFlag
from typing import Dict

# Magic number identifying a valid Java class file
JAVA_MAGIC = 0xCAFEBABE

class JavaMajorVersion(IntEnum):
    JAVA_1_1 = 45
    JAVA_1_2 = 46
    JAVA_1_3 = 47
    JAVA_1_4 = 48
    JAVA_5 = 49
    JAVA_6 = 50
    JAVA_7 = 51
    JAVA_8 = 52
    JAVA_9 = 53
    JAVA_10 = 54
    JAVA_11 = 55
    JAVA_12 = 56
    JAVA_13 = 57
    JAVA_14 = 58
    JAVA_15 = 59
    JAVA_16 = 60
    JAVA_17 = 61
    JAVA_18 = 62
    JAVA_19 = 63
    JAVA_20 = 64
    JAVA_21 = 65


class ConstantTag(IntEnum):
    """All 17 constant pool tags defined in JVMS §4.4."""
    UTF8 = 1
    INTEGER = 3
    FLOAT = 4
    LONG = 5
    DOUBLE = 6
    CLASS = 7
    STRING = 8
    FIELDREF = 9
    METHODREF = 10
    INTERFACE_METHODREF = 11
    NAME_AND_TYPE = 12
    METHOD_HANDLE = 15
    METHOD_TYPE = 16
    DYNAMIC = 17
    INVOKE_DYNAMIC = 18
    MODULE = 19
    PACKAGE = 20


class ReferenceKind(IntEnum):
    """Reference kinds for CONSTANT_MethodHandle_info (JVMS §4.4.8)."""
    REF_getField = 1
    REF_getStatic = 2
    REF_putField = 3
    REF_putStatic = 4
    REF_invokeVirtual = 5
    REF_invokeStatic = 6
    REF_invokeSpecial = 7
    REF_newInvokeSpecial = 8
    REF_invokeInterface = 9


class AccessFlags(IntFlag):
    """JVM class, field, and method access and property flags (JVMS §4.1, §4.5, §4.6)."""
    PUBLIC = 0x0001
    PRIVATE = 0x0002
    PROTECTED = 0x0004
    STATIC = 0x0008
    FINAL = 0x0010
    SUPER = 0x0020
    SYNCHRONIZED = 0x0020
    OPEN = 0x0020
    VOLATILE = 0x0040
    BRIDGE = 0x0040
    TRANSITIVE = 0x0040
    TRANSIENT = 0x0080
    VARARGS = 0x0080
    STATIC_PHASE = 0x0080
    NATIVE = 0x0100
    INTERFACE = 0x0200
    ABSTRACT = 0x0400
    STRICT = 0x0800
    SYNTHETIC = 0x1000
    ANNOTATION = 0x2000
    ENUM = 0x4000
    MANDATED = 0x8000
    MODULE = 0x8000


class VerificationTypeTag(IntEnum):
    """StackMapTable verification type info tags (JVMS §4.7.4)."""
    TOP = 0
    INTEGER = 1
    FLOAT = 2
    DOUBLE = 3
    LONG = 4
    NULL = 5
    UNINITIALIZED_THIS = 6
    OBJECT = 7
    UNINITIALIZED = 8


class Opcode(IntEnum):
    """Standard JVM instruction opcodes (JVMS §6.5)."""
    NOP = 0x00
    ACONST_NULL = 0x01
    ICONST_M1 = 0x02
    ICONST_0 = 0x03
    ICONST_1 = 0x04
    ICONST_2 = 0x05
    ICONST_3 = 0x06
    ICONST_4 = 0x07
    ICONST_5 = 0x08
    LCONST_0 = 0x09
    LCONST_1 = 0x0A
    FCONST_0 = 0x0B
    FCONST_1 = 0x0C
    FCONST_2 = 0x0D
    DCONST_0 = 0x0E
    DCONST_1 = 0x0F
    BIPUSH = 0x10
    SIPUSH = 0x11
    LDC = 0x12
    LDC_W = 0x13
    LDC2_W = 0x14
    ILOAD = 0x15
    LLOAD = 0x16
    FLOAD = 0x17
    DLOAD = 0x18
    ALOAD = 0x19
    ILOAD_0 = 0x1A
    ILOAD_1 = 0x1B
    ILOAD_2 = 0x1C
    ILOAD_3 = 0x1D
    LLOAD_0 = 0x1E
    LLOAD_1 = 0x1F
    LLOAD_2 = 0x20
    LLOAD_3 = 0x21
    FLOAD_0 = 0x22
    FLOAD_1 = 0x23
    FLOAD_2 = 0x24
    FLOAD_3 = 0x25
    DLOAD_0 = 0x26
    DLOAD_1 = 0x27
    DLOAD_2 = 0x28
    DLOAD_3 = 0x29
    ALOAD_0 = 0x2A
    ALOAD_1 = 0x2B
    ALOAD_2 = 0x2C
    ALOAD_3 = 0x2D
    IALOAD = 0x2E
    LALOAD = 0x2F
    FALOAD = 0x30
    DALOAD = 0x31
    AALOAD = 0x32
    BALOAD = 0x33
    CALOAD = 0x34
    SALOAD = 0x35
    ISTORE = 0x36
    LSTORE = 0x37
    FSTORE = 0x38
    DSTORE = 0x39
    ASTORE = 0x3A
    ISTORE_0 = 0x3B
    ISTORE_1 = 0x3C
    ISTORE_2 = 0x3D
    ISTORE_3 = 0x3E
    LSTORE_0 = 0x3F
    LSTORE_1 = 0x40
    LSTORE_2 = 0x41
    LSTORE_3 = 0x42
    FSTORE_0 = 0x43
    FSTORE_1 = 0x44
    FSTORE_2 = 0x45
    FSTORE_3 = 0x46
    DSTORE_0 = 0x47
    DSTORE_1 = 0x48
    DSTORE_2 = 0x49
    DSTORE_3 = 0x4A
    ASTORE_0 = 0x4B
    ASTORE_1 = 0x4C
    ASTORE_2 = 0x4D
    ASTORE_3 = 0x4E
    IASTORE = 0x4F
    LASTORE = 0x50
    FASTORE = 0x51
    DASTORE = 0x52
    AASTORE = 0x53
    BASTORE = 0x54
    CASTORE = 0x55
    SASTORE = 0x56
    POP = 0x57
    POP2 = 0x58
    DUP = 0x59
    DUP_X1 = 0x5A
    DUP_X2 = 0x5B
    DUP2 = 0x5C
    DUP2_X1 = 0x5D
    DUP2_X2 = 0x5E
    SWAP = 0x5F
    IADD = 0x60
    LADD = 0x61
    FADD = 0x62
    DADD = 0x63
    ISUB = 0x64
    LSUB = 0x65
    FSUB = 0x66
    DSUB = 0x67
    IMUL = 0x68
    LMUL = 0x69
    FMUL = 0x6A
    DMUL = 0x6B
    IDIV = 0x6C
    LDIV = 0x6D
    FDIV = 0x6E
    DDIV = 0x6F
    IREM = 0x70
    LREM = 0x71
    FREM = 0x72
    DREM = 0x73
    INEG = 0x74
    LNEG = 0x75
    FNEG = 0x76
    DNEG = 0x77
    ISHL = 0x78
    LSHL = 0x79
    ISHR = 0x7A
    LSHR = 0x7B
    IUSHR = 0x7C
    LUSHR = 0x7D
    IAND = 0x7E
    LAND = 0x7F
    IOR = 0x80
    LOR = 0x81
    IXOR = 0x82
    LXOR = 0x83
    IINC = 0x84
    I2L = 0x85
    I2F = 0x86
    I2D = 0x87
    L2I = 0x88
    L2F = 0x89
    L2D = 0x8A
    F2I = 0x8B
    F2L = 0x8C
    F2D = 0x8D
    D2I = 0x8E
    D2L = 0x8F
    D2F = 0x90
    I2B = 0x91
    I2C = 0x92
    I2S = 0x93
    LCMP = 0x94
    FCMPL = 0x95
    FCMPG = 0x96
    DCMPL = 0x97
    DCMPG = 0x98
    IFEQ = 0x99
    IFNE = 0x9A
    IFLT = 0x9B
    IFGE = 0x9C
    IFGT = 0x9D
    IFLE = 0x9E
    IF_ICMPEQ = 0x9F
    IF_ICMPNE = 0xA0
    IF_ICMPLT = 0xA1
    IF_ICMPGE = 0xA2
    IF_ICMPGT = 0xA3
    IF_ICMPLE = 0xA4
    IF_ACMPEQ = 0xA5
    IF_ACMPNE = 0xA6
    GOTO = 0xA7
    JSR = 0xA8
    RET = 0xA9
    TABLESWITCH = 0xAA
    LOOKUPSWITCH = 0xAB
    IRETURN = 0xAC
    LRETURN = 0xAD
    FRETURN = 0xAE
    DRETURN = 0xAF
    ARETURN = 0xB0
    RETURN = 0xB1
    GETSTATIC = 0xB2
    PUTSTATIC = 0xB3
    GETFIELD = 0xB4
    PUTFIELD = 0xB5
    INVOKEVIRTUAL = 0xB6
    INVOKESPECIAL = 0xB7
    INVOKESTATIC = 0xB8
    INVOKEINTERFACE = 0xB9
    INVOKEDYNAMIC = 0xBA
    NEW = 0xBB
    NEWARRAY = 0xBC
    ANEWARRAY = 0xBD
    ARRAYLENGTH = 0xBE
    ATHROW = 0xBF
    CHECKCAST = 0xC0
    INSTANCEOF = 0xC1
    MONITORENTER = 0xC2
    MONITOREXIT = 0xC3
    WIDE = 0xC4
    MULTIANEWARRAY = 0xC5
    IFNULL = 0xC6
    IFNONNULL = 0xC7
    GOTO_W = 0xC8
    JSR_W = 0xC9


# Fixed instruction length lookup for instructions where operand length is constant.
# Does not include variable-length opcodes (tableswitch, lookupswitch, wide).
FIXED_OPCODE_LENGTHS: Dict[int, int] = {
    # 1-byte opcodes (no operand)
    0x00: 1, 0x01: 1, 0x02: 1, 0x03: 1, 0x04: 1, 0x05: 1, 0x06: 1, 0x07: 1,
    0x08: 1, 0x09: 1, 0x0A: 1, 0x0B: 1, 0x0C: 1, 0x0D: 1, 0x0E: 1, 0x0F: 1,
    0x1A: 1, 0x1B: 1, 0x1C: 1, 0x1D: 1, 0x1E: 1, 0x1F: 1, 0x20: 1, 0x21: 1,
    0x22: 1, 0x23: 1, 0x24: 1, 0x25: 1, 0x26: 1, 0x27: 1, 0x28: 1, 0x29: 1,
    0x2A: 1, 0x2B: 1, 0x2C: 1, 0x2D: 1, 0x2E: 1, 0x2F: 1, 0x30: 1, 0x31: 1,
    0x32: 1, 0x33: 1, 0x34: 1, 0x35: 1, 0x3B: 1, 0x3C: 1, 0x3D: 1, 0x3E: 1,
    0x3F: 1, 0x40: 1, 0x41: 1, 0x42: 1, 0x43: 1, 0x44: 1, 0x45: 1, 0x46: 1,
    0x47: 1, 0x48: 1, 0x49: 1, 0x4A: 1, 0x4B: 1, 0x4C: 1, 0x4D: 1, 0x4E: 1,
    0x4F: 1, 0x50: 1, 0x51: 1, 0x52: 1, 0x53: 1, 0x54: 1, 0x55: 1, 0x56: 1,
    0x57: 1, 0x58: 1, 0x59: 1, 0x5A: 1, 0x5B: 1, 0x5C: 1, 0x5D: 1, 0x5E: 1,
    0x5F: 1, 0x60: 1, 0x61: 1, 0x62: 1, 0x63: 1, 0x64: 1, 0x65: 1, 0x66: 1,
    0x67: 1, 0x68: 1, 0x69: 1, 0x6A: 1, 0x6B: 1, 0x6C: 1, 0x6D: 1, 0x6E: 1,
    0x6F: 1, 0x70: 1, 0x71: 1, 0x72: 1, 0x73: 1, 0x74: 1, 0x75: 1, 0x76: 1,
    0x77: 1, 0x78: 1, 0x79: 1, 0x7A: 1, 0x7B: 1, 0x7C: 1, 0x7D: 1, 0x7E: 1,
    0x7F: 1, 0x80: 1, 0x81: 1, 0x82: 1, 0x83: 1, 0x85: 1, 0x86: 1, 0x87: 1,
    0x88: 1, 0x89: 1, 0x8A: 1, 0x8B: 1, 0x8C: 1, 0x8D: 1, 0x8E: 1, 0x8F: 1,
    0x90: 1, 0x91: 1, 0x92: 1, 0x93: 1, 0x94: 1, 0x95: 1, 0x96: 1, 0x97: 1,
    0x98: 1, 0xAC: 1, 0xAD: 1, 0xAE: 1, 0xAF: 1, 0xB0: 1, 0xB1: 1, 0xBE: 1,
    0xBF: 1, 0xC2: 1, 0xC3: 1,

    # 2-byte opcodes (1 operand byte)
    0x10: 2,  # bipush
    0x12: 2,  # ldc
    0x15: 2, 0x16: 2, 0x17: 2, 0x18: 2, 0x19: 2,  # iload..aload
    0x36: 2, 0x37: 2, 0x38: 2, 0x39: 2, 0x3A: 2,  # istore..astore
    0xA9: 2,  # ret
    0xBC: 2,  # newarray

    # 3-byte opcodes (2 operand bytes)
    0x11: 3,  # sipush
    0x13: 3,  # ldc_w
    0x14: 3,  # ldc2_w
    0x84: 3,  # iinc
    0x99: 3, 0x9A: 3, 0x9B: 3, 0x9C: 3, 0x9D: 3, 0x9E: 3,  # ifeq..ifle
    0x9F: 3, 0xA0: 3, 0xA1: 3, 0xA2: 3, 0xA3: 3, 0xA4: 3,  # if_icmp*
    0xA5: 3, 0xA6: 3,  # if_acmp*
    0xA7: 3,  # goto
    0xA8: 3,  # jsr
    0xB2: 3,  # getstatic
    0xB3: 3,  # putstatic
    0xB4: 3,  # getfield
    0xB5: 3,  # putfield
    0xB6: 3,  # invokevirtual
    0xB7: 3,  # invokespecial
    0xB8: 3,  # invokestatic
    0xBB: 3,  # new
    0xBD: 3,  # anewarray
    0xC0: 3,  # checkcast
    0xC1: 3,  # instanceof
    0xC6: 3,  # ifnull
    0xC7: 3,  # ifnonnull

    # 4-byte opcodes (3 operand bytes)
    0xC5: 4,  # multianewarray

    # 5-byte opcodes (4 operand bytes)
    0xB9: 5,  # invokeinterface
    0xBA: 5,  # invokedynamic
    0xC8: 5,  # goto_w
    0xC9: 5,  # jsr_w
}
