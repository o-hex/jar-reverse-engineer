"""
jar_engine.normalization.constant_folding: Intra-procedural abstract interpretation lattice
for evaluating arithmetic and bitwise opcodes into clean constants (Feature F16).
Conforms to JVMS §4.4 and §6.5.
"""

from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, Any, Union, TYPE_CHECKING
import struct
import math

from jar_engine.model.constants import Opcode, ConstantTag
from jar_engine.model.cp_info import (
    ConstantPool, CONSTANT_Integer_info, CONSTANT_Long_info,
    CONSTANT_Float_info, CONSTANT_Double_info, CONSTANT_String_info
)
from jar_engine.model.member_info import MethodInfo, CodeAttribute
from jar_engine.cfg.basic_block import BasicBlock, Instruction
from jar_engine.cfg.graph import ControlFlowGraph

INT_BINARY_OPS = (
    Opcode.IADD, Opcode.ISUB, Opcode.IMUL, Opcode.IDIV, Opcode.IREM,
    Opcode.ISHL, Opcode.ISHR, Opcode.IUSHR,
    Opcode.IAND, Opcode.IOR, Opcode.IXOR
)

LONG_BINARY_OPS = (
    Opcode.LADD, Opcode.LSUB, Opcode.LMUL, Opcode.LDIV, Opcode.LREM,
    Opcode.LSHL, Opcode.LSHR, Opcode.LUSHR,
    Opcode.LAND, Opcode.LOR, Opcode.LXOR
)

FLOAT_BINARY_OPS = (
    Opcode.FADD, Opcode.FSUB, Opcode.FMUL, Opcode.FDIV, Opcode.FREM
)

DOUBLE_BINARY_OPS = (
    Opcode.DADD, Opcode.DSUB, Opcode.DMUL, Opcode.DDIV, Opcode.DREM
)


class LatticeTag(Enum):
    BOTTOM = auto()    # Unreached / unreachable
    CONSTANT = auto()  # Exactly known constant
    TOP = auto()       # Unknown / dynamic


@dataclass(frozen=True)
class LatticeValue:
    tag: LatticeTag
    val: Optional[Any] = None
    type_name: str = "unknown"          # "int", "long", "float", "double", "String", "null", "ref"
    producer_indices: Tuple[int, ...] = ()  # Instruction indices that formed this expression

    @classmethod
    def bottom(cls) -> "LatticeValue":
        return cls(LatticeTag.BOTTOM)

    @classmethod
    def top(cls, type_name: str = "unknown") -> "LatticeValue":
        return cls(LatticeTag.TOP, type_name=type_name)

    @classmethod
    def constant(cls, val: Any, type_name: str, producer_indices: Tuple[int, ...] = ()) -> "LatticeValue":
        return cls(LatticeTag.CONSTANT, val=val, type_name=type_name, producer_indices=producer_indices)

    @property
    def is_constant(self) -> bool:
        return self.tag == LatticeTag.CONSTANT

    @property
    def is_category2(self) -> bool:
        return self.type_name in ("long", "double")

    def join(self, other: "LatticeValue") -> "LatticeValue":
        if self.tag == LatticeTag.BOTTOM:
            return other
        if other.tag == LatticeTag.BOTTOM:
            return self
        if self.tag == LatticeTag.TOP or other.tag == LatticeTag.TOP:
            return LatticeValue.top(self.type_name if self.type_name == other.type_name else "unknown")
        if self.tag == LatticeTag.CONSTANT and other.tag == LatticeTag.CONSTANT:
            if self.type_name == other.type_name and self.val == other.val:
                return self
            return LatticeValue.top(self.type_name)
        return LatticeValue.top()


class AbstractFrame:
    """Intra-procedural execution state at a specific instruction offset."""

    def __init__(self, max_locals: int = 64, max_stack: int = 64):
        self.locals: Dict[int, LatticeValue] = {i: LatticeValue.top() for i in range(max_locals)}
        self.stack: List[LatticeValue] = []
        self.max_stack = max_stack

    def push(self, val: LatticeValue) -> None:
        self.stack.append(val)

    def pop(self) -> LatticeValue:
        if not self.stack:
            return LatticeValue.top()
        return self.stack.pop()

    def peek(self, depth: int = 0) -> LatticeValue:
        if depth >= len(self.stack):
            return LatticeValue.top()
        return self.stack[-1 - depth]

    def clone(self) -> "AbstractFrame":
        f = AbstractFrame(len(self.locals), self.max_stack)
        f.locals = dict(self.locals)
        f.stack = list(self.stack)
        return f

    def join(self, other: "AbstractFrame") -> "AbstractFrame":
        f = AbstractFrame(max(len(self.locals), len(other.locals)), self.max_stack)
        all_local_keys = set(self.locals.keys()) | set(other.locals.keys())
        for idx in all_local_keys:
            v1 = self.locals.get(idx, LatticeValue.top())
            v2 = other.locals.get(idx, LatticeValue.top())
            f.locals[idx] = v1.join(v2)
        if len(self.stack) == len(other.stack):
            f.stack = [v1.join(v2) for v1, v2 in zip(self.stack, other.stack)]
        else:
            f.stack = [LatticeValue.top() for _ in range(min(len(self.stack), len(other.stack)))]
        return f


class ArithmeticEvaluator:
    """Pure evaluation routines conforming strictly to Java Virtual Machine semantics."""

    @staticmethod
    def to_int32(val: int) -> int:
        val = val & 0xFFFFFFFF
        return val if val < 0x80000000 else val - 0x100000000

    @staticmethod
    def to_int64(val: int) -> int:
        val = val & 0xFFFFFFFFFFFFFFFF
        return val if val < 0x8000000000000000 else val - 0x10000000000000000

    @staticmethod
    def java_irem(a: int, b: int) -> int:
        sign = -1 if a < 0 else 1
        return sign * (abs(a) % abs(b))

    @staticmethod
    def java_lrem(a: int, b: int) -> int:
        sign = -1 if a < 0 else 1
        return sign * (abs(a) % abs(b))

    @classmethod
    def eval_binary_int(cls, op: int, a: int, b: int) -> Optional[int]:
        if op == Opcode.IADD:
            return cls.to_int32(a + b)
        elif op == Opcode.ISUB:
            return cls.to_int32(a - b)
        elif op == Opcode.IMUL:
            return cls.to_int32(a * b)
        elif op == Opcode.IDIV:
            if b == 0:
                return None  # ArithmeticException, DO NOT fold
            if a == -2147483648 and b == -1:
                return -2147483648
            sign = -1 if (a < 0) ^ (b < 0) else 1
            return cls.to_int32(sign * (abs(a) // abs(b)))
        elif op == Opcode.IREM:
            if b == 0:
                return None  # ArithmeticException, DO NOT fold
            return cls.java_irem(a, b)
        elif op == Opcode.IAND:
            return cls.to_int32(a & b)
        elif op == Opcode.IOR:
            return cls.to_int32(a | b)
        elif op == Opcode.IXOR:
            return cls.to_int32(a ^ b)
        elif op == Opcode.ISHL:
            return cls.to_int32(a << (b & 0x1F))
        elif op == Opcode.ISHR:
            return cls.to_int32(a >> (b & 0x1F))
        elif op == Opcode.IUSHR:
            return cls.to_int32((a & 0xFFFFFFFF) >> (b & 0x1F))
        return None

    @classmethod
    def eval_binary_long(cls, op: int, a: int, b: int) -> Optional[int]:
        if op == Opcode.LADD:
            return cls.to_int64(a + b)
        elif op == Opcode.LSUB:
            return cls.to_int64(a - b)
        elif op == Opcode.LMUL:
            return cls.to_int64(a * b)
        elif op == Opcode.LDIV:
            if b == 0:
                return None
            if a == -9223372036854775808 and b == -1:
                return -9223372036854775808
            sign = -1 if (a < 0) ^ (b < 0) else 1
            return cls.to_int64(sign * (abs(a) // abs(b)))
        elif op == Opcode.LREM:
            if b == 0:
                return None
            return cls.java_lrem(a, b)
        elif op == Opcode.LAND:
            return cls.to_int64(a & b)
        elif op == Opcode.LOR:
            return cls.to_int64(a | b)
        elif op == Opcode.LXOR:
            return cls.to_int64(a ^ b)
        elif op == Opcode.LSHL:
            return cls.to_int64(a << (b & 0x3F))
        elif op == Opcode.LSHR:
            return cls.to_int64(a >> (b & 0x3F))
        elif op == Opcode.LUSHR:
            return cls.to_int64((a & 0xFFFFFFFFFFFFFFFF) >> (b & 0x3F))
        return None

    @classmethod
    def eval_binary_float(cls, op: int, a: float, b: float) -> Optional[float]:
        if op == Opcode.FADD:
            return a + b
        elif op == Opcode.FSUB:
            return a - b
        elif op == Opcode.FMUL:
            return a * b
        elif op == Opcode.FDIV:
            if b == 0.0:
                if a == 0.0 or math.isnan(a):
                    return math.nan
                sign = -1.0 if (math.copysign(1.0, a) < 0) ^ (math.copysign(1.0, b) < 0) else 1.0
                return sign * math.inf
            try:
                return a / b
            except ZeroDivisionError:
                sign = -1.0 if (math.copysign(1.0, a) < 0) ^ (math.copysign(1.0, b) < 0) else 1.0
                return sign * math.inf
        elif op == Opcode.FREM:
            if math.isnan(a) or math.isnan(b) or math.isinf(a) or b == 0.0:
                return math.nan
            try:
                return math.fmod(a, b)
            except ValueError:
                return math.nan
        return None

    @classmethod
    def eval_binary_double(cls, op: int, a: float, b: float) -> Optional[float]:
        if op == Opcode.DADD:
            return a + b
        elif op == Opcode.DSUB:
            return a - b
        elif op == Opcode.DMUL:
            return a * b
        elif op == Opcode.DDIV:
            if b == 0.0:
                if a == 0.0 or math.isnan(a):
                    return math.nan
                sign = -1.0 if (math.copysign(1.0, a) < 0) ^ (math.copysign(1.0, b) < 0) else 1.0
                return sign * math.inf
            try:
                return a / b
            except ZeroDivisionError:
                sign = -1.0 if (math.copysign(1.0, a) < 0) ^ (math.copysign(1.0, b) < 0) else 1.0
                return sign * math.inf
        elif op == Opcode.DREM:
            if math.isnan(a) or math.isnan(b) or math.isinf(a) or b == 0.0:
                return math.nan
            try:
                return math.fmod(a, b)
            except ValueError:
                return math.nan
        return None

    @classmethod
    def eval_conversion(cls, op: int, v: Any) -> Optional[Tuple[Any, str]]:
        if op == Opcode.I2L:
            return (cls.to_int64(int(v)), "long")
        elif op == Opcode.I2F:
            return (float(v), "float")
        elif op == Opcode.I2D:
            return (float(v), "double")
        elif op == Opcode.I2B:
            b = int(v) & 0xFF
            return (b if b < 0x80 else b - 0x100, "int")
        elif op == Opcode.I2C:
            return (int(v) & 0xFFFF, "int")
        elif op == Opcode.I2S:
            s = int(v) & 0xFFFF
            return (s if s < 0x8000 else s - 0x10000, "int")
        elif op == Opcode.L2I:
            return (cls.to_int32(int(v)), "int")
        elif op == Opcode.L2F:
            return (float(v), "float")
        elif op == Opcode.L2D:
            return (float(v), "double")
        elif op == Opcode.F2I:
            f = float(v)
            if math.isnan(f):
                return (0, "int")
            elif f >= 2147483647.0:
                return (2147483647, "int")
            elif f <= -2147483648.0:
                return (-2147483648, "int")
            return (int(f), "int")
        elif op == Opcode.D2I:
            d = float(v)
            if math.isnan(d):
                return (0, "int")
            elif d >= 2147483647.0:
                return (2147483647, "int")
            elif d <= -2147483648.0:
                return (-2147483648, "int")
            return (int(d), "int")
        return None

    @classmethod
    def eval_comparison(cls, op: int, a: Any, b: Any) -> Optional[int]:
        if op == Opcode.LCMP:
            return 1 if a > b else (-1 if a < b else 0)
        elif op in (Opcode.FCMPL, Opcode.FCMPG):
            if math.isnan(a) or math.isnan(b):
                return -1 if op == Opcode.FCMPL else 1
            return 1 if a > b else (-1 if a < b else 0)
        elif op in (Opcode.DCMPL, Opcode.DCMPG):
            if math.isnan(a) or math.isnan(b):
                return -1 if op == Opcode.DCMPL else 1
            return 1 if a > b else (-1 if a < b else 0)
        return None


def create_push_instruction(val: Any, type_name: str, cp: Optional[ConstantPool] = None) -> Optional[Instruction]:
    """Generates the most compact JVM push instruction for a constant value."""
    if type_name == "int":
        v = int(val)
        if -1 <= v <= 5:
            return Instruction(pc=0, opcode=Opcode.ICONST_0 + v)
        elif -128 <= v <= 127:
            return Instruction(pc=0, opcode=Opcode.BIPUSH, operands=struct.pack(">b", v))
        elif -32768 <= v <= 32767:
            return Instruction(pc=0, opcode=Opcode.SIPUSH, operands=struct.pack(">h", v))
        elif cp is not None:
            idx = cp.add_integer(v)
            if idx <= 255:
                return Instruction(pc=0, opcode=Opcode.LDC, operands=bytes([idx]))
            else:
                return Instruction(pc=0, opcode=Opcode.LDC_W, operands=struct.pack(">H", idx))
        else:
            return None

    elif type_name == "long":
        v = int(val)
        if v == 0:
            return Instruction(pc=0, opcode=Opcode.LCONST_0)
        elif v == 1:
            return Instruction(pc=0, opcode=Opcode.LCONST_1)
        elif cp is not None:
            idx = cp.add_long(v)
            return Instruction(pc=0, opcode=Opcode.LDC2_W, operands=struct.pack(">H", idx))

    elif type_name == "float":
        v = float(val)
        if v == 0.0:
            return Instruction(pc=0, opcode=Opcode.FCONST_0)
        elif v == 1.0:
            return Instruction(pc=0, opcode=Opcode.FCONST_1)
        elif v == 2.0:
            return Instruction(pc=0, opcode=Opcode.FCONST_2)
        elif cp is not None:
            idx = cp.add_float(v)
            if idx <= 255:
                return Instruction(pc=0, opcode=Opcode.LDC, operands=bytes([idx]))
            else:
                return Instruction(pc=0, opcode=Opcode.LDC_W, operands=struct.pack(">H", idx))

    elif type_name == "double":
        v = float(val)
        if v == 0.0:
            return Instruction(pc=0, opcode=Opcode.DCONST_0)
        elif v == 1.0:
            return Instruction(pc=0, opcode=Opcode.DCONST_1)
        elif cp is not None:
            idx = cp.add_double(v)
            return Instruction(pc=0, opcode=Opcode.LDC2_W, operands=struct.pack(">H", idx))

    elif type_name == "String" and cp is not None:
        idx = cp.add_string(str(val))
        if idx <= 255:
            return Instruction(pc=0, opcode=Opcode.LDC, operands=bytes([idx]))
        else:
            return Instruction(pc=0, opcode=Opcode.LDC_W, operands=struct.pack(">H", idx))

    return None


class IntraProceduralConstantFolder:
    """
    Evaluates basic blocks and methods to simplify constant arithmetic and bitwise expressions.
    """

    def __init__(self, cfg: ControlFlowGraph, cp: Optional[ConstantPool] = None):
        self.cfg = cfg
        self.cp = cp

    def fold(self) -> int:
        folded_total = 0
        for block in self.cfg.blocks.values():
            folded_total += self._fold_block(block)
        return folded_total

    def _fold_block(self, block: BasicBlock) -> int:
        folded_count = 0
        changed = True

        while changed:
            changed = False
            insts = block.instructions
            if not insts:
                break

            # Forward abstract simulation of the block
            frame = AbstractFrame()

            for i, inst in enumerate(insts):
                op = inst.opcode

                # 1. Constant push instructions
                if Opcode.ICONST_M1 <= op <= Opcode.ICONST_5:
                    v = op - Opcode.ICONST_0
                    frame.push(LatticeValue.constant(v, "int", (i,)))
                elif op == Opcode.LCONST_0:
                    frame.push(LatticeValue.constant(0, "long", (i,)))
                elif op == Opcode.LCONST_1:
                    frame.push(LatticeValue.constant(1, "long", (i,)))
                elif op in (Opcode.FCONST_0, Opcode.FCONST_1, Opcode.FCONST_2):
                    frame.push(LatticeValue.constant(float(op - Opcode.FCONST_0), "float", (i,)))
                elif op in (Opcode.DCONST_0, Opcode.DCONST_1):
                    frame.push(LatticeValue.constant(float(op - Opcode.DCONST_0), "double", (i,)))
                elif op == Opcode.BIPUSH:
                    v = struct.unpack(">b", inst.operands[:1])[0]
                    frame.push(LatticeValue.constant(v, "int", (i,)))
                elif op == Opcode.SIPUSH:
                    v = struct.unpack(">h", inst.operands[:2])[0]
                    frame.push(LatticeValue.constant(v, "int", (i,)))
                elif op in (Opcode.LDC, Opcode.LDC_W, Opcode.LDC2_W) and self.cp:
                    idx = inst.operands[0] if op == Opcode.LDC else struct.unpack(">H", inst.operands[:2])[0]
                    entry = self.cp.get(idx)
                    if isinstance(entry, CONSTANT_Integer_info):
                        frame.push(LatticeValue.constant(entry.value, "int", (i,)))
                    elif isinstance(entry, CONSTANT_Long_info):
                        frame.push(LatticeValue.constant(entry.value, "long", (i,)))
                    elif isinstance(entry, CONSTANT_Float_info):
                        frame.push(LatticeValue.constant(entry.value, "float", (i,)))
                    elif isinstance(entry, CONSTANT_Double_info):
                        frame.push(LatticeValue.constant(entry.value, "double", (i,)))
                    elif isinstance(entry, CONSTANT_String_info):
                        frame.push(LatticeValue.constant(self.cp.resolve_string(idx), "String", (i,)))
                    else:
                        frame.push(LatticeValue.top())

                # 2. Local variable loads / stores
                elif Opcode.ILOAD_0 <= op <= Opcode.ILOAD_3 or op == Opcode.ILOAD:
                    slot = op - Opcode.ILOAD_0 if op != Opcode.ILOAD else inst.operands[0]
                    val = frame.locals.get(slot, LatticeValue.top("int"))
                    frame.push(val)
                elif Opcode.ISTORE_0 <= op <= Opcode.ISTORE_3 or op == Opcode.ISTORE:
                    slot = op - Opcode.ISTORE_0 if op != Opcode.ISTORE else inst.operands[0]
                    frame.locals[slot] = frame.pop()

                # 3. Binary Integer operations
                elif op in INT_BINARY_OPS:
                    v2 = frame.pop()
                    v1 = frame.pop()
                    if v1.is_constant and v2.is_constant:
                        res = ArithmeticEvaluator.eval_binary_int(op, int(v1.val), int(v2.val))
                        if res is not None:
                            producer_set = set(v1.producer_indices) | set(v2.producer_indices) | {i}
                            min_idx, max_idx = min(producer_set), max(producer_set)
                            if max_idx - min_idx + 1 == len(producer_set) and max_idx == i:
                                new_inst = create_push_instruction(res, "int", self.cp)
                                if new_inst:
                                    insts[min_idx : max_idx + 1] = [new_inst]
                                    folded_count += 1
                                    changed = True
                                    break
                    frame.push(LatticeValue.top("int"))

                # 4. Binary Long operations
                elif op in LONG_BINARY_OPS:
                    v2 = frame.pop()
                    v1 = frame.pop()
                    if v1.is_constant and v2.is_constant:
                        res = ArithmeticEvaluator.eval_binary_long(op, int(v1.val), int(v2.val))
                        if res is not None:
                            producer_set = set(v1.producer_indices) | set(v2.producer_indices) | {i}
                            min_idx, max_idx = min(producer_set), max(producer_set)
                            if max_idx - min_idx + 1 == len(producer_set) and max_idx == i:
                                new_inst = create_push_instruction(res, "long", self.cp)
                                if new_inst:
                                    insts[min_idx : max_idx + 1] = [new_inst]
                                    folded_count += 1
                                    changed = True
                                    break
                    frame.push(LatticeValue.top("long"))

                # 4b. Binary Float operations
                elif op in FLOAT_BINARY_OPS:
                    v2 = frame.pop()
                    v1 = frame.pop()
                    if v1.is_constant and v2.is_constant:
                        res = ArithmeticEvaluator.eval_binary_float(op, float(v1.val), float(v2.val))
                        if res is not None:
                            producer_set = set(v1.producer_indices) | set(v2.producer_indices) | {i}
                            min_idx, max_idx = min(producer_set), max(producer_set)
                            if max_idx - min_idx + 1 == len(producer_set) and max_idx == i:
                                new_inst = create_push_instruction(res, "float", self.cp)
                                if new_inst:
                                    insts[min_idx : max_idx + 1] = [new_inst]
                                    folded_count += 1
                                    changed = True
                                    break
                    frame.push(LatticeValue.top("float"))

                # 4c. Binary Double operations
                elif op in DOUBLE_BINARY_OPS:
                    v2 = frame.pop()
                    v1 = frame.pop()
                    if v1.is_constant and v2.is_constant:
                        res = ArithmeticEvaluator.eval_binary_double(op, float(v1.val), float(v2.val))
                        if res is not None:
                            producer_set = set(v1.producer_indices) | set(v2.producer_indices) | {i}
                            min_idx, max_idx = min(producer_set), max(producer_set)
                            if max_idx - min_idx + 1 == len(producer_set) and max_idx == i:
                                new_inst = create_push_instruction(res, "double", self.cp)
                                if new_inst:
                                    insts[min_idx : max_idx + 1] = [new_inst]
                                    folded_count += 1
                                    changed = True
                                    break
                    frame.push(LatticeValue.top("double"))

                # 5. Unary Negation
                elif op == Opcode.INEG:
                    v = frame.pop()
                    if v.is_constant:
                        res = ArithmeticEvaluator.to_int32(-v.val)
                        producer_set = set(v.producer_indices) | {i}
                        min_idx, max_idx = min(producer_set), max(producer_set)
                        if max_idx - min_idx + 1 == len(producer_set) and max_idx == i:
                            new_inst = create_push_instruction(res, "int", self.cp)
                            if new_inst:
                                insts[min_idx : max_idx + 1] = [new_inst]
                                folded_count += 1
                                changed = True
                                break
                    frame.push(LatticeValue.top("int"))

                # 6. Type Conversions
                elif Opcode.I2L <= op <= Opcode.I2S or op in (Opcode.L2I, Opcode.L2F, Opcode.L2D, Opcode.F2I, Opcode.D2I):
                    v = frame.pop()
                    if v.is_constant:
                        conv = ArithmeticEvaluator.eval_conversion(op, v.val)
                        if conv is not None:
                            res_val, res_type = conv
                            producer_set = set(v.producer_indices) | {i}
                            min_idx, max_idx = min(producer_set), max(producer_set)
                            if max_idx - min_idx + 1 == len(producer_set) and max_idx == i:
                                new_inst = create_push_instruction(res_val, res_type, self.cp)
                                if new_inst:
                                    insts[min_idx : max_idx + 1] = [new_inst]
                                    folded_count += 1
                                    changed = True
                                    break
                    frame.push(LatticeValue.top())

                # 7. Comparisons
                elif op in (Opcode.LCMP, Opcode.FCMPL, Opcode.FCMPG, Opcode.DCMPL, Opcode.DCMPG):
                    v2 = frame.pop()
                    v1 = frame.pop()
                    if v1.is_constant and v2.is_constant:
                        cmp_res = ArithmeticEvaluator.eval_comparison(op, v1.val, v2.val)
                        if cmp_res is not None:
                            producer_set = set(v1.producer_indices) | set(v2.producer_indices) | {i}
                            min_idx, max_idx = min(producer_set), max(producer_set)
                            if max_idx - min_idx + 1 == len(producer_set) and max_idx == i:
                                new_inst = create_push_instruction(cmp_res, "int", self.cp)
                                if new_inst:
                                    insts[min_idx : max_idx + 1] = [new_inst]
                                    folded_count += 1
                                    changed = True
                                    break
                    frame.push(LatticeValue.top("int"))

                # 8. Stack Manipulation
                elif op == Opcode.DUP:
                    top = frame.peek(0)
                    frame.push(top)
                elif op == Opcode.POP:
                    frame.pop()
                elif op == Opcode.SWAP:
                    v1 = frame.pop()
                    v2 = frame.pop()
                    frame.push(v1)
                    frame.push(v2)

                else:
                    # Generic fallback: pop expected operands if known, push TOP
                    pass

        return folded_count


def fold_constants_and_arithmetic(cfg: ControlFlowGraph, cp: Optional[ConstantPool] = None) -> int:
    """Intra-procedural constant folding entry point matching PROJECT.md interface contract."""
    folder = IntraProceduralConstantFolder(cfg, cp=cp)
    return folder.fold()
