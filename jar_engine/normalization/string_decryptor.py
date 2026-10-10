"""
jar_engine.normalization.string_decryptor: Pattern detection and sandboxed symbolic emulation
of static string decryption methods with literal inlining (Feature F17).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, Any, Union, TYPE_CHECKING
import struct
import io

from jar_engine.model.constants import Opcode, AccessFlags
from jar_engine.model.member_info import MethodInfo, CodeAttribute
from jar_engine.model.cp_info import (
    ConstantPool, CONSTANT_String_info, CONSTANT_Utf8_info,
    CONSTANT_Integer_info, CONSTANT_Long_info
)
from jar_engine.walker.call_graph import InstructionScanner

if TYPE_CHECKING:
    from jar_engine.model.class_file import ClassFile
    from jar_engine.model.jar_archive import JarArchive


@dataclass
class DecryptorCandidate:
    class_name: str
    method: MethodInfo
    param_count: int


class DecryptorMethodDetector:
    """Scans ClassFile or JarArchive to identify candidate static string decryptors."""

    @classmethod
    def is_candidate(cls, cf: "ClassFile", m: MethodInfo) -> bool:
        cp = cf.constant_pool
        # 1. Must be static, non-abstract, non-native
        if not (m.access_flags & AccessFlags.STATIC) or m.is_abstract or m.is_native:
            return False

        # 2. Return type must be String
        desc = m.descriptor(cp)
        if not desc.endswith(")Ljava/lang/String;"):
            return False

        code_attr = m.code_attribute(cp)
        if code_attr is None or len(code_attr.code) == 0 or len(code_attr.code) > 500:
            return False

        # 3. Purity & Opcode signature check
        has_transform = False
        has_array_ops = False
        for pc, op, operands in InstructionScanner.scan(code_attr.code):
            # Prohibited side effects
            if op in (Opcode.PUTSTATIC, Opcode.PUTFIELD, Opcode.ATHROW, Opcode.MONITORENTER):
                return False
            if op in (Opcode.INVOKESTATIC, Opcode.INVOKEVIRTUAL, Opcode.INVOKESPECIAL):
                if len(operands) >= 2:
                    ref_idx = struct.unpack(">H", operands[:2])[0]
                    try:
                        target_cls, target_name, _ = cp.resolve_member_ref(ref_idx)
                        if not (target_cls.startswith("java/lang/String") or
                                target_cls.startswith("java/lang/StringBuilder") or
                                target_cls.startswith("java/lang/StringBuffer") or
                                target_cls.startswith("java/lang/Integer") or
                                target_cls.startswith("java/lang/Character") or
                                target_cls.startswith("java/lang/Math") or
                                target_cls == cf.this_class_name):
                            return False
                    except Exception:
                        return False

            if op in (Opcode.IXOR, Opcode.ISUB, Opcode.IADD, Opcode.IOR, Opcode.IAND):
                has_transform = True
            if op in (Opcode.CALOAD, Opcode.CASTORE, Opcode.BALOAD, Opcode.BASTORE,
                      Opcode.IALOAD, Opcode.IASTORE, Opcode.NEWARRAY, Opcode.ARRAYLENGTH):
                has_array_ops = True

        return has_transform or has_array_ops


class BytecodeEmulator:
    """Lightweight pure-Python sandbox for executing pure string decryption routines."""

    def __init__(self, max_steps: int = 50000):
        self.max_steps = max_steps

    def execute(self, cf: "ClassFile", method: MethodInfo, args: List[Any]) -> Optional[str]:
        """
        Executes decryptor method with constant arguments.
        Returns decrypted string on success, or None on failure/timeout.
        """
        cp = cf.constant_pool
        code_attr = method.code_attribute(cp)
        if not code_attr:
            return None

        code = code_attr.code
        # Pre-decode instructions: pc -> (op, operands, next_pc)
        inst_map: Dict[int, Tuple[int, bytes, int]] = {}
        for pc, op, operands in InstructionScanner.scan(code):
            next_pc = pc + 1 + len(operands)
            inst_map[pc] = (op, operands, next_pc)

        locals_map: Dict[int, Any] = {}
        for i, arg in enumerate(args):
            locals_map[i] = arg

        stack: List[Any] = []
        pc = 0
        steps = 0

        while pc in inst_map and steps < self.max_steps:
            steps += 1
            op, operands, next_pc = inst_map[pc]

            try:
                # 1. Constants
                if Opcode.ICONST_M1 <= op <= Opcode.ICONST_5:
                    stack.append(op - Opcode.ICONST_0)
                    pc = next_pc
                elif op == Opcode.ACONST_NULL:
                    stack.append(None)
                    pc = next_pc
                elif op == Opcode.BIPUSH:
                    stack.append(struct.unpack(">b", operands[:1])[0])
                    pc = next_pc
                elif op == Opcode.SIPUSH:
                    stack.append(struct.unpack(">h", operands[:2])[0])
                    pc = next_pc
                elif op in (Opcode.LDC, Opcode.LDC_W):
                    idx = operands[0] if op == Opcode.LDC else struct.unpack(">H", operands[:2])[0]
                    entry = cp.get(idx)
                    if isinstance(entry, CONSTANT_String_info):
                        stack.append(cp.resolve_string(idx))
                    elif isinstance(entry, CONSTANT_Integer_info):
                        stack.append(entry.value)
                    elif isinstance(entry, CONSTANT_Long_info):
                        stack.append(entry.value)
                    else:
                        stack.append(None)
                    pc = next_pc

                # 2. Local variables
                elif Opcode.ILOAD_0 <= op <= Opcode.ILOAD_3 or op == Opcode.ILOAD:
                    slot = op - Opcode.ILOAD_0 if op != Opcode.ILOAD else operands[0]
                    stack.append(locals_map.get(slot, 0))
                    pc = next_pc
                elif Opcode.ALOAD_0 <= op <= Opcode.ALOAD_3 or op == Opcode.ALOAD:
                    slot = op - Opcode.ALOAD_0 if op != Opcode.ALOAD else operands[0]
                    stack.append(locals_map.get(slot, None))
                    pc = next_pc
                elif Opcode.ISTORE_0 <= op <= Opcode.ISTORE_3 or op == Opcode.ISTORE:
                    slot = op - Opcode.ISTORE_0 if op != Opcode.ISTORE else operands[0]
                    locals_map[slot] = stack.pop()
                    pc = next_pc
                elif Opcode.ASTORE_0 <= op <= Opcode.ASTORE_3 or op == Opcode.ASTORE:
                    slot = op - Opcode.ASTORE_0 if op != Opcode.ASTORE else operands[0]
                    locals_map[slot] = stack.pop()
                    pc = next_pc
                elif op == Opcode.IINC:
                    slot, incr = operands[0], struct.unpack(">b", operands[1:2])[0]
                    locals_map[slot] = (locals_map.get(slot, 0) + incr) & 0xFFFFFFFF
                    if locals_map[slot] >= 0x80000000:
                        locals_map[slot] -= 0x100000000
                    pc = next_pc

                # 3. Arithmetic & Bitwise
                elif op == Opcode.IADD:
                    b, a = stack.pop(), stack.pop()
                    val = (a + b) & 0xFFFFFFFF
                    stack.append(val if val < 0x80000000 else val - 0x100000000)
                    pc = next_pc
                elif op == Opcode.ISUB:
                    b, a = stack.pop(), stack.pop()
                    val = (a - b) & 0xFFFFFFFF
                    stack.append(val if val < 0x80000000 else val - 0x100000000)
                    pc = next_pc
                elif op == Opcode.IMUL:
                    b, a = stack.pop(), stack.pop()
                    val = (a * b) & 0xFFFFFFFF
                    stack.append(val if val < 0x80000000 else val - 0x100000000)
                    pc = next_pc
                elif op == Opcode.IDIV:
                    b, a = stack.pop(), stack.pop()
                    if b == 0:
                        return None
                    val = int(a / b) & 0xFFFFFFFF
                    stack.append(val if val < 0x80000000 else val - 0x100000000)
                    pc = next_pc
                elif op == Opcode.IREM:
                    b, a = stack.pop(), stack.pop()
                    if b == 0:
                        return None
                    sign = -1 if a < 0 else 1
                    stack.append(sign * (abs(a) % abs(b)))
                    pc = next_pc
                elif op == Opcode.IXOR:
                    b, a = stack.pop(), stack.pop()
                    val = (a ^ b) & 0xFFFFFFFF
                    stack.append(val if val < 0x80000000 else val - 0x100000000)
                    pc = next_pc
                elif op == Opcode.IAND:
                    b, a = stack.pop(), stack.pop()
                    val = (a & b) & 0xFFFFFFFF
                    stack.append(val if val < 0x80000000 else val - 0x100000000)
                    pc = next_pc
                elif op == Opcode.IOR:
                    b, a = stack.pop(), stack.pop()
                    val = (a | b) & 0xFFFFFFFF
                    stack.append(val if val < 0x80000000 else val - 0x100000000)
                    pc = next_pc
                elif op == Opcode.ISHL:
                    b, a = stack.pop(), stack.pop()
                    val = (a << (b & 0x1F)) & 0xFFFFFFFF
                    stack.append(val if val < 0x80000000 else val - 0x100000000)
                    pc = next_pc
                elif op == Opcode.ISHR:
                    b, a = stack.pop(), stack.pop()
                    val = (a >> (b & 0x1F)) & 0xFFFFFFFF
                    stack.append(val if val < 0x80000000 else val - 0x100000000)
                    pc = next_pc
                elif op == Opcode.IUSHR:
                    b, a = stack.pop(), stack.pop()
                    val = ((a & 0xFFFFFFFF) >> (b & 0x1F))
                    stack.append(val if val < 0x80000000 else val - 0x100000000)
                    pc = next_pc
                elif op == Opcode.I2B:
                    v = stack.pop() & 0xFF
                    stack.append(v if v < 0x80 else v - 0x100)
                    pc = next_pc
                elif op == Opcode.I2C:
                    stack.append(stack.pop() & 0xFFFF)
                    pc = next_pc
                elif op == Opcode.I2S:
                    v = stack.pop() & 0xFFFF
                    stack.append(v if v < 0x8000 else v - 0x10000)
                    pc = next_pc

                # 4. Arrays
                elif op == Opcode.NEWARRAY:
                    count = stack.pop()
                    atype = operands[0]
                    # 5 = T_CHAR, 8 = T_BYTE, 10 = T_INT
                    if atype == 5:
                        stack.append([0] * max(0, count))
                    elif atype == 8:
                        stack.append(bytearray(max(0, count)))
                    else:
                        stack.append([0] * max(0, count))
                    pc = next_pc
                elif op == Opcode.ARRAYLENGTH:
                    arr = stack.pop()
                    stack.append(len(arr) if arr is not None else 0)
                    pc = next_pc
                elif op in (Opcode.CALOAD, Opcode.BALOAD, Opcode.IALOAD):
                    idx, arr = stack.pop(), stack.pop()
                    stack.append(arr[idx])
                    pc = next_pc
                elif op in (Opcode.CASTORE, Opcode.BASTORE, Opcode.IASTORE):
                    val, idx, arr = stack.pop(), stack.pop(), stack.pop()
                    arr[idx] = val & (0xFFFF if op == Opcode.CASTORE else (0xFF if op == Opcode.BASTORE else 0xFFFFFFFF))
                    pc = next_pc

                # 5. Control Flow
                elif op == Opcode.GOTO:
                    offset = struct.unpack(">h", operands[:2])[0]
                    pc = pc + offset
                elif op == Opcode.GOTO_W:
                    offset = struct.unpack(">i", operands[:4])[0]
                    pc = pc + offset
                elif op == Opcode.IFEQ:
                    offset = struct.unpack(">h", operands[:2])[0]
                    pc = (pc + offset) if stack.pop() == 0 else next_pc
                elif op == Opcode.IFNE:
                    offset = struct.unpack(">h", operands[:2])[0]
                    pc = (pc + offset) if stack.pop() != 0 else next_pc
                elif op == Opcode.IFLT:
                    offset = struct.unpack(">h", operands[:2])[0]
                    pc = (pc + offset) if stack.pop() < 0 else next_pc
                elif op == Opcode.IFGE:
                    offset = struct.unpack(">h", operands[:2])[0]
                    pc = (pc + offset) if stack.pop() >= 0 else next_pc
                elif op == Opcode.IFGT:
                    offset = struct.unpack(">h", operands[:2])[0]
                    pc = (pc + offset) if stack.pop() > 0 else next_pc
                elif op == Opcode.IFLE:
                    offset = struct.unpack(">h", operands[:2])[0]
                    pc = (pc + offset) if stack.pop() <= 0 else next_pc
                elif op == Opcode.IF_ICMPEQ:
                    b, a = stack.pop(), stack.pop()
                    offset = struct.unpack(">h", operands[:2])[0]
                    pc = (pc + offset) if a == b else next_pc
                elif op == Opcode.IF_ICMPNE:
                    b, a = stack.pop(), stack.pop()
                    offset = struct.unpack(">h", operands[:2])[0]
                    pc = (pc + offset) if a != b else next_pc
                elif op == Opcode.IF_ICMPLT:
                    b, a = stack.pop(), stack.pop()
                    offset = struct.unpack(">h", operands[:2])[0]
                    pc = (pc + offset) if a < b else next_pc
                elif op == Opcode.IF_ICMPGE:
                    b, a = stack.pop(), stack.pop()
                    offset = struct.unpack(">h", operands[:2])[0]
                    pc = (pc + offset) if a >= b else next_pc
                elif op == Opcode.IF_ICMPGT:
                    b, a = stack.pop(), stack.pop()
                    offset = struct.unpack(">h", operands[:2])[0]
                    pc = (pc + offset) if a > b else next_pc
                elif op == Opcode.IF_ICMPLE:
                    b, a = stack.pop(), stack.pop()
                    offset = struct.unpack(">h", operands[:2])[0]
                    pc = (pc + offset) if a <= b else next_pc

                # 6. Method Invocations
                elif op in (Opcode.INVOKEVIRTUAL, Opcode.INVOKESPECIAL, Opcode.INVOKESTATIC):
                    ref_idx = struct.unpack(">H", operands[:2])[0]
                    target_cls, target_name, target_desc = cp.resolve_member_ref(ref_idx)

                    if target_cls == "java/lang/String" and target_name == "toCharArray":
                        s = stack.pop()
                        stack.append([ord(c) for c in str(s)])
                    elif target_cls == "java/lang/String" and target_name == "length":
                        s = stack.pop()
                        stack.append(len(str(s)))
                    elif target_cls == "java/lang/String" and target_name == "charAt":
                        idx, s = stack.pop(), stack.pop()
                        stack.append(ord(str(s)[idx]))
                    elif target_cls == "java/lang/String" and target_name == "<init>":
                        # new String(chars/bytes)
                        if "([C)" in target_desc:
                            chars = stack.pop()
                            stack.pop()  # uninitialized this reference from NEW
                            stack.append("".join(chr(c) for c in chars))
                        elif "([B)" in target_desc:
                            data = stack.pop()
                            stack.pop()
                            raw_bytes = bytes([b & 0xFF for b in data]) if isinstance(data, (list, tuple, bytearray)) else (data if isinstance(data, (bytes, bytearray)) else bytes())
                            stack.append(raw_bytes.decode("utf-8", errors="replace"))
                        elif "([BLjava/lang/String;)" in target_desc:
                            charset = stack.pop()
                            data = stack.pop()
                            stack.pop()
                            raw_bytes = bytes([b & 0xFF for b in data]) if isinstance(data, (list, tuple, bytearray)) else (data if isinstance(data, (bytes, bytearray)) else bytes())
                            stack.append(raw_bytes.decode(charset, errors="replace"))
                    elif target_cls in ("java/lang/StringBuilder", "java/lang/StringBuffer") and target_name == "<init>":
                        stack.pop()  # uninitialized this
                        stack.append(io.StringIO())
                    elif target_cls in ("java/lang/StringBuilder", "java/lang/StringBuffer") and target_name == "append":
                        arg = stack.pop()
                        sb = stack.pop()
                        if isinstance(arg, int) and target_desc == "(C)Ljava/lang/StringBuilder;":
                            sb.write(chr(arg))
                        else:
                            sb.write(str(arg))
                        stack.append(sb)
                    elif target_cls in ("java/lang/StringBuilder", "java/lang/StringBuffer") and target_name == "toString":
                        sb = stack.pop()
                        stack.append(sb.getvalue())
                    pc = next_pc

                elif op == Opcode.NEW:
                    stack.append("<uninitialized_object>")
                    pc = next_pc

                elif op == Opcode.DUP:
                    top = stack[-1] if stack else None
                    stack.append(top)
                    pc = next_pc

                elif op == Opcode.POP:
                    if stack:
                        stack.pop()
                    pc = next_pc

                # 7. Return
                elif op == Opcode.ARETURN:
                    res = stack.pop() if stack else None
                    if isinstance(res, str):
                        return res
                    return None

                else:
                    pc = next_pc

            except Exception:
                return None

        return None


class StringDecryptorResolver:
    """Reconstructs encrypted string literals across methods, classes, and JAR archives."""

    def __init__(self, archive: Optional["JarArchive"] = None):
        self.archive = archive
        self.emulator = BytecodeEmulator()

    @staticmethod
    def _parse_param_types(desc: str) -> List[str]:
        start = desc.find("(")
        end = desc.find(")")
        if start == -1 or end == -1:
            return []
        param_desc = desc[start + 1 : end]
        types: List[str] = []
        p_idx = 0
        while p_idx < len(param_desc):
            c = param_desc[p_idx]
            if c in "BCDFIJSZ":
                types.append(c)
                p_idx += 1
            elif c == "L":
                semi = param_desc.find(";", p_idx)
                if semi == -1:
                    break
                types.append(param_desc[p_idx : semi + 1])
                p_idx = semi + 1
            elif c == "[":
                start_p = p_idx
                while p_idx < len(param_desc) and param_desc[p_idx] == "[":
                    p_idx += 1
                if p_idx < len(param_desc) and param_desc[p_idx] == "L":
                    semi = param_desc.find(";", p_idx)
                    p_idx = semi + 1 if semi != -1 else p_idx
                else:
                    p_idx += 1
                types.append(param_desc[start_p:p_idx])
            else:
                p_idx += 1
        return types

    def reconstruct_encrypted_strings(self, cf: "ClassFile", method: MethodInfo) -> int:
        """
        Finds decryptor call sites in method and inlines decrypted plaintext literals.
        Returns count of decrypted strings inlined.
        """
        cp = cf.constant_pool
        code_attr = method.code_attribute(cp)
        if not code_attr or not code_attr.code:
            return 0

        # Find candidate decryptors in the class
        candidates: Dict[str, MethodInfo] = {}
        for m in cf.methods:
            if DecryptorMethodDetector.is_candidate(cf, m):
                key = f"{m.name(cp)}{m.descriptor(cp)}"
                candidates[key] = m

        if not candidates and self.archive:
            # Check other classes in archive
            for other_cf in self.archive.classes.values():
                for m in other_cf.methods:
                    if DecryptorMethodDetector.is_candidate(other_cf, m):
                        key = f"L{other_cf.this_class_name};->{m.name(other_cf.constant_pool)}{m.descriptor(other_cf.constant_pool)}"
                        candidates[key] = m

        if not candidates:
            return 0

        # Scan method instructions for call sites
        instructions = list(InstructionScanner.scan(code_attr.code))
        decrypted_count = 0
        new_code = bytearray(code_attr.code)

        # Helper to parse arguments pushed before invokestatic
        i = 0
        while i < len(instructions):
            pc, op, operands = instructions[i]

            if op == Opcode.INVOKESTATIC and len(operands) >= 2:
                ref_idx = struct.unpack(">H", operands[:2])[0]
                try:
                    target_cls, target_name, target_desc = cp.resolve_member_ref(ref_idx)
                except Exception:
                    i += 1
                    continue

                target_method: Optional[MethodInfo] = None
                target_cf = cf
                local_key = f"{target_name}{target_desc}"
                global_key = f"L{target_cls};->{target_name}{target_desc}"

                if local_key in candidates and target_cls == cf.this_class_name:
                    target_method = candidates[local_key]
                elif global_key in candidates:
                    target_method = candidates[global_key]
                    if self.archive and target_cls in self.archive.classes:
                        target_cf = self.archive.classes[target_cls]

                if target_method:
                    # Trace back constant arguments pushed prior to this instruction
                    param_types = self._parse_param_types(target_desc)
                    arg_count = len(param_types)

                    if 1 <= arg_count <= 4 and i >= arg_count:
                        arg_insts = instructions[i - arg_count : i]
                        args: List[Any] = []
                        valid_constants = True

                        for a_pc, a_op, a_ops in arg_insts:
                            if a_op in (Opcode.LDC, Opcode.LDC_W):
                                c_idx = a_ops[0] if a_op == Opcode.LDC else struct.unpack(">H", a_ops[:2])[0]
                                c_entry = cp.get(c_idx)
                                if isinstance(c_entry, CONSTANT_String_info):
                                    args.append(cp.resolve_string(c_idx))
                                elif isinstance(c_entry, CONSTANT_Integer_info):
                                    args.append(c_entry.value)
                                else:
                                    valid_constants = False
                            elif Opcode.ICONST_M1 <= a_op <= Opcode.ICONST_5:
                                args.append(a_op - Opcode.ICONST_0)
                            elif a_op == Opcode.BIPUSH:
                                args.append(struct.unpack(">b", a_ops[:1])[0])
                            elif a_op == Opcode.SIPUSH:
                                args.append(struct.unpack(">h", a_ops[:2])[0])
                            else:
                                valid_constants = False

                        if valid_constants and len(args) == arg_count:
                            decrypted = self.emulator.execute(target_cf, target_method, args)
                            if decrypted is not None:
                                # Mutate ConstantPool
                                str_idx = cp.add_string(decrypted)

                                # Replace span from arg_insts[0].pc to pc + 3 with LDC + NOPs
                                start_pc = arg_insts[0][0]
                                end_pc = pc + 3
                                span_len = end_pc - start_pc

                                if str_idx <= 255:
                                    replacement = bytes([Opcode.LDC, str_idx]) + (b"\x00" * (span_len - 2))
                                else:
                                    replacement = bytes([Opcode.LDC_W]) + struct.pack(">H", str_idx) + (b"\x00" * (span_len - 3))

                                new_code[start_pc : end_pc] = replacement
                                decrypted_count += 1

            i += 1

        if decrypted_count > 0:
            code_attr.code = bytes(new_code)

        return decrypted_count

    def resolve_archive_strings(self, archive: "JarArchive") -> int:
        self.archive = archive
        total = 0
        for cf in archive.classes.values():
            for m in cf.methods:
                total += self.reconstruct_encrypted_strings(cf, m)
        return total


def reconstruct_encrypted_strings(cf: "ClassFile", method: MethodInfo) -> int:
    """Standalone functional interface matching PROJECT.md MethodNormalizer contract."""
    resolver = StringDecryptorResolver()
    return resolver.reconstruct_encrypted_strings(cf, method)
