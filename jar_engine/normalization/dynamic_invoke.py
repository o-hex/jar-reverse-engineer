"""
jar_engine.normalization.dynamic_invoke: Dynamic Invoke and Reflection Resolution (Feature F18).
Statically resolves invokedynamic bootstraps (StringConcatFactory, LambdaMetafactory)
and constant reflection idioms.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, Any, Union, TYPE_CHECKING
import struct

from jar_engine.model.constants import Opcode, ConstantTag, ReferenceKind
from jar_engine.model.member_info import (
    MethodInfo, CodeAttribute, BootstrapMethodsAttribute, BootstrapMethod
)
from jar_engine.model.cp_info import (
    ConstantPool, CONSTANT_InvokeDynamic_info, CONSTANT_MethodHandle_info,
    CONSTANT_MethodType_info, CONSTANT_String_info, CONSTANT_Integer_info,
    CONSTANT_Long_info, CONSTANT_Float_info, CONSTANT_Double_info,
    CONSTANT_Class_info
)
from jar_engine.walker.call_graph import InstructionScanner

if TYPE_CHECKING:
    from jar_engine.model.class_file import ClassFile
    from jar_engine.cfg.graph import ControlFlowGraph


@dataclass
class ResolvedStringConcat:
    recipe: str
    constant_args: List[Any]
    dynamic_param_types: List[str]
    is_fully_constant: bool
    folded_constant: Optional[str] = None


@dataclass
class ResolvedLambda:
    sam_interface: str            # Target functional interface (e.g. "java/util/function/Function")
    sam_method_name: str          # SAM method name (e.g. "apply")
    sam_descriptor: str           # SAM descriptor (e.g. "(Ljava/lang/Object;)Ljava/lang/Object;")
    instantiated_descriptor: str  # Specialized descriptor
    impl_class: str               # Target class containing body
    impl_method_name: str         # Target method name
    impl_descriptor: str          # Target method descriptor
    impl_kind: ReferenceKind      # ReferenceKind enum
    captured_types: List[str]     # Types captured by closure


@dataclass
class ResolvedReflectionCall:
    target_class: str
    target_method: str
    target_descriptor: Optional[str] = None
    is_static: bool = False
    pc_invoke: int = 0
    confidence: float = 1.0


class DynamicInvokeResolver:
    """Statically resolves invokedynamic and reflection call sites."""

    def __init__(self, cf: "ClassFile"):
        self.cf = cf
        self.cp = cf.constant_pool
        self.bsm_attr = self._find_bsm_attribute()

    def _find_bsm_attribute(self) -> Optional[BootstrapMethodsAttribute]:
        for attr in self.cf.attributes:
            if isinstance(attr, BootstrapMethodsAttribute):
                return attr
            try:
                if attr.name(self.cp) == "BootstrapMethods":
                    return attr
            except Exception:
                continue
        return None

    def resolve_string_concat(self, indy_entry: CONSTANT_InvokeDynamic_info) -> Optional[ResolvedStringConcat]:
        """Resolves StringConcatFactory bootstrap methods."""
        if not self.bsm_attr or indy_entry.bootstrap_method_attr_index >= len(self.bsm_attr.bootstrap_methods):
            return None

        bm = self.bsm_attr.bootstrap_methods[indy_entry.bootstrap_method_attr_index]
        try:
            bsm_handle = self.cp[bm.bootstrap_method_ref]
            if not isinstance(bsm_handle, CONSTANT_MethodHandle_info):
                return None
            bsm_cls, bsm_name, bsm_desc = self.cp.resolve_member_ref(bsm_handle.reference_index)
        except Exception:
            return None

        if "StringConcatFactory" not in bsm_cls:
            return None

        nat_name, nat_desc = self.cp.resolve_name_and_type(indy_entry.name_and_type_index)

        # Parse parameter types from call site descriptor
        param_desc = nat_desc[1 : nat_desc.find(")")]
        # Count dynamic args
        dynamic_params: List[str] = []
        p_idx = 0
        while p_idx < len(param_desc):
            c = param_desc[p_idx]
            if c in "BCDFIJSZ":
                dynamic_params.append(c)
                p_idx += 1
            elif c == "L":
                semi = param_desc.find(";", p_idx)
                if semi == -1:
                    break
                dynamic_params.append(param_desc[p_idx : semi + 1])
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
                dynamic_params.append(param_desc[start_p:p_idx])

        if bsm_name == "makeConcat":
            recipe = "\u0001" * len(dynamic_params)
            is_const = (len(dynamic_params) == 0)
            return ResolvedStringConcat(
                recipe=recipe,
                constant_args=[],
                dynamic_param_types=dynamic_params,
                is_fully_constant=is_const,
                folded_constant="" if is_const else None
            )

        elif bsm_name == "makeConcatWithConstants":
            if not bm.bootstrap_arguments:
                return None
            try:
                recipe = self.cp.resolve_string(bm.bootstrap_arguments[0])
            except Exception:
                recipe = ""

            const_args: List[Any] = []
            for arg_idx in bm.bootstrap_arguments[1:]:
                entry = self.cp.get(arg_idx)
                if isinstance(entry, CONSTANT_String_info):
                    const_args.append(self.cp.resolve_string(arg_idx))
                elif isinstance(entry, (CONSTANT_Integer_info, CONSTANT_Long_info, CONSTANT_Float_info, CONSTANT_Double_info)):
                    const_args.append(entry.value)
                elif isinstance(entry, CONSTANT_Class_info):
                    const_args.append(self.cp.resolve_class_name(arg_idx))
                else:
                    const_args.append("")

            # Evaluate if fully constant (no \u0001 dynamic arguments AND no dynamic callsite parameters)
            if "\u0001" not in recipe and len(dynamic_params) == 0:
                out_parts: List[str] = []
                c_idx = 0
                for ch in recipe:
                    if ch == "\u0002":
                        if c_idx < len(const_args):
                            out_parts.append(str(const_args[c_idx]))
                            c_idx += 1
                    else:
                        out_parts.append(ch)
                return ResolvedStringConcat(
                    recipe=recipe,
                    constant_args=const_args,
                    dynamic_param_types=dynamic_params,
                    is_fully_constant=True,
                    folded_constant="".join(out_parts)
                )
            else:
                return ResolvedStringConcat(
                    recipe=recipe,
                    constant_args=const_args,
                    dynamic_param_types=dynamic_params,
                    is_fully_constant=False,
                    folded_constant=None
                )

        return None

    def resolve_lambda(self, indy_entry: CONSTANT_InvokeDynamic_info) -> Optional[ResolvedLambda]:
        """Resolves LambdaMetafactory bootstrap methods."""
        if not self.bsm_attr or indy_entry.bootstrap_method_attr_index >= len(self.bsm_attr.bootstrap_methods):
            return None

        bm = self.bsm_attr.bootstrap_methods[indy_entry.bootstrap_method_attr_index]
        try:
            bsm_handle = self.cp[bm.bootstrap_method_ref]
            if not isinstance(bsm_handle, CONSTANT_MethodHandle_info):
                return None
            bsm_cls, bsm_name, _ = self.cp.resolve_member_ref(bsm_handle.reference_index)
        except Exception:
            return None

        if "LambdaMetafactory" not in bsm_cls:
            return None

        if len(bm.bootstrap_arguments) < 3:
            return None

        try:
            # 1. SAM interface & method from call site
            call_name, call_desc = self.cp.resolve_name_and_type(indy_entry.name_and_type_index)
            ret_desc = call_desc[call_desc.find(")") + 1 :]
            sam_interface = ret_desc[1:-1] if ret_desc.startswith("L") and ret_desc.endswith(";") else ret_desc

            # 2. samMethodType
            sam_mt = self.cp[bm.bootstrap_arguments[0]]
            sam_desc = self.cp.resolve_utf8(sam_mt.descriptor_index) if isinstance(sam_mt, CONSTANT_MethodType_info) else ""

            # 3. implMethod
            impl_mh = self.cp[bm.bootstrap_arguments[1]]
            if not isinstance(impl_mh, CONSTANT_MethodHandle_info):
                return None
            impl_cls, impl_name, impl_desc = self.cp.resolve_member_ref(impl_mh.reference_index)
            impl_kind = ReferenceKind(impl_mh.reference_kind)

            # 4. instantiatedMethodType
            inst_mt = self.cp[bm.bootstrap_arguments[2]]
            inst_desc = self.cp.resolve_utf8(inst_mt.descriptor_index) if isinstance(inst_mt, CONSTANT_MethodType_info) else ""

            return ResolvedLambda(
                sam_interface=sam_interface,
                sam_method_name=call_name,
                sam_descriptor=sam_desc,
                instantiated_descriptor=inst_desc,
                impl_class=impl_cls,
                impl_method_name=impl_name,
                impl_descriptor=impl_desc,
                impl_kind=impl_kind,
                captured_types=[]
            )
        except Exception:
            return None

    def resolve_reflection_calls(self, method: MethodInfo) -> List[ResolvedReflectionCall]:
        """Detects and resolves Class.forName(...).getMethod(...).invoke(...) calls."""
        code_attr = method.code_attribute(self.cp)
        if not code_attr or not code_attr.code:
            return []

        resolved: List[ResolvedReflectionCall] = []
        instructions = list(InstructionScanner.scan(code_attr.code))

        # Look for pattern:
        # ldc <class_name>; invokestatic Class.forName; ldc <method_name>; ... invokevirtual Method.invoke
        current_class: Optional[str] = None
        current_method: Optional[str] = None
        last_string: Optional[str] = None

        for pc, op, operands in instructions:
            if op in (Opcode.LDC, Opcode.LDC_W):
                idx = operands[0] if op == Opcode.LDC else struct.unpack(">H", operands[:2])[0]
                entry = self.cp.get(idx)
                if isinstance(entry, CONSTANT_String_info):
                    last_string = self.cp.resolve_string(idx)
                else:
                    last_string = None

            elif op in (Opcode.POP, Opcode.POP2):
                last_string = None

            elif op in (Opcode.INVOKESTATIC, Opcode.INVOKEVIRTUAL):
                if len(operands) >= 2:
                    ref_idx = struct.unpack(">H", operands[:2])[0]
                    try:
                        cls_name, m_name, _ = self.cp.resolve_member_ref(ref_idx)
                        if cls_name == "java/lang/Class" and m_name == "forName":
                            if last_string is not None:
                                current_class = last_string.replace(".", "/")
                                last_string = None
                        elif cls_name == "java/lang/Class" and m_name in ("getMethod", "getDeclaredMethod"):
                            if last_string is not None:
                                current_method = last_string
                                last_string = None
                        elif cls_name == "java/lang/reflect/Method" and m_name == "invoke":
                            m_target = current_method or last_string
                            if current_class and m_target:
                                resolved.append(ResolvedReflectionCall(
                                    target_class=current_class,
                                    target_method=m_target,
                                    pc_invoke=pc
                                ))
                                current_class = None
                                current_method = None
                                last_string = None
                    except Exception:
                        pass

        return resolved

    def resolve_and_fold(self, method: MethodInfo, cfg: Optional["ControlFlowGraph"] = None) -> int:
        """
        Scans method for resolvable dynamic invocations and reflection calls.
        Inlines completely folded string concatenations.
        """
        code_attr = method.code_attribute(self.cp)
        if not code_attr or not code_attr.code:
            return 0

        instructions = list(InstructionScanner.scan(code_attr.code))
        resolved_count = 0
        new_code = bytearray(code_attr.code)

        for pc, op, operands in instructions:
            if op == Opcode.INVOKEDYNAMIC and len(operands) >= 4:
                indy_idx = struct.unpack(">H", operands[:2])[0]
                try:
                    indy_entry = self.cp[indy_idx]
                    if isinstance(indy_entry, CONSTANT_InvokeDynamic_info):
                        concat = self.resolve_string_concat(indy_entry)
                        if concat and concat.is_fully_constant and concat.folded_constant is not None:
                            str_idx = self.cp.add_string(concat.folded_constant)
                            if str_idx <= 255:
                                replacement = bytes([Opcode.LDC, str_idx]) + (b"\x00" * 3)
                            else:
                                replacement = bytes([Opcode.LDC_W]) + struct.pack(">H", str_idx) + (b"\x00" * 2)

                            new_code[pc : pc + 5] = replacement
                            resolved_count += 1
                except Exception:
                    continue

        if resolved_count > 0:
            code_attr.code = bytes(new_code)

        # Count lambdas & reflection calls
        for _, op, operands in instructions:
            if op == Opcode.INVOKEDYNAMIC and len(operands) >= 4:
                indy_idx = struct.unpack(">H", operands[:2])[0]
                try:
                    indy_entry = self.cp[indy_idx]
                    if isinstance(indy_entry, CONSTANT_InvokeDynamic_info):
                        if self.resolve_lambda(indy_entry) is not None:
                            resolved_count += 1
                except Exception:
                    pass

        resolved_count += len(self.resolve_reflection_calls(method))
        return resolved_count


def resolve_dynamic_invokes_and_reflection(
    cf: "ClassFile",
    method: MethodInfo,
    cfg: Optional["ControlFlowGraph"] = None
) -> int:
    """Entry point matching MethodNormalizer pipeline."""
    resolver = DynamicInvokeResolver(cf)
    return resolver.resolve_and_fold(method, cfg=cfg)
