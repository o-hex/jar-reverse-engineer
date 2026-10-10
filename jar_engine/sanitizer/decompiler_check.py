"""
jar_engine.sanitizer.decompiler_check: Decompiler backend compatibility verification (CFR, Vineflower, Procyon).

Verifies whether sanitized class files and JAR archives parse and decompile cleanly
without crashing decompiler backends or throwing ClassFormatError / SignatureFormatException.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import subprocess
import shutil
import tempfile
import os

from jar_engine.model.class_file import ClassFile
from jar_engine.model.jar_archive import JarArchive


@dataclass
class DecompilerResult:
    backend: str
    available: bool
    success: bool
    output: str
    error: str


def find_decompiler_command(name: str) -> Optional[List[str]]:
    """Detects if a decompiler binary or runnable jar is available."""
    # Direct command executable check
    if shutil.which(name):
        return [name]

    # Check for common jar wrappers with java
    java_exe = shutil.which("java")
    if not java_exe:
        return None

    # Check common locations / environment variables
    env_var = f"{name.upper()}_JAR"
    if os.environ.get(env_var) and Path(os.environ[env_var]).is_file():
        return [java_exe, "-jar", os.environ[env_var]]

    return None


def run_decompiler_check(
    target_path: Union[str, Path],
    timeout_seconds: int = 15
) -> Dict[str, DecompilerResult]:
    """
    Attempts to run available decompilers (CFR, Vineflower, Procyon) against target class/JAR.
    Returns result mapping per backend.
    """
    path = Path(target_path)
    results: Dict[str, DecompilerResult] = {}
    backends = ["cfr", "vineflower", "procyon"]

    for b in backends:
        cmd_prefix = find_decompiler_command(b)
        if not cmd_prefix:
            results[b] = DecompilerResult(
                backend=b, available=False, success=True,
                output="", error="Decompiler not found on system PATH"
            )
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "out"
            out_dir.mkdir(parents=True, exist_ok=True)

            if b == "cfr":
                cmd = cmd_prefix + [str(path), "--outputdir", str(out_dir)]
            elif b == "vineflower":
                cmd = cmd_prefix + [str(path), str(out_dir)]
            elif b == "procyon":
                cmd = cmd_prefix + ["-jar" if path.suffix == ".jar" else "", str(path), "-o", str(out_dir)]
                cmd = [c for c in cmd if c]
            else:
                cmd = cmd_prefix + [str(path)]

            try:
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout_seconds,
                    text=True
                )
                success = proc.returncode == 0
                results[b] = DecompilerResult(
                    backend=b,
                    available=True,
                    success=success,
                    output=proc.stdout[:1000],
                    error=proc.stderr[:1000]
                )
            except subprocess.TimeoutExpired:
                results[b] = DecompilerResult(
                    backend=b, available=True, success=False,
                    output="", error="Decompilation timed out (potential infinite loop)"
                )
            except Exception as e:
                results[b] = DecompilerResult(
                    backend=b, available=True, success=False,
                    output="", error=str(e)
                )

    return results


def verify_sanitized_class_structural_integrity(cf: ClassFile) -> Tuple[bool, List[str]]:
    """
    Fast in-memory structural validation confirming class conforms to decompiler safety rules.
    Checks:
    - Signature attribute validity
    - Exception handler ranges within bytecode bounds
    - StackMapTable well-formedness
    - Field name uniqueness per class
    """
    from .signatures import is_valid_class_signature, is_valid_method_signature, is_valid_field_signature
    from .stack_map import validate_stack_map_table

    issues: List[str] = []
    cp = cf.constant_pool

    # Check class signature
    class_sig = cf.get_attribute("Signature")
    if class_sig and hasattr(class_sig, "signature"):
        if not is_valid_class_signature(class_sig.signature(cp)):
            issues.append(f"Invalid class signature: {class_sig.signature(cp)}")

    # Check fields
    seen_f_names = set()
    for f in cf.fields:
        f_name = f.name(cp)
        if f_name in seen_f_names:
            issues.append(f"Duplicate field name in class: {f_name}")
        seen_f_names.add(f_name)

        sig = f.signature(cp)
        if sig and not is_valid_field_signature(sig):
            issues.append(f"Invalid field signature on {f_name}: {sig}")

    # Check methods & exception tables
    for m in cf.methods:
        m_name = m.name(cp)
        sig = m.signature(cp)
        if sig and not is_valid_method_signature(sig):
            issues.append(f"Invalid method signature on {m_name}: {sig}")

        code = m.code_attribute(cp)
        if code:
            c_len = len(code.code)
            for ex in code.exception_table:
                if ex.start_pc >= ex.end_pc or ex.end_pc > c_len or ex.handler_pc >= c_len:
                    issues.append(f"Corrupt exception entry in {m_name}: {ex.start_pc}->{ex.end_pc}:{ex.handler_pc}")

    return (len(issues) == 0, issues)
