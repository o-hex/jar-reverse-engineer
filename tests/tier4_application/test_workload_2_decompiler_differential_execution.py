"""
tests/tier4_application/test_workload_2_decompiler_differential_execution.py: Real-world Workload 2.
Decompiler Differential Execution across CFR 0.152, Vineflower 1.12.0, and Procyon 0.6.0.
Exercises:
- M2 Bytecode Sanitizer: F8-F12
- M3 Control Flow Normalization: F14
- M5 Decompiled Source Export: F30
"""

from pathlib import Path
import pytest

from jar_engine.model.jar_archive import JarArchive
from jar_engine.sanitizer import (
    BytecodeSanitizer,
    verify_sanitized_class_structural_integrity,
)
from jar_engine.sanitizer.decompiler_check import (
    run_decompiler_check,
    find_decompiler_command,
)


def test_workload_2_decompiler_differential_execution(corrupted_structures_jar: Path, temp_workspace: Path):
    """
    Workload 2: Differential execution testing of external decompilers (CFR, Vineflower, Procyon)
    and in-memory structural verification on corrupted_structures.jar before vs after sanitization.

    Invariants:
    - Pre-sanitization: Structural integrity check detects invalid generic signatures
      and corrupted exception bounds. Procyon fails or throws GenericSignatureFormatError.
    - Sanitization: BytecodeSanitizer cleans frames, signatures, exceptions, and synthetics.
    - Post-sanitization: 100% of archive classes pass structural verification with zero issues.
      All available decompiler backends (CFR, Vineflower, Procyon) decompile successfully with success=True.
    """
    # 1. Ingest corrupted structures archive
    archive = JarArchive.from_file(corrupted_structures_jar)
    assert "com/synthetic/corrupted/CorruptSignatures" in archive.classes
    assert "com/synthetic/corrupted/CorruptExceptions" in archive.classes

    # 2. Structural verification BEFORE sanitization: must detect issues
    cf_sig = archive.classes["com/synthetic/corrupted/CorruptSignatures"]
    ok_sig_before, issues_sig_before = verify_sanitized_class_structural_integrity(cf_sig)
    assert ok_sig_before is False
    assert len(issues_sig_before) >= 1
    assert any("signature" in issue.lower() for issue in issues_sig_before)

    cf_ex = archive.classes["com/synthetic/corrupted/CorruptExceptions"]
    ok_ex_before, issues_ex_before = verify_sanitized_class_structural_integrity(cf_ex)
    assert ok_ex_before is False
    assert len(issues_ex_before) >= 1
    assert any("exception" in issue.lower() for issue in issues_ex_before)

    # 3. Differential check BEFORE sanitization on external decompilers
    res_before = run_decompiler_check(corrupted_structures_jar, timeout_seconds=15)
    # If Procyon is available, it must fail prior to sanitization due to GenericSignatureFormatError
    if res_before.get("procyon") and res_before["procyon"].available:
        assert res_before["procyon"].success is False or "GenericSignatureFormatError" in res_before["procyon"].error

    # 4. Execute whole-archive bytecode sanitization pass (M2)
    sanitizer = BytecodeSanitizer()
    report = sanitizer.sanitize_archive(archive)
    assert report.total_signatures_stripped >= 1
    assert report.total_exceptions_fixed >= 1

    # 5. Structural verification AFTER sanitization: 100% clean
    for c_name, cf in archive.classes.items():
        ok, issues = verify_sanitized_class_structural_integrity(cf)
        assert ok is True, f"Class {c_name} failed structural verification after sanitization: {issues}"
        assert len(issues) == 0

    # 6. Save sanitized archive to disk
    sanitized_jar_path = temp_workspace / "sanitized_structures.jar"
    archive.save(sanitized_jar_path)
    assert sanitized_jar_path.is_file() and sanitized_jar_path.stat().st_size > 0

    # 7. Differential check AFTER sanitization on external decompilers
    res_after = run_decompiler_check(sanitized_jar_path, timeout_seconds=15)
    for backend, result in res_after.items():
        if result.available:
            assert result.success is True, f"Decompiler backend '{backend}' failed on sanitized archive: {result.error}"

    # 8. Verify differential improvement: Procyon succeeded after sanitization
    if res_before.get("procyon") and res_before["procyon"].available:
        assert res_after["procyon"].success is True
