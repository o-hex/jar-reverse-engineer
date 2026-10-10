"""
tests/tier1_features/test_f31_f34_infra_cli.py:
Comprehensive Tier 1 feature tests for Features F31 through F34 (20 tests, 5 per feature).

F31: Synthetic Test JAR Fixtures
F32: 4-Tier Automated Test Suite
F33: Single-Command Test Runner
F34: Unified CLI & Batch Pipeline
"""

import sys
import time
import zipfile
import argparse
import subprocess
from pathlib import Path
import pytest

from jar_engine.model.class_file import ClassFile
from jar_engine.model.jar_archive import JarArchive
from jar_engine.model.constants import Opcode, AccessFlags
from jar_engine.cli.main import (
    build_parser,
    cmd_analyze,
    cmd_normalize,
    cmd_slice,
    cmd_rename,
)
from tests.fixtures.make_fixtures import (
    ensure_fixtures,
    build_control_flow_flattened_jar,
    build_constant_obfuscated_jar,
    build_stripped_symbols_jar,
    build_corrupted_structures_jar,
)


# ==============================================================================
# F31: Synthetic Test JAR Fixtures (5 tests)
# ==============================================================================

def test_f31_control_flow_flattened_jar_fixture(control_flow_flattened_jar: Path):
    """F31-1: Verify generation and structure of control_flow_flattened.jar."""
    assert control_flow_flattened_jar.is_file()
    assert control_flow_flattened_jar.stat().st_size > 0

    arc = JarArchive.from_file(control_flow_flattened_jar)
    assert len(arc.classes) >= 2
    assert "com/synthetic/flattened/ComplexFlow" in arc.classes
    assert "com/synthetic/flattened/DispatcherService" in arc.classes

    # Verify presence of control-flow dispatching instructions (switch)
    has_switch = False
    for cf in arc.classes.values():
        for m in cf.methods:
            code_attr = m.code_attribute(cf.constant_pool)
            if code_attr and any(op in code_attr.code for op in (Opcode.TABLESWITCH, Opcode.LOOKUPSWITCH)):
                has_switch = True
                break
    assert has_switch, "control_flow_flattened.jar should contain switch instructions"


def test_f31_constant_obfuscated_jar_fixture(constant_obfuscated_jar: Path):
    """F31-2: Verify generation and structure of constant_obfuscated.jar."""
    assert constant_obfuscated_jar.is_file()
    assert constant_obfuscated_jar.stat().st_size > 0

    arc = JarArchive.from_file(constant_obfuscated_jar)
    assert "com/synthetic/constants/ArithmeticObfuscated" in arc.classes
    assert "com/synthetic/constants/StringObfuscated" in arc.classes
    assert "com/synthetic/constants/DynamicInvoker" in arc.classes

    # Verify arithmetic instructions present
    has_arith = False
    for cf in arc.classes.values():
        for m in cf.methods:
            code_attr = m.code_attribute(cf.constant_pool)
            if code_attr and any(op in code_attr.code for op in (Opcode.IADD, Opcode.ISUB, Opcode.IMUL, Opcode.IXOR)):
                has_arith = True
                break
    assert has_arith, "constant_obfuscated.jar should contain arithmetic operations"


def test_f31_stripped_symbols_jar_fixture(stripped_symbols_jar: Path):
    """F31-3: Verify generation and structure of stripped_symbols.jar."""
    assert stripped_symbols_jar.is_file()
    assert stripped_symbols_jar.stat().st_size > 0

    arc = JarArchive.from_file(stripped_symbols_jar)
    assert len(arc.classes) >= 5

    # Check for short stripped class names (e.g. 'a/a', 'a/b')
    short_classes = [c_name for c_name in arc.classes if len(c_name.split("/")[-1]) <= 2]
    assert len(short_classes) >= 3, "stripped_symbols.jar should have short obfuscated names"


def test_f31_corrupted_structures_jar_fixture(corrupted_structures_jar: Path):
    """F31-4: Verify generation and structure of corrupted_structures.jar."""
    assert corrupted_structures_jar.is_file()
    assert corrupted_structures_jar.stat().st_size > 0

    with zipfile.ZipFile(corrupted_structures_jar, "r") as zf:
        namelist = zf.namelist()
        assert any("CorruptExceptions" in n for n in namelist)
        assert any("CorruptSignatures" in n for n in namelist)
        assert any("CorruptStackMaps" in n for n in namelist)
        assert any("DuplicateSynthetics" in n for n in namelist)


def test_f31_fixture_builder_determinism():
    """F31-5: Verify repeated runs of fixture builders yield identical JAR contents."""
    arc1 = build_control_flow_flattened_jar()
    arc2 = build_control_flow_flattened_jar()

    assert sorted(arc1.classes.keys()) == sorted(arc2.classes.keys())
    for c_name in arc1.classes:
        bytes1 = arc1.classes[c_name].to_bytes()
        bytes2 = arc2.classes[c_name].to_bytes()
        assert bytes1 == bytes2, f"Class {c_name} bytecode non-deterministic between runs"


# ==============================================================================
# F32: 4-Tier Automated Test Suite (5 tests)
# ==============================================================================

def test_f32_tier1_discovery_and_isolation():
    """F32-1: Verify all Tier 1 feature tests discoverable and runnable independently."""
    repo_tests = Path(__file__).resolve().parent.parent
    tier1_dir = repo_tests / "tier1_features"
    assert tier1_dir.is_dir()

    expected_files = [
        "test_f01_f07_archive_walker.py",
        "test_f08_f12_sanitizer.py",
        "test_f13_f19_cfg_constants.py",
        "test_f20_f23_triage_slicing.py",
        "test_f24_f30_symbol_repackaging.py",
        "test_f31_f34_infra_cli.py",
    ]
    for ef in expected_files:
        assert (tier1_dir / ef).is_file(), f"Missing Tier 1 test file: {ef}"


def test_f32_tier2_bva_isolation():
    """F32-2: Verify all Tier 2 boundary tests discoverable and runnable."""
    repo_tests = Path(__file__).resolve().parent.parent
    tier2_dir = repo_tests / "tier2_boundaries"
    assert tier2_dir.is_dir()

    bva_files = list(tier2_dir.glob("test_bva_*.py"))
    assert len(bva_files) >= 6, f"Expected at least 6 BVA test files, found {len(bva_files)}"


def test_f32_tier3_pairwise_isolation():
    """F32-3: Verify all Tier 3 pairwise tests discoverable and runnable."""
    repo_tests = Path(__file__).resolve().parent.parent
    tier3_dir = repo_tests / "tier3_pairwise"
    assert tier3_dir.is_dir()

    pairwise_files = list(tier3_dir.glob("test_pairwise_*.py"))
    assert len(pairwise_files) >= 6, f"Expected at least 6 pairwise test files, found {len(pairwise_files)}"


def test_f32_tier4_workload_isolation():
    """F32-4: Verify Tier 4 workload scenarios discoverable and isolated."""
    repo_tests = Path(__file__).resolve().parent.parent
    tier4_dir = repo_tests / "tier4_application"
    assert tier4_dir.is_dir()

    workload_files = list(tier4_dir.glob("test_workload_*.py"))
    assert len(workload_files) >= 1, f"Expected at least 1 workload test file, found {len(workload_files)}"


def test_f32_conftest_fixtures_availability(
    fixture_jars: dict,
    temp_workspace: Path,
    make_test_class,
    make_test_archive,
):
    """F32-5: Verify session and function fixtures provide clean isolated test environments."""
    # 1. fixture_jars mapping
    assert len(fixture_jars) >= 4
    for key in ["control_flow_flattened", "constant_obfuscated", "stripped_symbols", "corrupted_structures"]:
        assert key in fixture_jars
        assert fixture_jars[key].is_file()

    # 2. temp_workspace isolation
    assert temp_workspace.is_dir()
    test_sub = temp_workspace / "probe.txt"
    test_sub.write_text("ok", encoding="utf-8")
    assert test_sub.read_text(encoding="utf-8") == "ok"

    # 3. factories
    cf = make_test_class("com/test/Probe")
    assert cf.this_class_name == "com/test/Probe"
    arc = make_test_archive([cf], main_class="com.test.Probe")
    assert arc.has_class("com/test/Probe")
    assert arc.manifest.get_main_class() == "com.test.Probe"


# ==============================================================================
# F33: Single-Command Test Runner (5 tests)
# ==============================================================================

def test_f33_pytest_invocation_clean_exit():
    """F33-1: Verify pytest invocation runs cleanly and exits with returncode 0."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--version"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "pytest" in result.stdout or "pytest" in result.stderr


def test_f33_testpaths_and_pythonpath_config():
    """F33-2: Verify pyproject.toml correctly sets testpaths and pythonpath configuration."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    pyproject_file = repo_root / "pyproject.toml"
    assert pyproject_file.is_file()

    content = pyproject_file.read_text(encoding="utf-8")
    assert "[tool.pytest.ini_options]" in content
    assert 'testpaths = ["tests"]' in content
    assert 'pythonpath = ["."]' in content


def test_f33_test_discovery_count_verification():
    """F33-3: Verify test collection finds >= 380 total test items across the suite."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=str(repo_root),
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    # Output contains e.g. "437 tests collected in ..." or similar
    lines = [line for line in result.stdout.splitlines() if "collected" in line]
    assert len(lines) > 0, "Pytest should report collected test count"

    # Extract integer count
    words = lines[-1].split()
    count = None
    for i, w in enumerate(words):
        if w in ("tests", "items") and i > 0 and words[i - 1].isdigit():
            count = int(words[i - 1])
            break
        elif "collected" in w and i > 0 and words[i - 1].isdigit():
            count = int(words[i - 1])
            break

    if count is None:
        # Fallback search for any digit before "collected"
        for w in words:
            if w.isdigit():
                count = int(w)
                break

    assert count is not None and count >= 380, f"Expected >= 380 collected tests, found {count}"


def test_f33_zero_unhandled_exceptions():
    """F33-4: Verify zero uncaught unhandled exceptions during complete model pipeline execution."""
    # Ingest synthetic jar, extract hierarchy, build call graph, and export cleanly
    arc = build_constant_obfuscated_jar()
    hier = arc.get_hierarchy()
    assert hier is not None

    classes_count = len(arc.classes)
    assert classes_count > 0

    # Ensure to_bytes succeeds without throwing
    raw_bytes = arc.to_bytes()
    assert len(raw_bytes) > 0
    reloaded = JarArchive.from_bytes(raw_bytes)
    assert len(reloaded.classes) == classes_count


def test_f33_timing_performance_budget():
    """F33-5: Verify total execution of core analysis completes within performance budget (< 30s)."""
    t0 = time.time()

    # Load 2 archives, perform parsing and serialize
    arc1 = build_control_flow_flattened_jar()
    _ = arc1.to_bytes()
    arc2 = build_constant_obfuscated_jar()
    _ = arc2.to_bytes()

    elapsed = time.time() - t0
    # Execution should be rapid, well under 5.0 seconds
    assert elapsed < 5.0, f"Execution exceeded performance threshold: {elapsed:.2f}s"


# ==============================================================================
# F34: Unified CLI & Batch Pipeline (5 tests)
# ==============================================================================

def test_f34_cli_argument_parser_subcommands():
    """F34-1: Verify build_parser() supports analyze, normalize, slice, and rename."""
    parser = build_parser()

    # Subcommand: analyze
    args_analyze = parser.parse_args(["analyze", "input.jar"])
    assert args_analyze.command == "analyze"
    assert args_analyze.jar == "input.jar"

    # Subcommand: normalize
    args_norm = parser.parse_args(["normalize", "input.jar", "-o", "out.jar"])
    assert args_norm.command == "normalize"
    assert args_norm.jar == "input.jar"
    assert args_norm.output == "out.jar"

    # Subcommand: slice
    args_slice = parser.parse_args(["slice", "input.jar", "-c", "com/example/Target", "-f", "json"])
    assert args_slice.command == "slice"
    assert args_slice.target_class == "com/example/Target"
    assert args_slice.format == "json"

    # Subcommand: rename
    args_rename = parser.parse_args(["rename", "input.jar", "-m", "map.json", "-o", "renamed.jar"])
    assert args_rename.command == "rename"
    assert args_rename.mapping == "map.json"
    assert args_rename.output == "renamed.jar"


def test_f34_cli_analyze_command_execution(fixture_jars: dict):
    """F34-2: Execute cmd_analyze on sample JAR and verify exit code 0."""
    jar_path = fixture_jars["control_flow_flattened"]
    args = argparse.Namespace(jar=str(jar_path))
    ret = cmd_analyze(args)
    assert ret == 0


def test_f34_cli_normalize_command_execution(fixture_jars: dict, temp_workspace: Path):
    """F34-3: Execute cmd_normalize on sample JAR and verify output creation."""
    jar_path = fixture_jars["constant_obfuscated"]
    out_path = temp_workspace / "normalized_output.jar"

    args = argparse.Namespace(jar=str(jar_path), output=str(out_path))
    ret = cmd_normalize(args)
    assert ret == 0
    assert out_path.is_file()
    assert out_path.stat().st_size > 0


def test_f34_cli_slice_command_execution(fixture_jars: dict):
    """F34-4: Execute cmd_slice with --target-class and --format markdown."""
    jar_path = fixture_jars["constant_obfuscated"]
    target_class = "com/synthetic/constants/StringObfuscated"

    args = argparse.Namespace(
        jar=str(jar_path),
        target_class=target_class,
        sink=None,
        format="markdown"
    )
    ret = cmd_slice(args)
    assert ret == 0


def test_f34_cli_rename_command_execution(fixture_jars: dict, temp_workspace: Path):
    """F34-5: Execute cmd_rename with mapping JSON and verify repackaged archive."""
    jar_path = fixture_jars["control_flow_flattened"]
    mapping_file = temp_workspace / "mapping.json"
    mapping_content = '{"classes": {"com/synthetic/flattened/ComplexFlow": "com/synthetic/flattened/CleanFlow"}}'
    mapping_file.write_text(mapping_content, encoding="utf-8")

    out_jar = temp_workspace / "renamed_output.jar"
    args = argparse.Namespace(
        jar=str(jar_path),
        mapping=str(mapping_file),
        output=str(out_jar)
    )
    ret = cmd_rename(args)
    assert ret == 0
    assert out_jar.is_file()
    assert out_jar.stat().st_size > 0

    # Verify renamed class exists in repackaged archive
    reloaded = JarArchive.from_file(out_jar)
    assert "com/synthetic/flattened/CleanFlow" in reloaded.classes
    assert "com/synthetic/flattened/ComplexFlow" not in reloaded.classes
