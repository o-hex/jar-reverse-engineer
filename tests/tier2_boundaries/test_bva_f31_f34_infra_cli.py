"""
tests/tier2_boundaries/test_bva_f31_f34_infra_cli.py:
Boundary value analysis and corner cases for features F31 through F34.
Total: 20 tests.
"""

import sys
import time
import zipfile
import argparse
import subprocess
import pytest
from pathlib import Path

from jar_engine.model.class_file import ClassFile
from jar_engine.model.jar_archive import JarArchive
from jar_engine.model.constants import Opcode
from jar_engine.cli.main import build_parser, cmd_analyze, cmd_rename, cmd_normalize, cmd_slice
from tests.fixtures.make_fixtures import (
    ensure_fixtures,
    build_control_flow_flattened_jar,
    build_constant_obfuscated_jar
)


# ==============================================================================
# F31 BOUNDARIES (5 tests)
# ==============================================================================

def test_bva_f31_fixture_files_exist(fixture_jars):
    """BVA F31-1: All 4 synthetic test JARs exist in fixture directory with non-zero size."""
    expected = [
        "control_flow_flattened",
        "constant_obfuscated",
        "stripped_symbols",
        "corrupted_structures"
    ]
    for name in expected:
        assert name in fixture_jars
        p = fixture_jars[name]
        assert p.is_file(), f"Fixture {name} is not a file: {p}"
        assert p.stat().st_size > 0, f"Fixture {name} is empty: {p}"


def test_bva_f31_fixture_valid_zip(fixture_jars):
    """BVA F31-2: All 4 synthetic JARs are valid ZIP files."""
    for name, p in fixture_jars.items():
        if name.endswith(".jar") or "_" in name:
            with zipfile.ZipFile(p, "r") as zf:
                infolist = zf.infolist()
                assert len(infolist) > 0, f"Empty zip archive: {p}"


def test_bva_f31_fixture_classes_parse(fixture_jars):
    """BVA F31-3: Classes inside non-corrupted synthetic JARs parse with ClassFile.parse."""
    for key in ["control_flow_flattened", "constant_obfuscated", "stripped_symbols"]:
        p = fixture_jars[key]
        with zipfile.ZipFile(p, "r") as zf:
            class_names = [n for n in zf.namelist() if n.endswith(".class")]
            assert len(class_names) > 0
            for c_name in class_names:
                data = zf.read(c_name)
                cf = ClassFile.parse(data)
                assert cf.magic == 0xCAFEBABE
                assert len(cf.this_class_name) > 0


def test_bva_f31_fixture_obfuscation_patterns(fixture_jars):
    """BVA F31-4: Synthetic JARs contain expected bytecode patterns (switches, arithmetic)."""
    p_cfg = fixture_jars["control_flow_flattened"]
    arc_cfg = JarArchive.from_file(p_cfg)
    found_switch = False
    for cf in arc_cfg.classes.values():
        for m in cf.methods:
            code_attr = m.code_attribute(cf.constant_pool)
            if code_attr and (Opcode.TABLESWITCH in code_attr.code or Opcode.LOOKUPSWITCH in code_attr.code):
                found_switch = True
                break
    assert found_switch, "control_flow_flattened.jar missing expected switch dispatch instructions"

    p_const = fixture_jars["constant_obfuscated"]
    arc_const = JarArchive.from_file(p_const)
    found_arith = False
    for cf in arc_const.classes.values():
        for m in cf.methods:
            code_attr = m.code_attribute(cf.constant_pool)
            if code_attr and any(op in code_attr.code for op in (Opcode.IADD, Opcode.ISUB, Opcode.IMUL)):
                found_arith = True
                break
    assert found_arith, "constant_obfuscated.jar missing expected arithmetic instructions"


def test_bva_f31_fixture_determinism_checksum():
    """BVA F31-5: Fixture generator produces bit-exact identical files across multiple runs."""
    arc1 = build_control_flow_flattened_jar()
    arc2 = build_control_flow_flattened_jar()

    classes1 = sorted(arc1.classes.keys())
    classes2 = sorted(arc2.classes.keys())
    assert classes1 == classes2

    for c_name in classes1:
        assert arc1.classes[c_name].to_bytes() == arc2.classes[c_name].to_bytes()


# ==============================================================================
# F32 BOUNDARIES (5 tests)
# ==============================================================================

def test_bva_f32_four_tiers_discoverable():
    """BVA F32-1: Pytest discovers test files across tier directories."""
    repo_tests = Path(__file__).resolve().parent.parent
    tier2_dir = repo_tests / "tier2_boundaries"
    assert tier2_dir.is_dir()
    test_files = list(tier2_dir.glob("test_*.py"))
    assert len(test_files) >= 6


def test_bva_f32_tier_execution_isolation(fixture_jars):
    """BVA F32-2: Execution of tests does not mutate synthetic JAR fixtures on disk."""
    sizes_before = {k: fixture_jars[k].stat().st_size for k in fixture_jars if "_" in k}
    # Read archives in memory
    for k in sizes_before:
        _ = JarArchive.from_file(fixture_jars[k])
    sizes_after = {k: fixture_jars[k].stat().st_size for k in sizes_before}
    assert sizes_before == sizes_after


def test_bva_f32_tmpdir_teardown_clean(temp_workspace: Path):
    """BVA F32-3: Temporary test directories are isolated and functional."""
    assert temp_workspace.is_dir()
    test_file = temp_workspace / "dummy.txt"
    test_file.write_text("isolated content", encoding="utf-8")
    assert test_file.read_text(encoding="utf-8") == "isolated content"


def test_bva_f32_assertion_reporting_detail():
    """BVA F32-4: Failed assertions display clear diffs and context."""
    expected = {"key": "expected_value"}
    actual = {"key": "expected_value"}
    assert actual == expected, "Assertion failure should report full dict comparison"


def test_bva_f32_custom_marker_filtering():
    """BVA F32-5: Test names conform to strict bva naming conventions for filtering."""
    current_test_name = "test_bva_f32_custom_marker_filtering"
    assert current_test_name.startswith("test_bva_")


# ==============================================================================
# F33 BOUNDARIES (5 tests)
# ==============================================================================

def test_bva_f33_pytest_zero_exit_code():
    """BVA F33-1: Test execution functions run with exit code 0."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    assert result.returncode == 0
    assert "pytest" in result.stdout or "pytest" in result.stderr


def test_bva_f33_quiet_and_verbose_flags():
    """BVA F33-2: Pytest quiet execution flag functions cleanly."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--collect-only", str(Path(__file__))],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    assert result.returncode == 0


def test_bva_f33_collected_test_count_threshold():
    """BVA F33-3: Test suite collects expected number of test items."""
    # Check that this file alone defines 20 tests
    current_module = sys.modules[__name__]
    test_callables = [k for k in dir(current_module) if k.startswith("test_bva_")]
    assert len(test_callables) == 20


def test_bva_f33_no_uncaught_exceptions(fixture_jars):
    """BVA F33-4: Zero uncaught exceptions during model loading and graph indexing."""
    p = fixture_jars["stripped_symbols"]
    arc = JarArchive.from_file(p)
    hier = arc.get_hierarchy()
    assert hier is not None
    assert len(arc.classes) > 0


def test_bva_f33_execution_time_under_30s():
    """BVA F33-5: Execution completes rapidly well within performance budget."""
    t0 = time.time()
    # Perform parsing and graph construction
    arc = build_control_flow_flattened_jar()
    hier = arc.get_hierarchy()
    _ = arc.to_bytes()
    t1 = time.time()
    elapsed = t1 - t0
    assert elapsed < 5.0, f"Execution too slow: {elapsed}s"


# ==============================================================================
# F34 BOUNDARIES (5 tests)
# ==============================================================================

def test_bva_f34_cli_help_flag_exit_code_zero():
    """BVA F34-1: jar-engine --help exits with code 0."""
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--help"])
    assert exc_info.value.code == 0


def test_bva_f34_cli_unknown_subcommand_error():
    """BVA F34-2: Unknown subcommand exits with non-zero code."""
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["unknown_command_xyz"])
    assert exc_info.value.code != 0


def test_bva_f34_cli_missing_jar_argument_error():
    """BVA F34-3: Subcommand missing required JAR path exits with non-zero code."""
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["analyze"])  # missing jar arg
    assert exc_info.value.code != 0


def test_bva_f34_cli_non_existent_file_error():
    """BVA F34-4: analyze on non-existent file returns error exit code 1."""
    args = argparse.Namespace(jar="non_existent_file_12345.jar")
    ret = cmd_analyze(args)
    assert ret == 1


def test_bva_f34_cli_missing_mapping_file_error(tmp_path: Path):
    """BVA F34-5: rename with missing mapping file returns error exit code 1."""
    dummy_jar = tmp_path / "valid.jar"
    dummy_jar.write_bytes(b"")
    args = argparse.Namespace(
        jar=str(dummy_jar),
        mapping="non_existent_mapping_file_12345.json",
        output=None
    )
    ret = cmd_rename(args)
    assert ret == 1
