"""
tests/tier3_pairwise/test_pairwise_pipeline_cli.py: Pairwise interactions between Test Infra and Unified CLI.
Covers:
- F31 (Synthetic Test JAR Fixtures) + F34 (Unified CLI & Batch Pipeline) [Analyze]
- F31 (Synthetic Test JAR Fixtures) + F34 (Unified CLI & Batch Pipeline) [Normalize]
- F31 (Synthetic Test JAR Fixtures) + F34 (Unified CLI & Batch Pipeline) [Rename]
- F31 (Synthetic Test JAR Fixtures) + F34 (Unified CLI & Batch Pipeline) [Slice]
- F32 (4-Tier Automated Test Suite) + F33 (Single-Command Test Runner)
- F33 (Single-Command Test Runner) + F34 (Unified CLI & Batch Pipeline)
"""

import json
from pathlib import Path
import pytest

from jar_engine.model.jar_archive import JarArchive
from jar_engine.cli.main import (
    build_parser,
    cmd_analyze,
    cmd_normalize,
    cmd_rename,
    cmd_slice,
)


def test_pairwise_f31_f34_synthetic_fixture_cli_analyze(constant_obfuscated_jar: Path, capsys):
    """
    Pairwise F31 + F34: Synthetic fixture processed directly via unified
    CLI analyze command, asserting exit code 0 and domain report.
    """
    parser = build_parser()
    args = parser.parse_args(["analyze", str(constant_obfuscated_jar)])
    ret = cmd_analyze(args)
    assert ret == 0

    captured = capsys.readouterr()
    assert "Architectural Domain Distribution" in captured.out
    assert "ArithmeticObfuscated" in captured.out or "Total classes" in captured.out


def test_pairwise_f31_f34_synthetic_fixture_cli_normalize(control_flow_flattened_jar: Path, temp_workspace: Path):
    """
    Pairwise F31 + F34: Synthetic fixture processed via CLI normalize command,
    producing clean de-flattened and sanitized output JAR.
    """
    out_jar = temp_workspace / "normalized_cli.jar"
    parser = build_parser()
    args = parser.parse_args(["normalize", str(control_flow_flattened_jar), "-o", str(out_jar)])
    ret = cmd_normalize(args)
    assert ret == 0
    assert out_jar.is_file() and out_jar.stat().st_size > 0

    # Verify output JAR can be read cleanly
    arc = JarArchive.from_file(out_jar)
    assert "com/synthetic/flattened/DispatcherService" in arc.classes


def test_pairwise_f31_f34_synthetic_fixture_cli_rename(stripped_symbols_jar: Path, temp_workspace: Path):
    """
    Pairwise F31 + F34: Synthetic fixture processed via CLI rename command
    with valid JSON mapping file, updating symbols and SPI services.
    """
    mapping_file = temp_workspace / "test_mapping.json"
    mapping_data = {
        "classes": {
            "a/a": "com/example/api/Service",
            "a/d": "com/example/impl/ServiceImpl"
        },
        "fields": {},
        "methods": {}
    }
    mapping_file.write_text(json.dumps(mapping_data, indent=2), encoding="utf-8")

    out_jar = temp_workspace / "renamed_cli.jar"
    parser = build_parser()
    args = parser.parse_args(["rename", str(stripped_symbols_jar), "-m", str(mapping_file), "-o", str(out_jar)])
    ret = cmd_rename(args)
    assert ret == 0
    assert out_jar.is_file()

    # Verify repackaged JAR contains renamed classes and updated SPI services
    reloaded = JarArchive.from_file(out_jar)
    assert "com/example/api/Service" in reloaded.classes
    assert "com/example/impl/ServiceImpl" in reloaded.classes
    assert "META-INF/services/com.example.api.Service" in reloaded.resources


def test_pairwise_f31_f34_synthetic_fixture_cli_slice(constant_obfuscated_jar: Path, capsys):
    """
    Pairwise F31 + F34: Synthetic fixture processed via CLI slice command
    generating JSON contextual summaries and security sink traces.
    """
    parser = build_parser()

    # 1. Target class slice in JSON format
    args_cls = parser.parse_args([
        "slice", str(constant_obfuscated_jar),
        "-c", "com/synthetic/constants/ArithmeticObfuscated",
        "-f", "json"
    ])
    ret_cls = cmd_slice(args_cls)
    assert ret_cls == 0
    captured_cls = capsys.readouterr()
    assert "ArithmeticObfuscated" in captured_cls.out

    # 2. Sink slice
    args_sink = parser.parse_args([
        "slice", str(constant_obfuscated_jar),
        "-s", "String;->toCharArray"
    ])
    ret_sink = cmd_slice(args_sink)
    assert ret_sink == 0
    captured_sink = capsys.readouterr()
    assert "String;->toCharArray" in captured_sink.out


def test_pairwise_f32_f33_four_tier_suite_pytest_runner():
    """
    Pairwise F32 + F33: 4-tier automated test suite discoverable and executable
    through single pytest command runner.
    """
    # Verify pytest discovery of tier3 tests
    ret = pytest.main(["--collect-only", "-q", "tests/tier3_pairwise"])
    assert ret == pytest.ExitCode.OK


def test_pairwise_f33_f34_cli_pipeline_clean_exit_code(
    stripped_symbols_jar: Path, control_flow_flattened_jar: Path, temp_workspace: Path, capsys
):
    """
    Pairwise F33 + F34: CLI batch pipeline executes all subcommands in sequence
    returning exit code 0 at every stage.
    """
    parser = build_parser()

    # Stage 1: Analyze
    args_analyze = parser.parse_args(["analyze", str(stripped_symbols_jar)])
    assert cmd_analyze(args_analyze) == 0

    # Stage 2: Normalize
    norm_out = temp_workspace / "pipeline_normalized.jar"
    args_norm = parser.parse_args(["normalize", str(control_flow_flattened_jar), "-o", str(norm_out)])
    assert cmd_normalize(args_norm) == 0
    assert norm_out.is_file()

    # Stage 3: Rename
    mapping_file = temp_workspace / "pipe_map.json"
    mapping_file.write_text(json.dumps({
        "classes": {"a/a": "com/app/Service"}
    }), encoding="utf-8")
    ren_out = temp_workspace / "pipeline_renamed.jar"
    args_rename = parser.parse_args(["rename", str(stripped_symbols_jar), "-m", str(mapping_file), "-o", str(ren_out)])
    assert cmd_rename(args_rename) == 0
    assert ren_out.is_file()

    # Stage 4: Slice on renamed JAR
    args_slice = parser.parse_args(["slice", str(ren_out), "-c", "com/app/Service", "-f", "markdown"])
    assert cmd_slice(args_slice) == 0
    captured = capsys.readouterr()
    assert "# Class: `com/app/Service`" in captured.out
