"""
tests/tier4_application/test_workload_5_bytecode_normalization_complexity.py: Real-World Workload 5.
Bytecode normalization, control flow de-flattening, opaque predicate elimination,
constant folding, dead code pruning, and cyclomatic complexity reduction.
"""

from pathlib import Path
import pytest

from jar_engine.model.jar_archive import JarArchive
from jar_engine.model.class_file import ClassFile
from jar_engine.normalization import MethodNormalizer, NormalizationStats


def test_workload_5_bytecode_normalization_complexity(fixtures_dir: Path, tmp_path: Path):
    """
    Workload 5 Scenario:
    1. Ingest control_flow_flattened.jar and constant_obfuscated.jar synthetic fixtures.
    2. Normalize DispatcherService in control_flow_flattened.jar:
       - dispatchAlgorithm: state-variable tableswitch dispatch loop.
         Verify cyclomatic complexity strictly decreases (4 -> 1) and deflattened=True.
       - opaqueBranchAlgorithm: algebraic tautologies ((x^x)==0, (x-x)==0).
         Verify cyclomatic complexity decreases (3 -> 1) and dead bogus blocks pruned (>=2).
       - redundantChainsAndNops: 4 leading NOPs and unreachable tail return.
         Verify nops_stripped == 4 and dead_blocks_pruned >= 1.
       - loopDispatch: state machine dispatch with back-edge loop.
         Verify cyclomatic complexity decreases (5 -> 3), deflattened=True, dead_blocks_pruned >= 1.
    3. Normalize ArithmeticObfuscated in constant_obfuscated.jar:
       - evaluateArithmetic: (10 + 20) * 2 - 15 -> folded (3 operations).
       - evaluateBitwise: (15 | 16) & 5 -> folded (2 operations).
       - evaluateShift: 1 << 4 -> folded (1 operation).
       - evaluateLongArithmetic: 1000L + 2000L -> folded (1 operation).
       - evaluateFloatArithmetic: 3.5f - 1.5f -> folded (1 operation).
    4. Normalize StringObfuscated in constant_obfuscated.jar:
       - getSecretMessage: decrypts "IFMMP" -> inlined "HELLO" (strings_decrypted == 1).
       - getDatabaseKey: decrypts "TFDSFU" -> inlined "SECRET" (strings_decrypted == 1).
    5. Normalize DynamicInvoker in constant_obfuscated.jar:
       - concatSecret: StringConcatFactory dynamic invoke resolved to constant.
       - getWorker: LambdaMetafactory call site resolved to direct methodref.
    6. Verify all normalized classes reassemble, serialize to bytes, and parse cleanly via ClassFile.parse.
    """
    flat_jar = fixtures_dir / "control_flow_flattened.jar"
    const_jar = fixtures_dir / "constant_obfuscated.jar"
    assert flat_jar.exists(), f"Missing required synthetic fixture: {flat_jar}"
    assert const_jar.exists(), f"Missing required synthetic fixture: {const_jar}"

    normalizer = MethodNormalizer()

    # =========================================================================
    # Part 1: Control Flow Flattening & Complexity Reduction
    # =========================================================================
    flat_archive = JarArchive.from_file(flat_jar)
    cf_dispatcher = flat_archive.classes["com/synthetic/flattened/DispatcherService"]
    cp_disp = cf_dispatcher.constant_pool

    method_map = {m.name(cp_disp): m for m in cf_dispatcher.methods}

    # 1. dispatchAlgorithm: tableswitch dispatcher loop
    m_disp = method_map["dispatchAlgorithm"]
    stats_disp = normalizer.normalize_method(cf_dispatcher, m_disp)
    assert stats_disp.cyclomatic_complexity_before == 4
    assert stats_disp.cyclomatic_complexity_after == 1
    assert stats_disp.cyclomatic_complexity_before > stats_disp.cyclomatic_complexity_after
    assert stats_disp.complexity_reduced == 3
    assert stats_disp.deflattened is True

    # 2. opaqueBranchAlgorithm: algebraic opaque predicates
    m_opaque = method_map["opaqueBranchAlgorithm"]
    stats_opaque = normalizer.normalize_method(cf_dispatcher, m_opaque)
    assert stats_opaque.cyclomatic_complexity_before == 3
    assert stats_opaque.cyclomatic_complexity_after == 1
    assert stats_opaque.cyclomatic_complexity_before > stats_opaque.cyclomatic_complexity_after
    assert stats_opaque.complexity_reduced == 2
    assert stats_opaque.dead_blocks_pruned >= 2

    # 3. redundantChainsAndNops: NOP instructions and dead tail block
    m_redundant = method_map["redundantChainsAndNops"]
    stats_redundant = normalizer.normalize_method(cf_dispatcher, m_redundant)
    assert stats_redundant.nops_stripped == 4
    assert stats_redundant.dead_blocks_pruned >= 1

    # 4. loopDispatch: state-variable loop
    m_loop = method_map["loopDispatch"]
    stats_loop = normalizer.normalize_method(cf_dispatcher, m_loop)
    assert stats_loop.cyclomatic_complexity_before == 5
    assert stats_loop.cyclomatic_complexity_after == 3
    assert stats_loop.cyclomatic_complexity_before > stats_loop.cyclomatic_complexity_after
    assert stats_loop.complexity_reduced == 2
    assert stats_loop.deflattened is True
    assert stats_loop.dead_blocks_pruned >= 1

    # Verify reassembled DispatcherService parses cleanly
    raw_disp = cf_dispatcher.to_bytes()
    reparsed_disp = ClassFile.parse(raw_disp)
    assert reparsed_disp.magic == 0xCAFEBABE
    assert reparsed_disp.this_class_name == "com/synthetic/flattened/DispatcherService"

    # =========================================================================
    # Part 2: Constant Folding Lattice Evaluation
    # =========================================================================
    const_archive = JarArchive.from_file(const_jar)
    cf_arith = const_archive.classes["com/synthetic/constants/ArithmeticObfuscated"]
    cp_arith = cf_arith.constant_pool
    arith_methods = {m.name(cp_arith): m for m in cf_arith.methods}

    # evaluateArithmetic: 3 operations folded
    stats_arith = normalizer.normalize_method(cf_arith, arith_methods["evaluateArithmetic"])
    assert stats_arith.constants_folded >= 3

    # evaluateBitwise: 2 operations folded
    stats_bit = normalizer.normalize_method(cf_arith, arith_methods["evaluateBitwise"])
    assert stats_bit.constants_folded >= 2

    # evaluateShift: 1 operation folded
    stats_shift = normalizer.normalize_method(cf_arith, arith_methods["evaluateShift"])
    assert stats_shift.constants_folded >= 1

    # evaluateLongArithmetic: 1 long operation folded
    stats_long = normalizer.normalize_method(cf_arith, arith_methods["evaluateLongArithmetic"])
    assert stats_long.constants_folded >= 1

    # evaluateFloatArithmetic: 1 float operation folded
    stats_float = normalizer.normalize_method(cf_arith, arith_methods["evaluateFloatArithmetic"])
    assert stats_float.constants_folded >= 1

    # =========================================================================
    # Part 3: Static String Decryption Inlining
    # =========================================================================
    cf_str = const_archive.classes["com/synthetic/constants/StringObfuscated"]
    cp_str = cf_str.constant_pool
    str_methods = {m.name(cp_str): m for m in cf_str.methods}

    # getSecretMessage: dec("IFMMP") -> inlined "HELLO"
    stats_msg1 = normalizer.normalize_method(cf_str, str_methods["getSecretMessage"])
    assert stats_msg1.strings_decrypted == 1

    # getDatabaseKey: dec("TFDSFU") -> inlined "SECRET"
    stats_msg2 = normalizer.normalize_method(cf_str, str_methods["getDatabaseKey"])
    assert stats_msg2.strings_decrypted == 1

    # =========================================================================
    # Part 4: Dynamic Invoke Resolution
    # =========================================================================
    cf_indy = const_archive.classes["com/synthetic/constants/DynamicInvoker"]
    cp_indy = cf_indy.constant_pool
    indy_methods = {m.name(cp_indy): m for m in cf_indy.methods}

    # concatSecret: StringConcatFactory resolution
    stats_indy1 = normalizer.normalize_method(cf_indy, indy_methods["concatSecret"])
    assert stats_indy1.dynamic_invokes_resolved == 1

    # getWorker: LambdaMetafactory resolution
    stats_indy2 = normalizer.normalize_method(cf_indy, indy_methods["getWorker"])
    assert stats_indy2.dynamic_invokes_resolved == 1

    # =========================================================================
    # Part 5: Complete Archive Bytecode Reassembly & Parse Verification
    # =========================================================================
    for archive in (flat_archive, const_archive):
        for c_name, cf in archive.classes.items():
            raw_bytes = cf.to_bytes()
            reparsed = ClassFile.parse(raw_bytes)
            assert reparsed.magic == 0xCAFEBABE
            assert reparsed.this_class_name == c_name
