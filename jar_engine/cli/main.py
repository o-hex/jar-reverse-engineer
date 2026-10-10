"""
jar_engine.cli.main: Unified command-line interface for the Java Bytecode Comprehension Engine (Feature F34).

Provides CLI commands for archive analysis, normalization, LLM contextual slicing,
and whole-project symbol renaming and repackaging.
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path
import json

from jar_engine.model.jar_archive import JarArchive
from jar_engine.sanitizer import BytecodeSanitizer
from jar_engine.normalization import MethodNormalizer
from jar_engine.triage import CapabilityTriager, ContextualSlicer
from jar_engine.renamer import SymbolRenamer, SymbolMapping


def cmd_analyze(args: argparse.Namespace) -> int:
    """Performs whole-archive dependency analysis and architectural capability triage."""
    jar_path = Path(args.jar)
    if not jar_path.is_file():
        print(f"Error: Target JAR file not found: {jar_path}", file=sys.stderr)
        return 1

    print(f"[*] Ingesting archive: {jar_path.name}")
    archive = JarArchive.from_file(jar_path)
    cg = archive.get_hierarchy()

    print(f"[*] Total classes: {len(archive.classes)}")
    triager = CapabilityTriager()
    report = triager.triage_archive(archive)

    print("\n[+] Architectural Domain Distribution:")
    for domain, count in sorted(report.domain_distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {domain.value}: {count} classes")

    if report.high_priority_classes:
        print(f"\n[+] High-Priority & Security-Sensitive Classes ({len(report.high_priority_classes)}):")
        for cls_name in report.high_priority_classes[:15]:
            res = report.triage_results[cls_name]
            dom = res.primary_domain.value if res.primary_domain else "Unclassified"
            print(f"  * {cls_name} -> {dom} (Confidence: {res.confidence})")

    return 0


def cmd_normalize(args: argparse.Namespace) -> int:
    """Executes decompiler sanitization, CFG de-flattening, and constant folding."""
    jar_path = Path(args.jar)
    out_path = Path(args.output) if args.output else jar_path.with_name(f"{jar_path.stem}_normalized.jar")

    print(f"[*] Normalizing archive: {jar_path.name}")
    archive = JarArchive.from_file(jar_path)

    # Pass 1: Sanitizer
    sanitizer = BytecodeSanitizer()
    s_report = sanitizer.sanitize_archive(archive)
    print(f"[+] Sanitization: {s_report.classes_modified} classes cleaned "
          f"({s_report.total_signatures_stripped} signatures, {s_report.total_exceptions_fixed} exception ranges)")

    # Pass 2: Method Normalization (CFG & constants)
    normalizer = MethodNormalizer()
    stats = normalizer.normalize_archive(archive)

    print(f"[+] CFG & Constant Normalization: complexity reduced by {stats.complexity_reduced}, "
          f"{stats.constants_folded} constants folded, {stats.dead_blocks_pruned} dead blocks pruned")

    archive.save(out_path)
    from jar_engine.sanitizer.stack_map import recompute_jar_stack_maps
    recompute_jar_stack_maps(out_path)
    print(f"[+] Saved normalized archive to: {out_path}")
    return 0


def cmd_slice(args: argparse.Namespace) -> int:
    """Generates an LLM-friendly contextual slice for a class."""
    jar_path = Path(args.jar)
    archive = JarArchive.from_file(jar_path)

    slicer = ContextualSlicer()
    if args.sink:
        from jar_engine.walker.call_graph import CallGraph
        cg = CallGraph.build(archive)
        sink_data = slicer.generate_sink_slice(args.sink, archive, cg)
        print(json.dumps(sink_data, indent=2))
    elif args.target_class:
        cls_slice = slicer.generate_llm_slice(args.target_class, archive, format=args.format)
        print(cls_slice)
    else:
        print("Error: Specify either --target-class or --sink", file=sys.stderr)
        return 1
    return 0


def cmd_rename(args: argparse.Namespace) -> int:
    """Applies a symbol mapping file to rename identifiers across the archive."""
    jar_path = Path(args.jar)
    mapping_path = Path(args.mapping)
    out_path = Path(args.output) if args.output else jar_path.with_name(f"{jar_path.stem}_renamed.jar")

    if not mapping_path.is_file():
        print(f"Error: Mapping file not found: {mapping_path}", file=sys.stderr)
        return 1

    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = SymbolMapping.from_json(f.read())

    archive = JarArchive.from_file(jar_path)
    renamer = SymbolRenamer()
    report = renamer.apply_mapping(archive, mapping)

    print(f"[+] Symbol Renaming Completed:")
    print(f"  - Classes renamed: {report.classes_renamed}")
    print(f"  - Fields renamed: {report.fields_renamed}")
    print(f"  - Methods renamed: {report.methods_renamed}")
    print(f"  - Descriptors updated: {report.descriptors_updated}")
    print(f"  - Pinned methods preserved: {report.pinned_skipped}")

    archive.save(out_path)
    print(f"[+] Saved refactored archive to: {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jar-engine",
        description="Java Bytecode Analysis, Deobfuscation, and Comprehension Engine"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Triage archive and display capability distribution")
    p_analyze.add_argument("jar", help="Path to input JAR archive")

    # normalize
    p_norm = subparsers.add_parser("normalize", help="Run sanitization, CFG unflattening, and constant folding")
    p_norm.add_argument("jar", help="Path to input JAR archive")
    p_norm.add_argument("-o", "--output", help="Output normalized JAR path")

    # slice
    p_slice = subparsers.add_parser("slice", help="Generate token-efficient LLM slice")
    p_slice.add_argument("jar", help="Path to input JAR archive")
    p_slice.add_argument("-c", "--target-class", help="Target internal class name (e.g. com/example/Foo)")
    p_slice.add_argument("-s", "--sink", help="Security sink pattern to slice backward (e.g. Cipher;->doFinal)")
    p_slice.add_argument("-f", "--format", default="markdown", choices=["json", "markdown"], help="Output format")

    # rename
    p_rename = subparsers.add_parser("rename", help="Apply symbol mapping and repackage archive")
    p_rename.add_argument("jar", help="Path to input JAR archive")
    p_rename.add_argument("-m", "--mapping", required=True, help="Path to mapping JSON file")
    p_rename.add_argument("-o", "--output", help="Output repackaged JAR path")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    handlers = {
        "analyze": cmd_analyze,
        "normalize": cmd_normalize,
        "slice": cmd_slice,
        "rename": cmd_rename,
    }

    handler = handlers.get(args.command)
    if handler:
        sys.exit(handler(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
