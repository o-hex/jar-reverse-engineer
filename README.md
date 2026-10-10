# jar-reverse-engineer (`jar-engine`)

A high-performance Java Bytecode Static Analysis, Deobfuscation Normalization, and Program Comprehension Engine built from scratch in pure Python for LLMs and security analysts.

## Overview

Modern Java obfuscators employ control-flow flattening, string encryption, opaque predicates, and broken bytecode structures to defeat decompilers and human analysis. `jar-engine` provides whole-archive static analysis and automated normalization passes to restore clean, readable code and export token-efficient contextual slices (achieving **85–95% token compression**) optimized for LLMs.

---

## Key Features

- **Whole-Archive Dependency & Call Graph Walker**:
  - Pure-Python JVMS §4 class parser and binary serializer with 100% byte-for-byte exact round-trip fidelity.
  - Inter-class call graphs (supporting both CHA and RTA resolution).
  - Class inheritance and interface hierarchy lattice with multidimensional array LCA unwrapping.
  - Tarjan Strongly Connected Components (SCC) condensation and topological dependency slicing.

- **Decompiler Compatibility Preprocessing**:
  - Generic `SignatureAttribute` grammar parser and repair.
  - Exception table sanitizer repairing inverted ranges, out-of-bounds PCs, and self-catching `try-finally` loops.
  - StackMapTable frame validator and Java 6 legacy frame stripper.
  - Synthetic member deduplication and recursive bridge method cycle breaking across all return types.
  - Verified differential decompilation across **CFR**, **Vineflower**, and **Procyon**.

- **Control Flow Normalization & Constant Resolution**:
  - Basic block partitioner, dominators, natural loops, and cyclomatic complexity quantification.
  - Switch-dispatcher de-flattening for state-variable dispatch loops.
  - Algebraic opaque predicate elimination covering 7 invariant families with strict stack neutrality.
  - Abstract interpretation constant folding lattice (32/64-bit integer, IEEE-754 float/double, bitwise shifts).
  - Sandboxed JVM bytecode emulator for string decryption routines and `invokedynamic` string concatenation resolution.
  - Dead code pruning, redundant fallthrough elimination, and tableswitch/lookupswitch byte-padding alignment.

- **Subsystem Capability Triage & Contextual Slicing**:
  - 10-domain architectural capability classifier (*Cryptography*, *Network/Sockets*, *Storage/IO*, *Diagnostics*, *Reflection*, *Serialization*, *UI*, *Concurrency*, *Database*, *Security*).
  - TF-IDF style term frequency and invocation footprint scoring.
  - Token-efficient Markdown & JSON contextual slicing achieving **85–95% token compression** for LLM analysis.

- **Semantic Symbol Renamer & Repackager**:
  - Disjoint-Set Union-Find override clustering ensuring synchronized renaming across polymorphic hierarchies.
  - JDK immutability pinning protecting standard library methods (`Object.equals`, `Runnable.run`, etc.).
  - Constant pool, descriptor, and instruction reference rewriter with access widening across package boundaries.
  - JAR repackaging updating `META-INF/MANIFEST.MF` (`Main-Class`) and `META-INF/services/` SPI provider files.

---

## Installation

```bash
git clone https://github.com/o-hex/jar-reverse-engineer.git
cd jar-reverse-engineer
pip install -e .
```

---

## CLI Usage

### 1. Capability Analysis & Triage
Analyze archive and inspect inferred subsystem capabilities:
```bash
jar-engine analyze <archive.jar>
```

### 2. Bytecode Normalization
Run sanitization, CFG de-flattening, opaque predicate elimination, and constant folding:
```bash
jar-engine normalize <input.jar> -o <clean.jar>
```

### 3. LLM Contextual Slicing
Export token-compressed structural slices in Markdown or JSON format:
```bash
jar-engine slice <archive.jar> -c com/example/TargetClass -f markdown
jar-engine slice <archive.jar> -c com/example/TargetClass -f json
```

### 4. Semantic Symbol Renaming & Repackaging
Refactor symbol names according to a mapping JSON and repackage the output archive:
```bash
jar-engine rename <input.jar> -m <symbol_mapping.json> -o <repackaged.jar>
```

---

## Testing

The test harness features pure-Python synthetic test JAR generators (`control_flow_flattened.jar`, `constant_obfuscated.jar`, `stripped_symbols.jar`, and `corrupted_structures.jar`) and a 6-tier automated test suite:

```bash
pytest --tb=short -q
```

All 601 tests pass cleanly across unit, feature, boundary value, pairwise integration, real-world workload, and adversarial stress tiers.

---

## License

MIT License.
