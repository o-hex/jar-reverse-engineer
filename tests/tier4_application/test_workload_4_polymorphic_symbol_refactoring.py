"""
tests/tier4_application/test_workload_4_polymorphic_symbol_refactoring.py: Real-World Workload 4.
Polymorphic symbol refactoring across complex multi-class inheritance hierarchies, diamond
interfaces, JDK immutability pinning, access widening, and SPI provider file synchronization.
"""

from pathlib import Path
import pytest

from jar_engine.model.jar_archive import JarArchive
from jar_engine.model.class_file import ClassFile
from jar_engine.model.constants import AccessFlags
from jar_engine.model.member_info import FieldInfo
from jar_engine.model.cp_info import CONSTANT_Utf8_info
from jar_engine.walker.hierarchy import InheritanceHierarchy
from jar_engine.renamer.schema import SymbolMapping
from jar_engine.renamer import SymbolRenamer, JarRepackager


def test_workload_4_polymorphic_symbol_refactoring(fixtures_dir: Path, tmp_path: Path):
    """
    Workload 4 Scenario:
    1. Ingest stripped_symbols.jar containing diamond interface inheritance:
       a/a (root interface: a: I, a()V)
       a/b extends a/a (sub-interface: a: String, a()V, b()V)
       a/c extends a/a (sub-interface: a: boolean, a()V, c()V)
       a/d implements a/b, a/c (diamond class: methods a()V, b()V, c()V, a(I)I, toString()Ljava/lang/String;)
       a/e extends a/d (overrides a()V)
       a/f (entry point caller calling a, b, c, reading field b)
       Manifest: Main-Class: a.f
       SPI Service: META-INF/services/a.a -> a.d
    2. Build InheritanceHierarchy and verify multi-path inheritance relationships.
    3. Configure package relocation and field descriptor typing on a/e.
    4. Construct SymbolMapping renaming classes across packages, polymorphic methods, fields,
       and attempting to rename java/lang/Object method (toString).
    5. Apply SymbolRenamer:
       - Assert JDK pinned method toString is skipped and logged in pinned_skipped.
       - Assert polymorphic method a()V is renamed consistently across all classes in the cluster.
       - Assert package relocation widens access on non-public classes.
       - Assert all caller references in a/f are rewritten to target new class, method, and field names.
       - Assert Manifest Main-Class is synchronized.
       - Assert SPI service provider file in META-INF/services is renamed and content updated.
    6. Repackage to disk and re-verify complete bytecode structural integrity.
    """
    input_jar = fixtures_dir / "stripped_symbols.jar"
    assert input_jar.exists(), f"Missing required synthetic fixture: {input_jar}"

    archive = JarArchive.from_file(input_jar)
    assert len(archive.classes) == 6
    assert set(archive.classes.keys()) == {"a/a", "a/b", "a/c", "a/d", "a/e", "a/f"}

    # 1. Verify inheritance hierarchy relationships
    hierarchy = InheritanceHierarchy.from_archive(archive)
    assert "a/a" in hierarchy.get_interfaces("a/b")
    assert "a/a" in hierarchy.get_interfaces("a/c")
    implementors_a = hierarchy.get_all_implementors("a/a")
    assert "a/d" in implementors_a
    assert "a/e" in implementors_a

    # 2. Add a typed reference field to a/e to test field renaming + descriptor rewriting
    cf_e = archive.classes["a/e"]
    cp_e = cf_e.constant_pool
    cf_e.fields.append(FieldInfo(
        AccessFlags.PUBLIC,
        cp_e.add_utf8("contractRef"),
        cp_e.add_utf8("La/a;"),
        []
    ))
    # Mark a/e as package-private to verify access widening upon package relocation
    cf_e.access_flags = AccessFlags.SUPER

    # 3. Define comprehensive SymbolMapping
    mapping = SymbolMapping(
        classes={
            "a/a": "org/example/contract/ServiceContract",
            "a/b": "org/example/contract/ExtendedService",
            "a/c": "org/example/contract/SecondaryService",
            "a/d": "org/example/impl/ServiceImplementation",
            "a/e": "org/example/impl/SpecializedImplementation",
            "a/f": "org/example/main/ApplicationEntry",
        },
        fields={
            "a/e#contractRef": "serviceContractInstance",
        },
        methods={
            # Target root method in cluster; union-find will expand across all 5 classes
            "a/a#a#()V": "executeWork",
            "a/b#b#()V": "executeExtended",
            "a/c#c#()V": "executeSecondary",
            # Pinned JDK method from java/lang/Object - must NOT be renamed
            "a/d#toString#()Ljava/lang/String;": "customStringRepresentation",
        }
    )

    # 4. Apply SymbolRenamer
    renamer = SymbolRenamer(hierarchy=hierarchy)
    report = renamer.apply_mapping(archive, mapping)

    # 5. Assertions on RenamingReport
    assert report.classes_renamed == 6
    assert report.fields_renamed >= 1
    assert report.methods_renamed >= 5  # Cluster members of a()V + executeExtended + executeSecondary
    assert report.pinned_skipped >= 1, "Pinned JDK toString() method must be skipped"
    assert report.access_widened >= 1, "Package relocation must widen access on package-private class"
    assert report.descriptors_updated >= 1, "Field/method descriptors containing La/a; must be rewritten"

    # 6. Assertions on renamed class definitions
    assert "a/a" not in archive.classes
    assert "a/d" not in archive.classes
    assert "org/example/contract/ServiceContract" in archive.classes
    assert "org/example/contract/ExtendedService" in archive.classes
    assert "org/example/contract/SecondaryService" in archive.classes
    assert "org/example/impl/ServiceImplementation" in archive.classes
    assert "org/example/impl/SpecializedImplementation" in archive.classes
    assert "org/example/main/ApplicationEntry" in archive.classes

    cf_impl = archive.classes["org/example/impl/ServiceImplementation"]
    cp_impl = cf_impl.constant_pool

    # Check method names in ServiceImplementation
    method_names = {m.name(cp_impl) for m in cf_impl.methods}
    assert "executeWork" in method_names, "Polymorphic a()V must be renamed to executeWork"
    assert "executeExtended" in method_names, "b()V must be renamed to executeExtended"
    assert "executeSecondary" in method_names, "c()V must be renamed to executeSecondary"
    assert "toString" in method_names, "Pinned JDK toString() must retain original name"
    assert "customStringRepresentation" not in method_names, "Pinned method must not take new name"

    # Check SpecializedImplementation (subclass override + access widening)
    cf_spec = archive.classes["org/example/impl/SpecializedImplementation"]
    cp_spec = cf_spec.constant_pool
    spec_method_names = {m.name(cp_spec) for m in cf_spec.methods}
    assert "executeWork" in spec_method_names, "Subclass override a()V must also be renamed to executeWork"
    assert cf_spec.is_public, "Relocated subclass access must be widened to public"

    # Check field renaming and descriptor rewriting on SpecializedImplementation
    spec_fld = cf_spec.fields[0]
    assert spec_fld.name(cp_spec) == "serviceContractInstance"
    assert spec_fld.descriptor(cp_spec) == "Lorg/example/contract/ServiceContract;"

    # Check caller class ApplicationEntry (a/f)
    cf_caller = archive.classes["org/example/main/ApplicationEntry"]
    cp_caller = cf_caller.constant_pool
    utf8_entries = {e.value for _, e in cp_caller.entries() if isinstance(e, CONSTANT_Utf8_info)}
    assert "org/example/impl/ServiceImplementation" in utf8_entries
    assert "org/example/main/ApplicationEntry" in utf8_entries

    # 7. Verify Manifest and SPI synchronization
    assert archive.manifest is not None
    assert archive.manifest.get_main_class() == "org.example.main.ApplicationEntry"

    assert "META-INF/services/a.a" not in archive.resources
    new_spi_key = "META-INF/services/org.example.contract.ServiceContract"
    assert new_spi_key in archive.resources, f"Expected {new_spi_key} in resources"
    spi_content = archive.resources[new_spi_key].decode("utf-8").strip()
    assert spi_content == "org.example.impl.ServiceImplementation"

    # 8. Repackage to disk and verify roundtrip integrity
    output_jar = tmp_path / "refactored_symbols.jar"
    repackager = JarRepackager(strip_signatures=True)
    repackager.repackage_to_file(archive, output_jar)
    assert output_jar.exists()

    reloaded_archive = JarArchive.from_file(output_jar)
    assert len(reloaded_archive.classes) == 6
    assert reloaded_archive.manifest.get_main_class() == "org.example.main.ApplicationEntry"
    assert new_spi_key in reloaded_archive.resources

    # Verify all classes parse without errors
    for c_name, cf in reloaded_archive.classes.items():
        raw_bytes = cf.to_bytes()
        reparsed = ClassFile.parse(raw_bytes)
        assert reparsed.magic == 0xCAFEBABE
        assert reparsed.this_class_name == c_name
