"""
tests/conftest.py: Session-scoped and function-scoped Pytest fixtures.
Provides deterministic synthetic test JARs, isolated temporary test workspaces,
and class/archive factory builders for test suites.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Callable
import pytest

from jar_engine.model.constants import AccessFlags
from jar_engine.model.cp_info import ConstantPool
from jar_engine.model.class_file import ClassFile
from jar_engine.model.jar_archive import JarArchive, Manifest
from tests.fixtures.make_fixtures import ensure_fixtures, make_all_fixtures


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Session-scoped fixture returning directory containing the synthetic test JARs."""
    paths = ensure_fixtures()
    return paths["control_flow_flattened"].parent


@pytest.fixture(scope="session")
def fixture_jars() -> Dict[str, Path]:
    """Session-scoped fixture returning mapping of all 4 synthetic test JAR paths."""
    return ensure_fixtures()


@pytest.fixture(scope="session")
def control_flow_flattened_jar(fixture_jars: Dict[str, Path]) -> Path:
    """Path to generated control_flow_flattened.jar."""
    return fixture_jars["control_flow_flattened"]


@pytest.fixture(scope="session")
def control_flow_jar(fixture_jars: Dict[str, Path]) -> Path:
    """Convenience alias for control_flow_flattened_jar."""
    return fixture_jars["control_flow_flattened"]


@pytest.fixture(scope="session")
def constant_obfuscated_jar(fixture_jars: Dict[str, Path]) -> Path:
    """Path to generated constant_obfuscated.jar."""
    return fixture_jars["constant_obfuscated"]


@pytest.fixture(scope="session")
def stripped_symbols_jar(fixture_jars: Dict[str, Path]) -> Path:
    """Path to generated stripped_symbols.jar."""
    return fixture_jars["stripped_symbols"]


@pytest.fixture(scope="session")
def corrupted_structures_jar(fixture_jars: Dict[str, Path]) -> Path:
    """Path to generated corrupted_structures.jar."""
    return fixture_jars["corrupted_structures"]


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """Provides a fresh isolated temporary workspace directory per test function."""
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    return ws


@pytest.fixture
def isolated_workspace(temp_workspace: Path) -> Path:
    """Convenience alias for temp_workspace."""
    return temp_workspace


@pytest.fixture
def make_test_class() -> Callable[..., ClassFile]:
    """Factory fixture to create minimal valid in-memory ClassFile instances."""
    def _create(
        name: str = "com/example/TestClass",
        super_name: str = "java/lang/Object",
        interfaces: Optional[List[str]] = None,
        access_flags: int = AccessFlags.PUBLIC
    ) -> ClassFile:
        cp = ConstantPool()
        this_c = cp.add_class(name)
        super_c = cp.add_class(super_name) if super_name else 0
        ifaces = [cp.add_class(i) for i in (interfaces or [])]
        return ClassFile(
            magic=0xCAFEBABE,
            minor_version=0,
            major_version=52,
            constant_pool=cp,
            access_flags=access_flags,
            this_class=this_c,
            super_class=super_c,
            interfaces=ifaces,
            fields=[],
            methods=[],
            attributes=[]
        )
    return _create


@pytest.fixture
def make_test_archive(make_test_class: Callable[..., ClassFile]) -> Callable[..., JarArchive]:
    """Factory fixture to create in-memory JarArchive instances."""
    def _create(classes: Optional[List[ClassFile]] = None, main_class: Optional[str] = None) -> JarArchive:
        archive = JarArchive()
        if main_class:
            archive.manifest = Manifest(main_attributes={"Main-Class": main_class})
        for c in (classes or [make_test_class()]):
            archive.add_class(c)
        return archive
    return _create
