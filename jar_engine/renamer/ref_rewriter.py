"""
jar_engine.renamer.ref_rewriter: Multi-class bytecode reference rewriter and access widener (Features F27, F28).

Rewrites class names, field names, and method names across constant pool entries,
method/field descriptors, and bytecode instructions across all classes in a JAR archive,
while widening access flags when classes move packages to prevent IllegalAccessError.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import re

from jar_engine.model.class_file import ClassFile
from jar_engine.model.jar_archive import JarArchive
from jar_engine.model.constants import AccessFlags, ConstantTag
from jar_engine.model.cp_info import (
    CONSTANT_Utf8_info, CONSTANT_Class_info, CONSTANT_NameAndType_info,
    CONSTANT_Fieldref_info, CONSTANT_Methodref_info, CONSTANT_InterfaceMethodref_info
)
from .schema import SymbolMapping
from .union_find import MethodEquivalenceCluster


@dataclass
class RenamingReport:
    """Statistics on symbols updated during renaming."""
    classes_renamed: int = 0
    fields_renamed: int = 0
    methods_renamed: int = 0
    descriptors_updated: int = 0
    access_widened: int = 0
    pinned_skipped: int = 0


def rewrite_descriptor(descriptor: str, class_renames: Dict[str, str]) -> str:
    """Rewrites class internal names within a method or field type descriptor."""
    if not class_renames:
        return descriptor

    # Matches L<internal_name>; in descriptors
    def replace_type(match: re.Match) -> str:
        old_name = match.group(1)
        new_name = class_renames.get(old_name, old_name)
        return f"L{new_name};"

    return re.sub(r"L([a-zA-Z0-9_$/]+);", replace_type, descriptor)


class ReferenceRewriter:
    """Rewrites symbol definitions and constant pool references across all classes in a JarArchive."""

    def __init__(self, cluster: Optional[MethodEquivalenceCluster] = None):
        self.cluster = cluster

    def apply_mapping(self, archive: JarArchive, mapping: SymbolMapping) -> RenamingReport:
        report = RenamingReport()
        class_renames = mapping.classes
        field_renames = mapping.fields
        method_renames = mapping.methods

        # Step 1: Expand method renames across polymorphic clusters
        expanded_method_renames: Dict[Tuple[str, str, str], str] = {}
        all_method_keys: Set[str] = set()

        for c_name, cf in archive.classes.items():
            cp = cf.constant_pool
            for m in cf.methods:
                all_method_keys.add(f"{c_name}#{m.name(cp)}#{m.descriptor(cp)}")

        for m_key, new_name in method_renames.items():
            # If cluster is available, verify pinning and expand to entire cluster
            if self.cluster:
                if self.cluster.is_method_pinned(m_key):
                    report.pinned_skipped += 1
                    continue
                cluster_members = self.cluster.get_cluster_members(m_key, all_method_keys)
                for member in cluster_members:
                    parts = member.split("#")
                    if len(parts) == 3:
                        expanded_method_renames[(parts[0], parts[1], parts[2])] = new_name
            else:
                parts = m_key.split("#")
                if len(parts) == 3:
                    expanded_method_renames[(parts[0], parts[1], parts[2])] = new_name

        # Step 2: Rewrite definitions in each class
        for old_c_name, cf in list(archive.classes.items()):
            cp = cf.constant_pool

            # Rename this class definition
            if old_c_name in class_renames:
                new_c_name = class_renames[old_c_name]
                # Update UTF-8 entry pointed to by this_class CONSTANT_Class_info
                this_cls_info = cp._entries[cf.this_class]
                if isinstance(this_cls_info, CONSTANT_Class_info):
                    cp.update_utf8(this_cls_info.name_index, new_c_name)
                report.classes_renamed += 1

                # Access widening: if moved across packages, ensure public access
                old_pkg = old_c_name.rsplit("/", 1)[0] if "/" in old_c_name else ""
                new_pkg = new_c_name.rsplit("/", 1)[0] if "/" in new_c_name else ""
                if old_pkg != new_pkg and not cf.is_public:
                    cf.access_flags = (cf.access_flags & ~AccessFlags.PRIVATE & ~AccessFlags.PROTECTED) | AccessFlags.PUBLIC
                    report.access_widened += 1

            # Rename fields
            for field in cf.fields:
                f_name = field.name(cp)
                f_desc = field.descriptor(cp)
                f_key = f"{old_c_name}#{f_name}"
                f_key_desc = f"{old_c_name}#{f_name}#{f_desc}"

                new_name = field_renames.get(f_key_desc, field_renames.get(f_key))
                if new_name:
                    cp.update_utf8(field.name_index, new_name)
                    report.fields_renamed += 1

                # Update field type descriptor if referenced classes were renamed
                new_desc = rewrite_descriptor(f_desc, class_renames)
                if new_desc != f_desc:
                    cp.update_utf8(field.descriptor_index, new_desc)
                    report.descriptors_updated += 1

            # Rename methods
            for method in cf.methods:
                m_name = method.name(cp)
                m_desc = method.descriptor(cp)

                # Check expanded renames
                new_name = expanded_method_renames.get((old_c_name, m_name, m_desc))
                if new_name:
                    cp.update_utf8(method.name_index, new_name)
                    report.methods_renamed += 1

                # Update method descriptor (parameter and return types)
                new_desc = rewrite_descriptor(m_desc, class_renames)
                if new_desc != m_desc:
                    cp.update_utf8(method.descriptor_index, new_desc)
                    report.descriptors_updated += 1

        # Step 3: Rewrite all constant pool cross-references across all classes
        for c_name, cf in archive.classes.items():
            cp = cf.constant_pool
            for idx, entry in enumerate(cp._entries):
                if entry is None:
                    continue

                # 1. Update CONSTANT_Class_info references
                if isinstance(entry, CONSTANT_Class_info):
                    try:
                        cls_str = cp.resolve_utf8(entry.name_index)
                        if cls_str in class_renames:
                            cp.update_utf8(entry.name_index, class_renames[cls_str])
                        elif cls_str.startswith("["):
                            new_cls_str = rewrite_descriptor(cls_str, class_renames)
                            if new_cls_str != cls_str:
                                cp.update_utf8(entry.name_index, new_cls_str)
                    except Exception:
                        pass

                # 2. Update CONSTANT_NameAndType_info descriptors and names
                elif isinstance(entry, CONSTANT_NameAndType_info):
                    try:
                        desc = cp.resolve_utf8(entry.descriptor_index)
                        new_desc = rewrite_descriptor(desc, class_renames)
                        if new_desc != desc:
                            cp.update_utf8(entry.descriptor_index, new_desc)
                    except Exception:
                        pass

                # 3. Update CONSTANT_Memberref_info (Fieldref / Methodref / InterfaceMethodref)
                elif isinstance(entry, (CONSTANT_Fieldref_info, CONSTANT_Methodref_info, CONSTANT_InterfaceMethodref_info)):
                    try:
                        owner_class, member_name, member_desc = cp.resolve_member_ref(idx)
                        # Check method renames
                        if isinstance(entry, (CONSTANT_Methodref_info, CONSTANT_InterfaceMethodref_info)):
                            new_m_name = expanded_method_renames.get((owner_class, member_name, member_desc))
                            if new_m_name:
                                nat = cp._entries[entry.name_and_type_index]
                                if isinstance(nat, CONSTANT_NameAndType_info):
                                    cp.update_utf8(nat.name_index, new_m_name)
                        # Check field renames
                        elif isinstance(entry, CONSTANT_Fieldref_info):
                            f_key = f"{owner_class}#{member_name}"
                            f_key_desc = f"{owner_class}#{member_name}#{member_desc}"
                            new_f_name = field_renames.get(f_key_desc, field_renames.get(f_key))
                            if new_f_name:
                                nat = cp._entries[entry.name_and_type_index]
                                if isinstance(nat, CONSTANT_NameAndType_info):
                                    cp.update_utf8(nat.name_index, new_f_name)
                    except Exception:
                        pass

        # Step 4: Re-index archive classes dictionary with new names
        new_classes_dict: Dict[str, ClassFile] = {}
        for old_name, cf in archive.classes.items():
            new_name = class_renames.get(old_name, old_name)
            new_classes_dict[new_name] = cf
        archive.classes = new_classes_dict

        return report
