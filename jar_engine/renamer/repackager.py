"""
jar_engine.renamer.repackager: JAR archive repackager and SPI provider synchronizer (Feature F29).

Ensures the repacked JAR adheres to RFC 822 Manifest specifications, synchronizes META-INF/services/
SPI provider registrations, and strips invalid cryptographic signatures.
"""

from __future__ import annotations
from typing import Dict, List, Optional
from pathlib import Path
import re

from jar_engine.model.jar_archive import JarArchive
from .schema import SymbolMapping


class JarRepackager:
    """Synchronizes JAR metadata and repackages archives after renaming passes."""

    def __init__(self, strip_signatures: bool = True):
        self.strip_signatures = strip_signatures

    def synchronize_manifest_and_services(
        self,
        archive: JarArchive,
        mapping: SymbolMapping
    ) -> int:
        """
        Updates Main-Class in MANIFEST.MF and provider class names in META-INF/services/*.
        Returns count of metadata items synchronized.
        """
        synced_count = 0
        class_renames = mapping.classes

        # 1. Synchronize Manifest Main-Class
        if archive.manifest is not None:
            main_cls = archive.manifest.get_main_class()
            if main_cls:
                # Check both dot and slash versions
                slash_name = main_cls.replace(".", "/")
                if slash_name in class_renames:
                    new_slash_name = class_renames[slash_name]
                    archive.manifest.set_main_class(new_slash_name)
                    synced_count += 1

        # 2. Synchronize META-INF/services/ SPI files
        spi_prefix = "META-INF/services/"
        new_resources: Dict[str, bytes] = {}

        for res_name, res_data in list(archive.resources.items()):
            if res_name.startswith(spi_prefix):
                service_interface = res_name[len(spi_prefix):].replace(".", "/")
                new_service_interface = class_renames.get(service_interface, service_interface)
                new_res_name = f"{spi_prefix}{new_service_interface.replace('/', '.')}"

                # Update provider class names inside the text file
                text = res_data.decode("utf-8", errors="replace")
                lines = text.splitlines()
                updated_lines = []
                for line in lines:
                    trimmed = line.strip()
                    if trimmed and not trimmed.startswith("#"):
                        provider_slash = trimmed.replace(".", "/")
                        new_provider = class_renames.get(provider_slash, provider_slash)
                        updated_lines.append(new_provider.replace("/", "."))
                        synced_count += 1
                    else:
                        updated_lines.append(line)

                new_data = ("\n".join(updated_lines) + "\n").encode("utf-8")
                new_resources[new_res_name] = new_data
            else:
                new_resources[res_name] = res_data

        archive.resources = new_resources

        # 3. Strip invalid signatures
        if self.strip_signatures:
            archive.strip_signatures()

        return synced_count

    def repackage_to_file(
        self,
        archive: JarArchive,
        output_path: Path | str,
        mapping: Optional[SymbolMapping] = None
    ) -> None:
        """Applies metadata synchronization and writes the clean JAR to disk."""
        if mapping:
            self.synchronize_manifest_and_services(archive, mapping)
        archive.save(output_path, strip_signatures=self.strip_signatures)
