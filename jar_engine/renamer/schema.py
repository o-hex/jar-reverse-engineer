"""
jar_engine.renamer.schema: Symbol mapping schema and validation (Feature F24).

Defines strongly-typed data structures and JSON validation rules for class, method,
and field renaming specifications.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
import json
import re


class MappingValidationError(ValueError):
    """Raised when a symbol mapping file violates schema or Java naming rules."""
    pass


IDENTIFIER_REGEX = re.compile(r"^[a-zA-Z_$][a-zA-Z0-9_$]*$")
CLASS_NAME_REGEX = re.compile(r"^[a-zA-Z_$][a-zA-Z0-9_$]*(/[a-zA-Z_$][a-zA-Z0-9_$]*)*$")


@dataclass
class SymbolMapping:
    """Complete specification of identifier renames across a JAR archive."""
    classes: Dict[str, str] = field(default_factory=dict)       # old_internal_name -> new_internal_name
    fields: Dict[str, str] = field(default_factory=dict)        # "class#field#desc" or "class#field" -> new_name
    methods: Dict[str, str] = field(default_factory=dict)      # "class#method#desc" -> new_name

    def validate(self) -> List[str]:
        """Validates all entries against Java naming rules and internal naming conventions."""
        errors: List[str] = []

        # Validate class names
        for old_c, new_c in self.classes.items():
            if not CLASS_NAME_REGEX.match(new_c):
                errors.append(f"Invalid new class internal name: '{new_c}' for '{old_c}'")

        # Validate field names
        for old_f, new_f in self.fields.items():
            if not IDENTIFIER_REGEX.match(new_f):
                errors.append(f"Invalid new field name: '{new_f}' for '{old_f}'")

        # Validate method names
        for old_m, new_m in self.methods.items():
            if not IDENTIFIER_REGEX.match(new_m):
                errors.append(f"Invalid new method name: '{new_m}' for '{old_m}'")

        return errors

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SymbolMapping":
        classes = data.get("classes", {})
        fields = data.get("fields", {})
        methods = data.get("methods", {})
        mapping = cls(classes=classes, fields=fields, methods=methods)
        errs = mapping.validate()
        if errs:
            raise MappingValidationError("; ".join(errs))
        return mapping

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classes": self.classes,
            "fields": self.fields,
            "methods": self.methods
        }

    @classmethod
    def from_json(cls, json_str: str) -> "SymbolMapping":
        return cls.from_dict(json.loads(json_str))

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
