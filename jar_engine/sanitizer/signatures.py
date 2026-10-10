"""
jar_engine.sanitizer.signatures: JVMS §4.7.9 Generic Signature validator and cleaner.

Validates ClassSignature, MethodSignature, and FieldSignature grammars. Strips corrupted,
truncated, or non-conforming signatures that cause decompiler crashes (CFR, Vineflower, Procyon).
"""

from __future__ import annotations
from typing import Optional, Tuple, Set, List
import re

from jar_engine.model.class_file import ClassFile
from jar_engine.model.member_info import SignatureAttribute, AttributeInfo


class SignatureParser:
    """
    Recursive descent parser for JVM signatures adhering strictly to JVMS §4.7.9.
    """
    def __init__(self, s: str):
        self.s = s
        self.pos = 0
        self.length = len(s)

    def peek(self) -> Optional[str]:
        if self.pos < self.length:
            return self.s[self.pos]
        return None

    def consume(self) -> Optional[str]:
        if self.pos < self.length:
            ch = self.s[self.pos]
            self.pos += 1
            return ch
        return None

    def expect(self, ch: str) -> bool:
        if self.peek() == ch:
            self.pos += 1
            return True
        return False

    def is_eof(self) -> bool:
        return self.pos >= self.length

    # Grammar rules per JVMS §4.7.9
    def parse_identifier(self) -> Optional[str]:
        start = self.pos
        # An identifier cannot contain '.', ';', '[', '/', '<', '>', ':'
        forbidden = {'.', ';', '[', '/', '<', '>', ':'}
        while self.pos < self.length and self.s[self.pos] not in forbidden:
            self.pos += 1
        if self.pos == start:
            return None
        return self.s[start:self.pos]

    def parse_type_variable_signature(self) -> bool:
        # 'T' Identifier ';'
        if not self.expect('T'):
            return False
        ident = self.parse_identifier()
        if not ident or not self.expect(';'):
            return False
        return True

    def parse_type_argument(self) -> bool:
        # WildcardIndicator? ReferenceTypeSignature | '*'
        if self.expect('*'):
            return True
        self.expect('+') or self.expect('-')  # Optional wildcard indicator
        return self.parse_reference_type_signature()

    def parse_type_arguments(self) -> bool:
        # '<' TypeArgument+ '>'
        if not self.expect('<'):
            return False
        count = 0
        while not self.is_eof() and self.peek() != '>':
            if not self.parse_type_argument():
                return False
            count += 1
        if count == 0 or not self.expect('>'):
            return False
        return True

    def parse_class_type_signature(self) -> bool:
        # 'L' [PackageSpecifier] SimpleClassTypeSignature {ClassTypeSignatureSuffix} ';'
        if not self.expect('L'):
            return False
        # Sequence of identifiers separated by '/'
        while True:
            ident = self.parse_identifier()
            if not ident:
                return False
            if self.peek() == '<':
                if not self.parse_type_arguments():
                    return False
            if self.expect('/'):
                continue
            elif self.expect('.'):
                # Suffix
                continue
            elif self.expect(';'):
                return True
            else:
                return False

    def parse_array_type_signature(self) -> bool:
        # '[' JavaTypeSignature
        if not self.expect('['):
            return False
        return self.parse_java_type_signature()

    def parse_reference_type_signature(self) -> bool:
        ch = self.peek()
        if ch == 'L':
            return self.parse_class_type_signature()
        elif ch == 'T':
            return self.parse_type_variable_signature()
        elif ch == '[':
            return self.parse_array_type_signature()
        return False

    def parse_base_type(self) -> bool:
        ch = self.peek()
        if ch in {'B', 'C', 'D', 'F', 'I', 'J', 'S', 'Z'}:
            self.consume()
            return True
        return False

    def parse_java_type_signature(self) -> bool:
        if self.parse_base_type():
            return True
        return self.parse_reference_type_signature()

    def parse_type_parameter(self) -> bool:
        # Identifier ':' [ReferenceTypeSignature] {':' ReferenceTypeSignature}
        ident = self.parse_identifier()
        if not ident or not self.expect(':'):
            return False
        # Optional class bound
        if self.peek() in {'L', 'T', '['}:
            if not self.parse_reference_type_signature():
                return False
        # Zero or more interface bounds
        while self.expect(':'):
            if not self.parse_reference_type_signature():
                return False
        return True

    def parse_type_parameters(self) -> bool:
        # '<' TypeParameter+ '>'
        if not self.expect('<'):
            return False
        count = 0
        while not self.is_eof() and self.peek() != '>':
            if not self.parse_type_parameter():
                return False
            count += 1
        if count == 0 or not self.expect('>'):
            return False
        return True

    def parse_class_signature(self) -> bool:
        # [TypeParameters] SuperclassSignature {SuperinterfaceSignature}
        if self.peek() == '<':
            if not self.parse_type_parameters():
                return False
        # SuperclassSignature (must be a ClassTypeSignature)
        if not self.parse_class_type_signature():
            return False
        # SuperinterfaceSignatures
        while not self.is_eof():
            if not self.parse_class_type_signature():
                return False
        return self.is_eof()

    def parse_method_signature(self) -> bool:
        # [TypeParameters] '(' {JavaTypeSignature} ')' Result {ThrowsSignature}
        if self.peek() == '<':
            if not self.parse_type_parameters():
                return False
        if not self.expect('('):
            return False
        while not self.is_eof() and self.peek() != ')':
            if not self.parse_java_type_signature():
                return False
        if not self.expect(')'):
            return False
        # Result: JavaTypeSignature | 'V'
        if self.peek() == 'V':
            self.consume()
        elif not self.parse_java_type_signature():
            return False
        # ThrowsSignature: '^' (ClassTypeSignature | TypeVariableSignature)
        while self.expect('^'):
            ch = self.peek()
            if ch == 'L':
                if not self.parse_class_type_signature():
                    return False
            elif ch == 'T':
                if not self.parse_type_variable_signature():
                    return False
            else:
                return False
        return self.is_eof()

    def parse_field_signature(self) -> bool:
        # ReferenceTypeSignature
        return self.parse_reference_type_signature() and self.is_eof()


def is_valid_class_signature(sig: str) -> bool:
    """Validates if a string conforms to JVMS §4.7.9 ClassSignature."""
    if not sig:
        return False
    return SignatureParser(sig).parse_class_signature()


def is_valid_method_signature(sig: str) -> bool:
    """Validates if a string conforms to JVMS §4.7.9 MethodSignature."""
    if not sig:
        return False
    return SignatureParser(sig).parse_method_signature()


def is_valid_field_signature(sig: str) -> bool:
    """Validates if a string conforms to JVMS §4.7.9 FieldTypeSignature."""
    if not sig:
        return False
    return SignatureParser(sig).parse_field_signature()


def sanitize_class_signatures(cf: ClassFile) -> int:
    """
    Validates and strips corrupted generic signatures across a ClassFile.
    Returns the total count of stripped signature attributes.
    """
    stripped_count = 0
    cp = cf.constant_pool

    # 1. Class-level Signature
    new_class_attrs: List[AttributeInfo] = []
    for attr in cf.attributes:
        if isinstance(attr, SignatureAttribute) or (hasattr(attr, "name") and attr.name(cp) == "Signature"):
            try:
                sig_str = attr.signature(cp) if hasattr(attr, "signature") else cp.resolve_utf8(attr.signature_index) # type: ignore
                if not is_valid_class_signature(sig_str):
                    stripped_count += 1
                    continue
            except Exception:
                stripped_count += 1
                continue
        new_class_attrs.append(attr)
    cf.attributes = new_class_attrs

    # 2. Field Signatures
    for field in cf.fields:
        new_f_attrs: List[AttributeInfo] = []
        for attr in field.attributes:
            if isinstance(attr, SignatureAttribute) or (hasattr(attr, "name") and attr.name(cp) == "Signature"):
                try:
                    sig_str = attr.signature(cp) if hasattr(attr, "signature") else cp.resolve_utf8(attr.signature_index) # type: ignore
                    if not is_valid_field_signature(sig_str):
                        stripped_count += 1
                        continue
                except Exception:
                    stripped_count += 1
                    continue
            new_f_attrs.append(attr)
        field.attributes = new_f_attrs

    # 3. Method Signatures
    for method in cf.methods:
        new_m_attrs: List[AttributeInfo] = []
        for attr in method.attributes:
            if isinstance(attr, SignatureAttribute) or (hasattr(attr, "name") and attr.name(cp) == "Signature"):
                try:
                    sig_str = attr.signature(cp) if hasattr(attr, "signature") else cp.resolve_utf8(attr.signature_index) # type: ignore
                    if not is_valid_method_signature(sig_str):
                        stripped_count += 1
                        continue
                except Exception:
                    stripped_count += 1
                    continue
            new_m_attrs.append(attr)
        method.attributes = new_m_attrs

    return stripped_count
