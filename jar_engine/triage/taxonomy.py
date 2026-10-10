"""
jar_engine.triage.taxonomy: 10-domain architectural capability taxonomy and API signatures (Feature F20).

Categorizes classes into architectural domains by matching JVM member references,
invocations, and interface types against known JDK and framework API footprints.
"""

from __future__ import annotations
from enum import Enum
from typing import Dict, List, Set, Tuple


class CapabilityDomain(str, Enum):
    CRYPTOGRAPHY = "Cryptography"
    NETWORK = "Network"
    STORAGE_IO = "Storage/IO"
    REFLECTION = "Reflection"
    PROCESS_ENV = "Process/Environment"
    SERIALIZATION = "Serialization"
    DATABASE = "Database"
    GUI_CLIENT = "GUI/Client"
    AUDIO_MEDIA = "Audio/Media"
    SECURITY_POLICY = "Security/Policy"


# Signature patterns: internal package/class prefixes and high-signal method signatures
DOMAIN_SIGNATURES: Dict[CapabilityDomain, List[str]] = {
    CapabilityDomain.CRYPTOGRAPHY: [
        "javax/crypto/",
        "java/security/",
        "javax/net/ssl/",
        "org/bouncycastle/",
        "Cipher",
        "MessageDigest",
        "SecretKey",
        "KeyStore",
        "Mac",
        "Signature",
        "Certificate",
    ],
    CapabilityDomain.NETWORK: [
        "java/net/",
        "java/net/http/",
        "javax/net/",
        "org/apache/http/",
        "okhttp3/",
        "Socket",
        "ServerSocket",
        "HttpURLConnection",
        "URL",
        "URI",
        "DatagramSocket",
    ],
    CapabilityDomain.STORAGE_IO: [
        "java/io/",
        "java/nio/",
        "File",
        "FileInputStream",
        "FileOutputStream",
        "RandomAccessFile",
        "Path",
        "Files",
        "ZipFile",
        "JarFile",
    ],
    CapabilityDomain.REFLECTION: [
        "java/lang/reflect/",
        "java/lang/invoke/",
        "sun/misc/Unsafe",
        "jdk/internal/misc/Unsafe",
        "Class;->forName",
        "Class;->getMethod",
        "Class;->getDeclaredMethod",
        "Class;->getField",
        "Method;->invoke",
        "Field;->get",
        "Field;->set",
    ],
    CapabilityDomain.PROCESS_ENV: [
        "java/lang/Process",
        "java/lang/ProcessBuilder",
        "java/lang/Runtime;->exec",
        "java/lang/Runtime;->getRuntime",
        "java/lang/System;->getenv",
        "java/lang/System;->getProperty",
        "java/lang/management/",
    ],
    CapabilityDomain.SERIALIZATION: [
        "java/io/Serializable",
        "java/io/Externalizable",
        "java/io/ObjectInputStream",
        "java/io/ObjectOutputStream",
        "com/google/gson/",
        "com/fasterxml/jackson/",
    ],
    CapabilityDomain.DATABASE: [
        "java/sql/",
        "javax/sql/",
        "Connection",
        "PreparedStatement",
        "ResultSet",
        "Statement",
        "DataSource",
        "DriverManager",
    ],
    CapabilityDomain.GUI_CLIENT: [
        "java/awt/",
        "javax/swing/",
        "javafx/",
        "Component",
        "JFrame",
        "JPanel",
        "JButton",
    ],
    CapabilityDomain.AUDIO_MEDIA: [
        "javax/sound/",
        "AudioSystem",
        "Clip",
        "AudioInputStream",
        "SourceDataLine",
    ],
    CapabilityDomain.SECURITY_POLICY: [
        "java/security/Permission",
        "java/security/AccessController",
        "java/lang/SecurityManager",
        "java/lang/ClassLoader;->defineClass",
    ],
}
