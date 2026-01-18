from __future__ import annotations
from typing import Any, Dict, List, Optional
import re

class Namer:
    """
    Naga naming logic.
    """
    def __init__(self) -> None:
        self.unique: Dict[str, int] = {}
        self.reserved_prefixes: List[str] = []

    def reset(self, module: Any) -> None:
        """
        Reset the namer and prepare it for a new module.
        
        Args:
            module: The Naga IR module to name.
        """
        self.unique = {}
        self.reserved_prefixes = []
        
        # In a full implementation, we would populate a name map here.
        # This is a skeleton implementation.

    def call(self, label_raw: str) -> str:
        """
        Return a new identifier based on label_raw.
        
        Args:
            label_raw: The suggested name for the identifier.
            
        Returns:
            A unique identifier string.
        """
        base = self._sanitize(label_raw)
        
        for prefix in self.reserved_prefixes:
            if base.startswith(prefix):
                base = f"gen_{base}"
                break

        if base in self.unique:
            self.unique[base] += 1
            return f"{base}_{self.unique[base]}"
        else:
            self.unique[base] = 0
            return base

    def _sanitize(self, label: str) -> str:
        """Sanitize a label to be a valid identifier."""
        # Remove characters that are not alphanumeric or underscore
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', label)
        # Drop leading digits
        sanitized = re.sub(r'^[0-9]+', '', sanitized)
        # Handle empty or still invalid starts
        if not sanitized or not (sanitized[0].isalpha() or sanitized[0] == '_'):
            sanitized = "unnamed" + sanitized
            
        return sanitized
