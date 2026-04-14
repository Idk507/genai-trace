"""
Prompt version management.

Manages versioned prompts as first-class artifacts.
Prompts are stored locally (JSON file) or remotely (database/API).

Concepts:
- name: logical identifier (e.g., "customer_support_system")
- version: semver string (e.g., "1.2.0")
- label: mutable pointer (e.g., "production", "staging", "latest")

Usage:
    registry = PromptRegistry()
    
    # Save a prompt
    registry.save(
        name="summarize_v2",
        template="Summarize the following in {{max_words}} words:\\n\\n{{text}}",
        version="1.0.0",
        label="production",
        metadata={"author": "alice", "model": "gpt-4o"},
    )
    
    # Fetch by label
    prompt = registry.get("summarize_v2", label="production")
    filled = prompt.compile(max_words=100, text=document)
    
    # Diff two versions
    diff = registry.diff("summarize_v2", "1.0.0", "1.1.0")
"""

import json
import hashlib
import re
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PromptVersion:
    """A versioned prompt template."""
    name: str
    version: str
    template: str
    labels: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    template_hash: str = ""
    
    def __post_init__(self):
        """Compute template hash."""
        if not self.template_hash:
            self.template_hash = hashlib.sha256(self.template.encode()).hexdigest()[:12]
    
    def compile(self, **variables) -> str:
        """
        Render the template by substituting {{variable}} placeholders.
        
        Args:
            **variables: Variable values to substitute
            
        Returns:
            Rendered template string
        """
        result = self.template
        for key, value in variables.items():
            result = result.replace(f"{{{{{key}}}}}", str(value))
        
        # Warn on unrendered placeholders
        remaining = re.findall(r"\{\{(\w+)\}\}", result)
        if remaining:
            warnings.warn(f"Prompt '{self.name}' has unrendered variables: {remaining}")
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class PromptRegistry:
    """
    Registry for managing versioned prompts.
    
    Supports:
    - Saving and retrieving prompts by name/version/label
    - Version diffing
    - Label-based deployment (production, staging, etc.)
    - Rollback to previous versions
    """
    
    def __init__(self, storage_path: str = "./prompt_registry.json"):
        """
        Initialize the prompt registry.
        
        Args:
            storage_path: Path to JSON storage file
        """
        self._path = Path(storage_path)
        self._store: Dict[str, List[dict]] = {}
        self._load()
    
    def save(
        self,
        name: str,
        template: str,
        version: str,
        labels: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
    ) -> PromptVersion:
        """
        Save a new prompt version.
        
        Args:
            name: Prompt name
            template: Template string with {{variable}} placeholders
            version: Semantic version string
            labels: Labels to assign (e.g., ["production"])
            metadata: Additional metadata
            
        Returns:
            Created PromptVersion
        """
        pv = PromptVersion(
            name=name,
            version=version,
            template=template,
            labels=labels or [],
            metadata=metadata or {},
        )
        
        if name not in self._store:
            self._store[name] = []
        
        # Remove labels from other versions if they already exist
        if labels:
            for lbl in labels:
                for existing in self._store[name]:
                    if lbl in existing.get("labels", []):
                        existing["labels"].remove(lbl)
        
        # Check if version already exists
        for i, existing in enumerate(self._store[name]):
            if existing["version"] == version:
                # Update existing version
                self._store[name][i] = asdict(pv)
                self._save()
                return pv
        
        # Add new version
        self._store[name].append(asdict(pv))
        self._save()
        return pv
    
    def get(
        self,
        name: str,
        version: Optional[str] = None,
        label: Optional[str] = None,
    ) -> Optional[PromptVersion]:
        """
        Get a prompt version.
        
        Args:
            name: Prompt name
            version: Specific version to get
            label: Label to get (e.g., "production")
            
        Returns:
            PromptVersion or None if not found
        """
        versions = self._store.get(name, [])
        if not versions:
            return None
        
        # Get by version
        if version:
            for v in versions:
                if v["version"] == version:
                    return PromptVersion(**v)
            return None
        
        # Get by label
        if label:
            for v in versions:
                if label in v.get("labels", []):
                    return PromptVersion(**v)
            return None
        
        # Default: return latest
        return PromptVersion(**versions[-1])
    
    def list_versions(self, name: str) -> List[str]:
        """List all versions of a prompt."""
        return [v["version"] for v in self._store.get(name, [])]
    
    def list_prompts(self) -> List[str]:
        """List all prompt names."""
        return list(self._store.keys())
    
    def diff(self, name: str, v1: str, v2: str) -> str:
        """
        Return a unified diff between two versions' templates.
        
        Args:
            name: Prompt name
            v1: First version
            v2: Second version
            
        Returns:
            Unified diff string
        """
        import difflib
        
        p1 = self.get(name, version=v1)
        p2 = self.get(name, version=v2)
        
        if not p1 or not p2:
            return "One or both versions not found."
        
        return "\n".join(difflib.unified_diff(
            p1.template.splitlines(),
            p2.template.splitlines(),
            fromfile=f"{name}@{v1}",
            tofile=f"{name}@{v2}",
            lineterm="",
        ))
    
    def rollback(
        self,
        name: str,
        to_version: str,
        label: str = "production"
    ) -> PromptVersion:
        """
        Move a label to point to an older version.
        
        Args:
            name: Prompt name
            to_version: Version to roll back to
            label: Label to move
            
        Returns:
            The target PromptVersion
            
        Raises:
            ValueError: If version not found
        """
        pv = self.get(name, version=to_version)
        if not pv:
            raise ValueError(f"Version {to_version} not found for prompt '{name}'")
        
        # Remove label from all versions
        for v in self._store.get(name, []):
            if label in v.get("labels", []):
                v["labels"].remove(label)
        
        # Add label to target version
        for v in self._store.get(name, []):
            if v["version"] == to_version:
                if "labels" not in v:
                    v["labels"] = []
                v["labels"].append(label)
                break
        
        self._save()
        return pv
    
    def delete(self, name: str, version: Optional[str] = None) -> bool:
        """
        Delete a prompt or specific version.
        
        Args:
            name: Prompt name
            version: Specific version to delete (None = delete all)
            
        Returns:
            True if deleted
        """
        if name not in self._store:
            return False
        
        if version:
            self._store[name] = [v for v in self._store[name] if v["version"] != version]
            if not self._store[name]:
                del self._store[name]
        else:
            del self._store[name]
        
        self._save()
        return True
    
    def _load(self):
        """Load from storage."""
        if self._path.exists():
            try:
                with open(self._path) as f:
                    self._store = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._store = {}
    
    def _save(self):
        """Save to storage."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._store, f, indent=2, default=str)
