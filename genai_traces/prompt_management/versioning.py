"""
Prompt versioning for GenAI-Traces.

Provides PromptVersion dataclass, diff, and changelog functionality.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import difflib
import hashlib


@dataclass
class PromptVersion:
    """
    Represents a version of a prompt template.
    
    Attributes:
        version: Version string (e.g., "1.0.0")
        template: The prompt template text
        variables: List of variable names in the template
        created_at: When this version was created
        author: Who created this version
        description: Description of changes
        parent_version: Previous version this was derived from
        metadata: Additional metadata
    """
    version: str
    template: str
    variables: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    author: Optional[str] = None
    description: str = ""
    parent_version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def hash(self) -> str:
        """Get hash of the template content."""
        return hashlib.sha256(self.template.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "template": self.template,
            "variables": self.variables,
            "created_at": self.created_at.isoformat(),
            "author": self.author,
            "description": self.description,
            "parent_version": self.parent_version,
            "metadata": self.metadata,
            "hash": self.hash,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptVersion":
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        data.pop("hash", None)
        return cls(**data)


def diff_prompts(
    old_version: PromptVersion,
    new_version: PromptVersion,
    context_lines: int = 3,
) -> str:
    """
    Generate a diff between two prompt versions.
    
    Args:
        old_version: The older version
        new_version: The newer version
        context_lines: Number of context lines to show
        
    Returns:
        Unified diff string
    """
    old_lines = old_version.template.splitlines(keepends=True)
    new_lines = new_version.template.splitlines(keepends=True)
    
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"v{old_version.version}",
        tofile=f"v{new_version.version}",
        lineterm="",
        n=context_lines,
    )
    
    return "".join(diff)


def diff_prompts_html(
    old_version: PromptVersion,
    new_version: PromptVersion,
) -> str:
    """
    Generate an HTML diff between two prompt versions.
    
    Args:
        old_version: The older version
        new_version: The newer version
        
    Returns:
        HTML diff string
    """
    old_lines = old_version.template.splitlines()
    new_lines = new_version.template.splitlines()
    
    differ = difflib.HtmlDiff()
    return differ.make_table(
        old_lines,
        new_lines,
        fromdesc=f"v{old_version.version}",
        todesc=f"v{new_version.version}",
    )


@dataclass
class ChangelogEntry:
    """A single changelog entry."""
    version: str
    date: datetime
    author: Optional[str]
    description: str
    changes: List[str] = field(default_factory=list)


class PromptChangelog:
    """
    Maintains a changelog for a prompt.
    
    Usage:
        changelog = PromptChangelog("my_prompt")
        changelog.add_entry(
            version="1.1.0",
            description="Added context variable",
            changes=["Added {context} variable", "Improved formatting"],
        )
    """
    
    def __init__(self, prompt_name: str):
        self.prompt_name = prompt_name
        self.entries: List[ChangelogEntry] = []
    
    def add_entry(
        self,
        version: str,
        description: str,
        changes: Optional[List[str]] = None,
        author: Optional[str] = None,
        date: Optional[datetime] = None,
    ) -> None:
        """Add a changelog entry."""
        entry = ChangelogEntry(
            version=version,
            date=date or datetime.utcnow(),
            author=author,
            description=description,
            changes=changes or [],
        )
        self.entries.append(entry)
    
    def add_from_diff(
        self,
        old_version: PromptVersion,
        new_version: PromptVersion,
    ) -> None:
        """Add a changelog entry from a diff."""
        changes = []
        
        old_vars = set(old_version.variables)
        new_vars = set(new_version.variables)
        
        added_vars = new_vars - old_vars
        removed_vars = old_vars - new_vars
        
        for var in added_vars:
            changes.append(f"Added variable: {{{var}}}")
        for var in removed_vars:
            changes.append(f"Removed variable: {{{var}}}")
        
        old_len = len(old_version.template)
        new_len = len(new_version.template)
        if new_len > old_len:
            changes.append(f"Template expanded by {new_len - old_len} characters")
        elif new_len < old_len:
            changes.append(f"Template reduced by {old_len - new_len} characters")
        
        self.add_entry(
            version=new_version.version,
            description=new_version.description or "Updated prompt",
            changes=changes,
            author=new_version.author,
            date=new_version.created_at,
        )
    
    def to_markdown(self) -> str:
        """Generate markdown changelog."""
        lines = [f"# Changelog for {self.prompt_name}", ""]
        
        for entry in reversed(self.entries):
            lines.append(f"## [{entry.version}] - {entry.date.strftime('%Y-%m-%d')}")
            if entry.author:
                lines.append(f"*Author: {entry.author}*")
            lines.append("")
            lines.append(entry.description)
            lines.append("")
            
            if entry.changes:
                for change in entry.changes:
                    lines.append(f"- {change}")
                lines.append("")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt_name": self.prompt_name,
            "entries": [
                {
                    "version": e.version,
                    "date": e.date.isoformat(),
                    "author": e.author,
                    "description": e.description,
                    "changes": e.changes,
                }
                for e in self.entries
            ],
        }


def parse_version(version: str) -> tuple:
    """Parse a version string into comparable tuple."""
    parts = version.lstrip("v").split(".")
    return tuple(int(p) if p.isdigit() else p for p in parts)


def compare_versions(v1: str, v2: str) -> int:
    """
    Compare two version strings.
    
    Returns:
        -1 if v1 < v2, 0 if equal, 1 if v1 > v2
    """
    p1 = parse_version(v1)
    p2 = parse_version(v2)
    
    if p1 < p2:
        return -1
    elif p1 > p2:
        return 1
    else:
        return 0


def increment_version(version: str, part: str = "patch") -> str:
    """
    Increment a semantic version.
    
    Args:
        version: Current version (e.g., "1.2.3")
        part: Which part to increment ("major", "minor", "patch")
        
    Returns:
        New version string
    """
    parts = version.lstrip("v").split(".")
    
    while len(parts) < 3:
        parts.append("0")
    
    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
    
    if part == "major":
        major += 1
        minor = 0
        patch = 0
    elif part == "minor":
        minor += 1
        patch = 0
    else:
        patch += 1
    
    return f"{major}.{minor}.{patch}"
