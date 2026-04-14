"""
Dynamic plugin discovery and loading.
"""

import importlib
import importlib.util
import os
from pathlib import Path
from typing import List, Optional, Any, Dict

from .registry import get_plugin_registry, PluginInfo


def load_plugins(
    plugin_paths: List[str],
    auto_register: bool = True,
) -> List[PluginInfo]:
    """
    Load plugins from specified paths.
    
    Args:
        plugin_paths: List of paths to plugin files or directories
        auto_register: Whether to auto-register discovered plugins
        
    Returns:
        List of loaded PluginInfo
    """
    loaded = []
    
    for path in plugin_paths:
        path = Path(path)
        
        if path.is_file() and path.suffix == ".py":
            info = _load_plugin_file(path, auto_register)
            if info:
                loaded.append(info)
        
        elif path.is_dir():
            for file in path.glob("*.py"):
                if not file.name.startswith("_"):
                    info = _load_plugin_file(file, auto_register)
                    if info:
                        loaded.append(info)
    
    return loaded


def discover_plugins(
    search_paths: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Discover available plugins without loading them.
    
    Args:
        search_paths: Paths to search (defaults to common locations)
        
    Returns:
        List of plugin metadata dicts
    """
    if search_paths is None:
        search_paths = [
            "./plugins",
            os.path.expanduser("~/.genai_traces/plugins"),
        ]
    
    discovered = []
    
    for search_path in search_paths:
        path = Path(search_path)
        if not path.exists():
            continue
        
        for file in path.glob("*.py"):
            if file.name.startswith("_"):
                continue
            
            metadata = _extract_plugin_metadata(file)
            if metadata:
                discovered.append(metadata)
    
    return discovered


def _load_plugin_file(
    path: Path,
    auto_register: bool = True,
) -> Optional[PluginInfo]:
    """Load a single plugin file."""
    try:
        spec = importlib.util.spec_from_file_location(
            f"genai_traces_plugin_{path.stem}",
            path
        )
        if spec is None or spec.loader is None:
            return None
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        plugin_name = getattr(module, "PLUGIN_NAME", path.stem)
        plugin_type = getattr(module, "PLUGIN_TYPE", "processor")
        plugin_version = getattr(module, "PLUGIN_VERSION", "1.0.0")
        plugin_description = getattr(module, "PLUGIN_DESCRIPTION", "")
        
        plugin_class = getattr(module, "Plugin", None)
        if plugin_class is None:
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and attr_name.endswith("Plugin"):
                    plugin_class = attr
                    break
        
        if plugin_class is None:
            return None
        
        instance = plugin_class()
        
        info = PluginInfo(
            name=plugin_name,
            plugin_type=plugin_type,
            version=plugin_version,
            description=plugin_description,
            instance=instance,
        )
        
        if auto_register:
            registry = get_plugin_registry()
            registry.register(
                name=plugin_name,
                plugin_type=plugin_type,
                instance=instance,
                version=plugin_version,
                description=plugin_description,
            )
        
        return info
        
    except Exception as e:
        print(f"Failed to load plugin {path}: {e}")
        return None


def _extract_plugin_metadata(path: Path) -> Optional[Dict[str, Any]]:
    """Extract metadata from a plugin file without fully loading it."""
    try:
        content = path.read_text()
        
        metadata = {
            "path": str(path),
            "name": path.stem,
            "type": "processor",
            "version": "1.0.0",
            "description": "",
        }
        
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("PLUGIN_NAME"):
                metadata["name"] = _extract_string_value(line)
            elif line.startswith("PLUGIN_TYPE"):
                metadata["type"] = _extract_string_value(line)
            elif line.startswith("PLUGIN_VERSION"):
                metadata["version"] = _extract_string_value(line)
            elif line.startswith("PLUGIN_DESCRIPTION"):
                metadata["description"] = _extract_string_value(line)
        
        return metadata
        
    except Exception:
        return None


def _extract_string_value(line: str) -> str:
    """Extract string value from an assignment line."""
    if "=" in line:
        value = line.split("=", 1)[1].strip()
        value = value.strip("\"'")
        return value
    return ""


def load_builtin_plugins() -> None:
    """Load built-in plugins."""
    pass
