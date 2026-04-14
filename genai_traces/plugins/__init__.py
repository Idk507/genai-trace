"""
Plugin system for GenAI-Traces.
"""

from .registry import PluginRegistry, get_plugin_registry
from .loader import load_plugins, discover_plugins

__all__ = [
    "PluginRegistry",
    "get_plugin_registry",
    "load_plugins",
    "discover_plugins",
]
