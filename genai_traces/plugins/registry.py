"""
Global plugin registry for GenAI-Traces.
"""

from typing import Dict, List, Optional, Any, Type, Callable
from dataclasses import dataclass


@dataclass
class PluginInfo:
    """Information about a registered plugin."""
    
    name: str
    plugin_type: str
    version: str
    description: str
    instance: Any
    enabled: bool = True


class PluginRegistry:
    """
    Central registry for all GenAI-Traces plugins.
    
    Supports:
    - Exporters
    - Evaluators
    - Instrumentations
    - Custom processors
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._plugins: Dict[str, Dict[str, PluginInfo]] = {
                "exporter": {},
                "evaluator": {},
                "instrumentation": {},
                "processor": {},
            }
            cls._instance._hooks: Dict[str, List[Callable]] = {}
        return cls._instance
    
    def register(
        self,
        name: str,
        plugin_type: str,
        instance: Any,
        version: str = "1.0.0",
        description: str = "",
    ) -> None:
        """
        Register a plugin.
        
        Args:
            name: Unique plugin name
            plugin_type: Type (exporter, evaluator, instrumentation, processor)
            instance: Plugin instance
            version: Plugin version
            description: Plugin description
        """
        if plugin_type not in self._plugins:
            self._plugins[plugin_type] = {}
        
        self._plugins[plugin_type][name] = PluginInfo(
            name=name,
            plugin_type=plugin_type,
            version=version,
            description=description,
            instance=instance,
            enabled=True,
        )
    
    def unregister(self, name: str, plugin_type: str) -> bool:
        """
        Unregister a plugin.
        
        Args:
            name: Plugin name
            plugin_type: Plugin type
            
        Returns:
            True if plugin was found and removed
        """
        if plugin_type in self._plugins and name in self._plugins[plugin_type]:
            del self._plugins[plugin_type][name]
            return True
        return False
    
    def get(self, name: str, plugin_type: str) -> Optional[Any]:
        """
        Get a plugin instance.
        
        Args:
            name: Plugin name
            plugin_type: Plugin type
            
        Returns:
            Plugin instance or None
        """
        if plugin_type in self._plugins and name in self._plugins[plugin_type]:
            info = self._plugins[plugin_type][name]
            if info.enabled:
                return info.instance
        return None
    
    def get_all(self, plugin_type: str) -> List[Any]:
        """
        Get all enabled plugins of a type.
        
        Args:
            plugin_type: Plugin type
            
        Returns:
            List of plugin instances
        """
        if plugin_type not in self._plugins:
            return []
        
        return [
            info.instance
            for info in self._plugins[plugin_type].values()
            if info.enabled
        ]
    
    def list_plugins(self, plugin_type: Optional[str] = None) -> List[PluginInfo]:
        """
        List registered plugins.
        
        Args:
            plugin_type: Optional filter by type
            
        Returns:
            List of PluginInfo
        """
        if plugin_type:
            return list(self._plugins.get(plugin_type, {}).values())
        
        all_plugins = []
        for plugins in self._plugins.values():
            all_plugins.extend(plugins.values())
        return all_plugins
    
    def enable(self, name: str, plugin_type: str) -> bool:
        """Enable a plugin."""
        if plugin_type in self._plugins and name in self._plugins[plugin_type]:
            self._plugins[plugin_type][name].enabled = True
            return True
        return False
    
    def disable(self, name: str, plugin_type: str) -> bool:
        """Disable a plugin."""
        if plugin_type in self._plugins and name in self._plugins[plugin_type]:
            self._plugins[plugin_type][name].enabled = False
            return True
        return False
    
    def register_hook(self, hook_name: str, callback: Callable) -> None:
        """
        Register a hook callback.
        
        Args:
            hook_name: Name of the hook
            callback: Callback function
        """
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
        self._hooks[hook_name].append(callback)
    
    def trigger_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """
        Trigger a hook and collect results.
        
        Args:
            hook_name: Name of the hook
            *args, **kwargs: Arguments to pass to callbacks
            
        Returns:
            List of callback results
        """
        results = []
        for callback in self._hooks.get(hook_name, []):
            try:
                result = callback(*args, **kwargs)
                results.append(result)
            except Exception:
                pass
        return results


_registry = PluginRegistry()


def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    return _registry


def register_exporter(name: str, exporter: Any, **kwargs) -> None:
    """Convenience function to register an exporter."""
    _registry.register(name, "exporter", exporter, **kwargs)


def register_evaluator(name: str, evaluator: Any, **kwargs) -> None:
    """Convenience function to register an evaluator."""
    _registry.register(name, "evaluator", evaluator, **kwargs)
