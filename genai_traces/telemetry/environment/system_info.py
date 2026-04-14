"""
System information collection for GenAI-Traces.

Captures OS, Python version, GPU info, and other environment details.
"""

import os
import sys
import platform
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class GPUInfo:
    """GPU information."""
    name: str
    memory_total_mb: int
    memory_free_mb: int
    driver_version: str
    cuda_version: Optional[str] = None


@dataclass
class SystemInfo:
    """System environment information."""
    
    os_name: str
    os_version: str
    os_release: str
    
    python_version: str
    python_implementation: str
    
    hostname: str
    cpu_count: int
    memory_total_gb: float
    
    architecture: str
    machine: str
    
    gpus: List[GPUInfo] = field(default_factory=list)
    
    environment_variables: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "os": {
                "name": self.os_name,
                "version": self.os_version,
                "release": self.os_release,
            },
            "python": {
                "version": self.python_version,
                "implementation": self.python_implementation,
            },
            "hostname": self.hostname,
            "cpu_count": self.cpu_count,
            "memory_total_gb": round(self.memory_total_gb, 2),
            "architecture": self.architecture,
            "machine": self.machine,
            "gpus": [
                {
                    "name": gpu.name,
                    "memory_total_mb": gpu.memory_total_mb,
                    "memory_free_mb": gpu.memory_free_mb,
                    "driver_version": gpu.driver_version,
                    "cuda_version": gpu.cuda_version,
                }
                for gpu in self.gpus
            ],
        }


def get_system_info(
    include_env_vars: bool = False,
    env_var_prefix: str = "GENAI_",
) -> SystemInfo:
    """
    Collect system information.
    
    Args:
        include_env_vars: Whether to include environment variables
        env_var_prefix: Prefix filter for environment variables
        
    Returns:
        SystemInfo object with collected data
    """
    memory_gb = 0.0
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass
    
    gpus = _get_gpu_info()
    
    env_vars = {}
    if include_env_vars:
        env_vars = {
            k: v for k, v in os.environ.items()
            if k.startswith(env_var_prefix)
        }
    
    return SystemInfo(
        os_name=platform.system(),
        os_version=platform.version(),
        os_release=platform.release(),
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        hostname=platform.node(),
        cpu_count=os.cpu_count() or 1,
        memory_total_gb=memory_gb,
        architecture=platform.architecture()[0],
        machine=platform.machine(),
        gpus=gpus,
        environment_variables=env_vars,
    )


def _get_gpu_info() -> List[GPUInfo]:
    """Get GPU information if available."""
    gpus = []
    
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                free_mem, total_mem = torch.cuda.mem_get_info(i)
                
                gpus.append(GPUInfo(
                    name=props.name,
                    memory_total_mb=total_mem // (1024 * 1024),
                    memory_free_mb=free_mem // (1024 * 1024),
                    driver_version=torch.version.cuda or "unknown",
                    cuda_version=torch.version.cuda,
                ))
    except ImportError:
        pass
    except Exception:
        pass
    
    if not gpus:
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4:
                        gpus.append(GPUInfo(
                            name=parts[0],
                            memory_total_mb=int(parts[1]),
                            memory_free_mb=int(parts[2]),
                            driver_version=parts[3],
                        ))
        except Exception:
            pass
    
    return gpus


_cached_info: Optional[SystemInfo] = None


def get_cached_system_info() -> SystemInfo:
    """Get cached system info (computed once)."""
    global _cached_info
    if _cached_info is None:
        _cached_info = get_system_info()
    return _cached_info
