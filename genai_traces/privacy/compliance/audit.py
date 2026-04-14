"""
Immutable audit log for trace access.
"""

import json
import hashlib
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import threading


@dataclass
class AuditEntry:
    """An audit log entry."""
    
    timestamp: str
    action: str
    user_id: str
    resource_type: str
    resource_id: str
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    previous_hash: Optional[str] = None
    entry_hash: Optional[str] = None
    
    def compute_hash(self) -> str:
        """Compute hash of this entry."""
        data = {
            "timestamp": self.timestamp,
            "action": self.action,
            "user_id": self.user_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "details": self.details,
            "previous_hash": self.previous_hash,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AuditLog:
    """
    Immutable audit log with hash chain for integrity verification.
    """
    
    ACTIONS = {
        "read": "Trace data was read",
        "export": "Trace data was exported",
        "delete": "Trace data was deleted",
        "modify": "Trace data was modified",
        "search": "Trace data was searched",
        "decrypt": "Encrypted field was decrypted",
    }
    
    def __init__(self, log_path: str = "./audit_log.jsonl"):
        """
        Initialize the audit log.
        
        Args:
            log_path: Path to the audit log file
        """
        self._path = Path(log_path)
        self._lock = threading.Lock()
        self._last_hash: Optional[str] = None
        
        self._load_last_hash()
    
    def log(
        self,
        action: str,
        user_id: str,
        resource_type: str,
        resource_id: str,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> AuditEntry:
        """
        Log an audit entry.
        
        Args:
            action: Action performed (read, export, delete, etc.)
            user_id: User who performed the action
            resource_type: Type of resource (trace, span, feedback)
            resource_id: ID of the resource
            details: Additional details
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            The created audit entry
        """
        with self._lock:
            entry = AuditEntry(
                timestamp=datetime.utcnow().isoformat(),
                action=action,
                user_id=user_id,
                resource_type=resource_type,
                resource_id=resource_id,
                details=details or {},
                ip_address=ip_address,
                user_agent=user_agent,
                previous_hash=self._last_hash,
            )
            
            entry.entry_hash = entry.compute_hash()
            self._last_hash = entry.entry_hash
            
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, 'a') as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
            
            return entry
    
    def verify_integrity(self) -> Dict[str, Any]:
        """
        Verify the integrity of the audit log.
        
        Returns:
            Dict with verification results
        """
        if not self._path.exists():
            return {"valid": True, "entries": 0, "errors": []}
        
        errors = []
        entries = 0
        previous_hash = None
        
        with open(self._path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    entry = AuditEntry(**data)
                    entries += 1
                    
                    if entry.previous_hash != previous_hash:
                        errors.append(f"Line {line_num}: Chain broken - previous hash mismatch")
                    
                    computed_hash = entry.compute_hash()
                    if computed_hash != entry.entry_hash:
                        errors.append(f"Line {line_num}: Entry hash mismatch - possible tampering")
                    
                    previous_hash = entry.entry_hash
                    
                except Exception as e:
                    errors.append(f"Line {line_num}: Parse error - {e}")
        
        return {
            "valid": len(errors) == 0,
            "entries": entries,
            "errors": errors,
        }
    
    def search(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditEntry]:
        """
        Search audit log entries.
        
        Args:
            user_id: Filter by user
            action: Filter by action
            resource_type: Filter by resource type
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum entries to return
            
        Returns:
            List of matching audit entries
        """
        if not self._path.exists():
            return []
        
        results = []
        
        with open(self._path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    entry = AuditEntry(**data)
                    
                    if user_id and entry.user_id != user_id:
                        continue
                    if action and entry.action != action:
                        continue
                    if resource_type and entry.resource_type != resource_type:
                        continue
                    
                    if start_time or end_time:
                        entry_time = datetime.fromisoformat(entry.timestamp)
                        if start_time and entry_time < start_time:
                            continue
                        if end_time and entry_time > end_time:
                            continue
                    
                    results.append(entry)
                    
                    if len(results) >= limit:
                        break
                        
                except Exception:
                    continue
        
        return results
    
    def _load_last_hash(self) -> None:
        """Load the last hash from the log file."""
        if not self._path.exists():
            return
        
        with open(self._path, 'r') as f:
            last_line = None
            for line in f:
                if line.strip():
                    last_line = line
            
            if last_line:
                try:
                    data = json.loads(last_line)
                    self._last_hash = data.get("entry_hash")
                except Exception:
                    pass


_audit_log: Optional[AuditLog] = None


def get_audit_log(path: str = "./audit_log.jsonl") -> AuditLog:
    """Get the global audit log instance."""
    global _audit_log
    if _audit_log is None:
        _audit_log = AuditLog(path)
    return _audit_log


def log_access(
    action: str,
    user_id: str,
    resource_type: str,
    resource_id: str,
    **kwargs
) -> AuditEntry:
    """Convenience function to log an access."""
    return get_audit_log().log(action, user_id, resource_type, resource_id, **kwargs)
