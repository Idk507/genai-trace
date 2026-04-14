"""
Data retention policy management.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import json


@dataclass
class RetentionPolicy:
    """Data retention policy configuration."""
    
    default_retention_days: int = 90
    pii_retention_days: int = 30
    error_retention_days: int = 180
    feedback_retention_days: int = 365
    
    auto_delete: bool = True
    archive_before_delete: bool = True
    archive_path: str = "./archives"
    
    def get_retention_days(self, data_type: str) -> int:
        """Get retention days for a data type."""
        retention_map = {
            "pii": self.pii_retention_days,
            "error": self.error_retention_days,
            "feedback": self.feedback_retention_days,
            "default": self.default_retention_days,
        }
        return retention_map.get(data_type, self.default_retention_days)
    
    def is_expired(self, created_at: datetime, data_type: str = "default") -> bool:
        """Check if data has expired based on retention policy."""
        retention_days = self.get_retention_days(data_type)
        expiry = created_at + timedelta(days=retention_days)
        return datetime.utcnow() > expiry
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "default_retention_days": self.default_retention_days,
            "pii_retention_days": self.pii_retention_days,
            "error_retention_days": self.error_retention_days,
            "feedback_retention_days": self.feedback_retention_days,
            "auto_delete": self.auto_delete,
            "archive_before_delete": self.archive_before_delete,
        }


def apply_retention(
    data_path: str,
    policy: RetentionPolicy,
    dry_run: bool = False,
) -> Dict[str, int]:
    """
    Apply retention policy to trace data.
    
    Args:
        data_path: Path to trace data (JSONL file or directory)
        policy: Retention policy to apply
        dry_run: If True, only report what would be deleted
        
    Returns:
        Dict with counts of processed, archived, deleted records
    """
    stats = {
        "processed": 0,
        "expired": 0,
        "archived": 0,
        "deleted": 0,
        "retained": 0,
    }
    
    path = Path(data_path)
    
    if path.is_file():
        stats = _process_jsonl_file(path, policy, dry_run)
    elif path.is_dir():
        for file in path.glob("*.jsonl"):
            file_stats = _process_jsonl_file(file, policy, dry_run)
            for key in stats:
                stats[key] += file_stats[key]
    
    return stats


def _process_jsonl_file(
    file_path: Path,
    policy: RetentionPolicy,
    dry_run: bool,
) -> Dict[str, int]:
    """Process a single JSONL file for retention."""
    stats = {
        "processed": 0,
        "expired": 0,
        "archived": 0,
        "deleted": 0,
        "retained": 0,
    }
    
    retained_records = []
    expired_records = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            
            stats["processed"] += 1
            
            try:
                record = json.loads(line)
                
                created_str = record.get("start_time") or record.get("created_at")
                if created_str:
                    created_at = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                else:
                    retained_records.append(line)
                    stats["retained"] += 1
                    continue
                
                data_type = "default"
                if record.get("attributes", {}).get("privacy.pii_detected"):
                    data_type = "pii"
                elif record.get("status") == "error":
                    data_type = "error"
                
                if policy.is_expired(created_at, data_type):
                    stats["expired"] += 1
                    expired_records.append(line)
                else:
                    retained_records.append(line)
                    stats["retained"] += 1
                    
            except (json.JSONDecodeError, ValueError):
                retained_records.append(line)
                stats["retained"] += 1
    
    if dry_run:
        return stats
    
    if expired_records and policy.archive_before_delete:
        archive_path = Path(policy.archive_path)
        archive_path.mkdir(parents=True, exist_ok=True)
        
        archive_file = archive_path / f"archived_{file_path.name}"
        with open(archive_file, 'a') as f:
            for record in expired_records:
                f.write(record)
                stats["archived"] += 1
    
    if policy.auto_delete and expired_records:
        with open(file_path, 'w') as f:
            for record in retained_records:
                f.write(record)
        stats["deleted"] = stats["expired"]
    
    return stats


def schedule_retention_cleanup(
    data_path: str,
    policy: RetentionPolicy,
    interval_hours: int = 24,
) -> None:
    """
    Schedule periodic retention cleanup.
    
    Note: This is a simple implementation. For production,
    use a proper scheduler like APScheduler or Celery.
    """
    import threading
    
    def cleanup_task():
        while True:
            try:
                apply_retention(data_path, policy)
            except Exception:
                pass
            
            import time
            time.sleep(interval_hours * 3600)
    
    thread = threading.Thread(target=cleanup_task, daemon=True)
    thread.start()
