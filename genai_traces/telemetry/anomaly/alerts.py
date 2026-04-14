"""
Alert manager for dispatching anomaly alerts.

Supports: log, webhook (Slack/PagerDuty), custom callback.
"""

import json
import logging
import urllib.request
from typing import Callable, Dict, List, Optional

from .detector import AnomalyEvent


class AlertManager:
    """
    Dispatches anomaly alerts to configured channels.
    
    Supported channel types:
    - log: Log to Python logger
    - slack: Send to Slack webhook
    - webhook: Send to generic webhook
    - callback: Call a custom function
    
    Usage:
        alert_manager = AlertManager(channels=[
            {"type": "log"},
            {"type": "slack", "webhook_url": "https://hooks.slack.com/..."},
        ])
        alert_manager.send(anomaly_event)
    """
    
    def __init__(self, channels: Optional[List[Dict]] = None):
        """
        Initialize the alert manager.
        
        Args:
            channels: List of channel configurations
        """
        self.channels = channels or [{"type": "log"}]
        self._logger = logging.getLogger("genai_traces.anomaly")
    
    def send(self, event: AnomalyEvent) -> None:
        """
        Send an anomaly alert to all configured channels.
        
        Args:
            event: AnomalyEvent to send
        """
        for channel in self.channels:
            try:
                self._dispatch(channel, event)
            except Exception as e:
                self._logger.error(f"Failed to dispatch alert to {channel.get('type')}: {e}")
    
    def _dispatch(self, channel: Dict, event: AnomalyEvent) -> None:
        """Dispatch to a specific channel."""
        ctype = channel.get("type", "log")
        
        if ctype == "log":
            self._send_log(event)
        elif ctype == "slack":
            self._send_slack(channel, event)
        elif ctype == "webhook":
            self._send_webhook(channel, event)
        elif ctype == "callback":
            self._send_callback(channel, event)
    
    def _send_log(self, event: AnomalyEvent) -> None:
        """Log the anomaly."""
        level = {
            "critical": logging.CRITICAL,
            "high": logging.ERROR,
            "medium": logging.WARNING,
            "low": logging.INFO,
        }.get(event.severity, logging.WARNING)
        
        self._logger.log(level, str(event))
    
    def _send_slack(self, channel: Dict, event: AnomalyEvent) -> None:
        """Send to Slack webhook."""
        webhook_url = channel.get("webhook_url")
        if not webhook_url:
            return
        
        emoji = {
            "critical": ":rotating_light:",
            "high": ":warning:",
            "medium": ":large_yellow_circle:",
            "low": ":information_source:",
        }.get(event.severity, ":warning:")
        
        payload = {
            "text": (
                f"{emoji} *GenAI-Traces Anomaly* [{event.severity.upper()}]\n"
                f"Model: `{event.model}` | Metric: `{event.metric}`\n"
                f"Value: `{event.value:.4f}` | Baseline: `{event.baseline:.4f}` | Z: `{event.z_score:.2f}`"
            )
        }
        
        req = urllib.request.Request(
            webhook_url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
    
    def _send_webhook(self, channel: Dict, event: AnomalyEvent) -> None:
        """Send to generic webhook."""
        url = channel.get("url")
        if not url:
            return
        
        headers = {"Content-Type": "application/json"}
        headers.update(channel.get("headers", {}))
        
        req = urllib.request.Request(
            url,
            data=json.dumps({"event": event.to_dict()}).encode(),
            headers=headers,
        )
        urllib.request.urlopen(req, timeout=5)
    
    def _send_callback(self, channel: Dict, event: AnomalyEvent) -> None:
        """Call a custom callback function."""
        fn: Optional[Callable] = channel.get("fn")
        if fn:
            fn(event)
    
    def add_channel(self, channel: Dict) -> None:
        """Add a new alert channel."""
        self.channels.append(channel)
    
    def remove_channel(self, channel_type: str) -> None:
        """Remove channels of a specific type."""
        self.channels = [c for c in self.channels if c.get("type") != channel_type]
