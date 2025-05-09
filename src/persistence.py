"""
Persistence module for saving and loading optimized settings.

This module provides functionality to persist learned and optimized settings
across application restarts, improving performance over time.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


class SettingsPersistence:
    """
    Manager for persisting optimized client settings.

    This class handles saving and loading settings like:
    - Adaptive timeout configuration per host
    - Optimal transport selection per host
    - Connection pooling parameters
    """

    def __init__(self, storage_path: Optional[str] = None, auto_save_interval: int = 3600):
        """
        Initialize the settings persistence manager.

        Args:
            storage_path: Path to store the settings file, defaults to ~/.enhanced_httpx/
            auto_save_interval: Interval in seconds for auto-saving settings (default: 1 hour)
        """
        if storage_path is None:
            # Default to ~/.enhanced_httpx directory
            self.storage_dir = Path.home() / ".enhanced_httpx"
        else:
            self.storage_dir = Path(storage_path)

        # Create directory if it doesn't exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.settings_file = self.storage_dir / "optimized_settings.json"
        self.timeout_file = self.storage_dir / "adaptive_timeouts.json"
        self.transport_file = self.storage_dir / "transport_metrics.json"

        # Track when settings were last saved
        self.last_save_time = 0
        self.auto_save_interval = auto_save_interval

        # Cache for settings
        self._timeout_settings = {}
        self._transport_settings = {}
        self._connection_settings = {}

    def load_timeout_settings(self) -> Dict[str, Any]:
        """
        Load saved adaptive timeout settings.

        Returns:
            Dictionary of host-specific timeout settings
        """
        if self._timeout_settings:
            return self._timeout_settings

        if self.timeout_file.exists():
            try:
                with open(self.timeout_file, "r") as f:
                    settings = json.load(f)
                    self._timeout_settings = settings
                    return settings
            except (json.JSONDecodeError, IOError):
                # If file is corrupted or can't be read, return empty dict
                return {}
        return {}

    def save_timeout_settings(self, settings: Dict[str, Any]) -> None:
        """
        Save adaptive timeout settings.

        Args:
            settings: Dictionary of host-specific timeout settings
        """
        self._timeout_settings = settings
        self._maybe_auto_save()

        try:
            with open(self.timeout_file, "w") as f:
                json.dump(settings, f)
        except IOError:
            # Log but don't fail if we can't save
            pass

    def load_transport_preferences(self) -> Dict[str, str]:
        """
        Load saved transport preferences for hosts.

        Returns:
            Dictionary mapping hosts to preferred transport
        """
        if self._transport_settings:
            return self._transport_settings

        if self.transport_file.exists():
            try:
                with open(self.transport_file, "r") as f:
                    settings = json.load(f)
                    self._transport_settings = settings
                    return settings
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def save_transport_preferences(self, preferences: Dict[str, str]) -> None:
        """
        Save transport preferences for hosts.

        Args:
            preferences: Dictionary mapping hosts to preferred transport
        """
        self._transport_settings = preferences
        self._maybe_auto_save()

        try:
            with open(self.transport_file, "w") as f:
                json.dump(preferences, f)
        except IOError:
            pass

    def load_connection_settings(self) -> Dict[str, Any]:
        """
        Load saved connection pool settings.

        Returns:
            Dictionary with optimized connection pool settings
        """
        if self._connection_settings:
            return self._connection_settings

        if self.settings_file.exists():
            try:
                with open(self.settings_file, "r") as f:
                    settings = json.load(f)
                    self._connection_settings = settings
                    return settings
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def save_connection_settings(self, settings: Dict[str, Any]) -> None:
        """
        Save optimized connection pool settings.

        Args:
            settings: Dictionary with connection pool settings
        """
        self._connection_settings = settings
        self._maybe_auto_save()

        try:
            with open(self.settings_file, "w") as f:
                json.dump(settings, f)
        except IOError:
            pass

    def _maybe_auto_save(self) -> None:
        """Check if it's time to auto-save settings."""
        current_time = time.time()
        if current_time - self.last_save_time > self.auto_save_interval:
            self.save_all()
            self.last_save_time = current_time

    def save_all(self) -> None:
        """Save all settings to disk."""
        if self._timeout_settings:
            self.save_timeout_settings(self._timeout_settings)

        if self._transport_settings:
            self.save_transport_preferences(self._transport_settings)

        if self._connection_settings:
            self.save_connection_settings(self._connection_settings)
