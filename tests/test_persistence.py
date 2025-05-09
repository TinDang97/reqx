"""
Tests for the persistence functionality.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.persistence import SettingsPersistence
from src.client import AdaptiveTimeoutManager, ReqxClient
from src.builder import ReqxClientBuilder


class TestSettingsPersistence:
    """Tests for the SettingsPersistence class."""

    def test_initialization(self):
        """Test that the persistence manager initializes correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            persistence = SettingsPersistence(storage_path=temp_dir)

            assert persistence.storage_dir == Path(temp_dir)
            assert persistence.settings_file == Path(temp_dir) / "optimized_settings.json"
            assert persistence.timeout_file == Path(temp_dir) / "adaptive_timeouts.json"
            assert persistence.transport_file == Path(temp_dir) / "transport_metrics.json"

    def test_save_load_timeout_settings(self):
        """Test saving and loading timeout settings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            persistence = SettingsPersistence(storage_path=temp_dir)

            # Sample timeout settings
            settings = {
                "example.com": {
                    "durations": [0.2, 0.3, 0.25],
                    "timeout_history": [5.0, 5.5],
                    "successes": 10,
                    "failures": 2,
                    "timeouts": 1,
                    "current_timeout": 5.5,
                },
                "api.example.org": {
                    "durations": [0.5, 0.6],
                    "timeout_history": [10.0],
                    "successes": 5,
                    "failures": 1,
                    "timeouts": 0,
                    "current_timeout": 10.0,
                },
            }

            # Save settings
            persistence.save_timeout_settings(settings)

            # Check that the file was created
            assert os.path.exists(persistence.timeout_file)

            # Load settings
            loaded_settings = persistence.load_timeout_settings()

            # Check that settings match
            assert loaded_settings == settings

    def test_save_load_transport_preferences(self):
        """Test saving and loading transport preferences."""
        with tempfile.TemporaryDirectory() as temp_dir:
            persistence = SettingsPersistence(storage_path=temp_dir)

            # Sample transport preferences
            preferences = {
                "example.com": "httpx",
                "api.google.com": "httpx",
                "api.example.org": "aiohttp",
            }

            # Save preferences
            persistence.save_transport_preferences(preferences)

            # Check that the file was created
            assert os.path.exists(persistence.transport_file)

            # Load preferences
            loaded_preferences = persistence.load_transport_preferences()

            # Check that preferences match
            assert loaded_preferences == preferences

    def test_save_load_connection_settings(self):
        """Test saving and loading connection pool settings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            persistence = SettingsPersistence(storage_path=temp_dir)

            # Sample connection settings
            settings = {
                "max_connections": 150,
                "max_keepalive_connections": 40,
                "keepalive_expiry": 90,
            }

            # Save settings
            persistence.save_connection_settings(settings)

            # Check that the file was created
            assert os.path.exists(persistence.settings_file)

            # Load settings
            loaded_settings = persistence.load_connection_settings()

            # Check that settings match
            assert loaded_settings == settings

    def test_auto_save(self):
        """Test that settings are auto-saved at intervals."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use a short auto-save interval for testing
            persistence = SettingsPersistence(storage_path=temp_dir, auto_save_interval=0)

            # Initial settings
            timeout_settings = {"example.com": {"current_timeout": 5.0}}
            transport_settings = {"example.com": "httpx"}
            connection_settings = {"max_connections": 100}

            # Save settings to trigger auto-save
            persistence.save_timeout_settings(timeout_settings)

            # Check that files were created
            assert os.path.exists(persistence.timeout_file)

            # Modify settings
            persistence._timeout_settings = {"example.com": {"current_timeout": 6.0}}
            persistence._transport_settings = transport_settings
            persistence._connection_settings = connection_settings

            # Trigger auto-save
            persistence._maybe_auto_save()

            # Load settings back and check they were updated
            with open(persistence.timeout_file, "r") as f:
                loaded = json.load(f)
                assert loaded["example.com"]["current_timeout"] == 6.0

    def test_handle_corrupted_file(self):
        """Test handling of corrupted settings files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            persistence = SettingsPersistence(storage_path=temp_dir)

            # Create a corrupted file
            with open(persistence.timeout_file, "w") as f:
                f.write("{not valid json")

            # Try to load settings - should return empty dict instead of raising error
            loaded_settings = persistence.load_timeout_settings()
            assert loaded_settings == {}


class TestAdaptiveTimeoutManagerPersistence:
    """Tests for the persistence functionality in AdaptiveTimeoutManager."""

    def test_timeout_manager_with_persistence(self):
        """Test that the timeout manager loads and saves settings when persistence is enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a manager with persistence enabled
            manager1 = AdaptiveTimeoutManager(
                default_timeout=10.0, persistence_enabled=True, persistence_path=temp_dir
            )

            # Add some data
            manager1.record_request("example.com", 0.3, success=True)
            manager1.record_request("example.com", 0.4, success=True)
            manager1.record_timeout("example.com")
            manager1.record_request("api.example.org", 0.5, success=True)

            # Explicitly save settings
            manager1.save_settings()

            # Create a new manager to test loading
            manager2 = AdaptiveTimeoutManager(
                default_timeout=10.0, persistence_enabled=True, persistence_path=temp_dir
            )

            # Check that settings were loaded
            assert "example.com" in manager2.host_metrics
            assert manager2.host_metrics["example.com"]["successes"] == 2
            assert manager2.host_metrics["example.com"]["timeouts"] == 1
            assert "api.example.org" in manager2.host_metrics

    def test_timeout_auto_save(self):
        """Test that timeout settings are auto-saved after certain operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a persistence object to check the file directly
            persistence = SettingsPersistence(storage_path=temp_dir)

            # Create a manager with persistence
            manager = AdaptiveTimeoutManager(
                default_timeout=10.0, persistence_enabled=True, persistence_path=temp_dir
            )

            # Record enough requests to trigger auto-save (20 is the threshold in our implementation)
            for i in range(25):
                manager.record_request("example.com", 0.3, success=True)

            # Check that the file exists and contains our data
            assert os.path.exists(persistence.timeout_file)

            # Load the file directly to verify contents
            with open(persistence.timeout_file, "r") as f:
                saved_data = json.load(f)

            assert "example.com" in saved_data
            assert saved_data["example.com"]["successes"] >= 20


class TestClientPersistence:
    """Tests for the persistence functionality in ReqxClient."""

    def test_client_with_persistence(self):
        """Test creating a client with persistence enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ReqxClient(
                adaptive_timeout=True, persistence_enabled=True, persistence_path=temp_dir
            )

            assert client.adaptive_timeout
            assert client.timeout_manager.persistence_enabled
            assert str(temp_dir) in str(client.timeout_manager.persistence.storage_dir)

    def test_builder_with_persistence(self):
        """Test configuring persistence with the builder pattern."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a client with persistence enabled via builder
            client = (
                ReqxClientBuilder()
                .with_adaptive_timeout(True)
                .with_persistence(True, temp_dir)
                .build()
            )

            # Check that persistence is correctly configured
            assert client.adaptive_timeout
            assert client.timeout_manager.persistence_enabled
            assert str(temp_dir) in str(client.timeout_manager.persistence.storage_dir)

    @patch("src.utils.get_optimal_connection_pool_settings")
    def test_high_performance_with_persistence(self, mock_pool_settings):
        """Test that high performance profile works with persistence."""
        mock_pool_settings.return_value = {
            "max_connections": 200,
            "max_keepalive_connections": 50,
            "keepalive_expiry": 60,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a client with high performance and persistence
            client = (
                ReqxClientBuilder().for_high_performance().with_persistence(True, temp_dir).build()
            )

            # Check settings
            assert client.adaptive_timeout
            assert client.timeout_manager.persistence_enabled
            assert client.http3 == False  # HTTP/3 should not be enabled by default
            assert client.max_retries == 2  # High performance uses fewer retries
