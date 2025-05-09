"""
Tests for the ReqxClientBuilder class.
"""

import pytest
from unittest.mock import patch, Mock

from src.builder import ReqxClientBuilder
from src.client import ReqxClient


class TestReqxClientBuilder:
    """Tests for ReqxClientBuilder."""

    def test_basic_builder_functionality(self):
        """Test basic builder configuration."""
        builder = ReqxClientBuilder()

        # Configure basic settings
        configured_builder = (
            builder.with_base_url("https://example.com")
            .with_timeout(10.0)
            .with_headers({"User-Agent": "Test"})
            .with_retry(max_retries=5, backoff=1.0)
        )

        # Build the client
        with patch("src.client.ReqxClient") as mock_client:
            mock_client.return_value = Mock()
            client = configured_builder.build()

            # Verify client was created with correct config
            mock_client.assert_called_once()
            call_kwargs = mock_client.call_args[1]

            assert call_kwargs["base_url"] == "https://example.com"
            assert call_kwargs["timeout"] == 10.0
            assert call_kwargs["headers"] == {"User-Agent": "Test"}
            assert call_kwargs["max_retries"] == 5
            assert call_kwargs["retry_backoff"] == 1.0

    def test_adaptive_timeout_configuration(self):
        """Test configuring adaptive timeouts."""
        builder = ReqxClientBuilder()

        # Enable adaptive timeouts
        configured_builder = builder.with_adaptive_timeout(True)

        # Build the client
        with patch("src.client.ReqxClient") as mock_client:
            mock_client.return_value = Mock()
            client = configured_builder.build()

            # Verify adaptive_timeout was set
            call_kwargs = mock_client.call_args[1]
            assert call_kwargs["adaptive_timeout"] == True

    def test_high_performance_profile(self):
        """Test the high performance profile configuration."""
        builder = ReqxClientBuilder()

        # Apply high performance profile
        configured_builder = builder.for_high_performance()

        # Build the client
        with patch("src.client.ReqxClient") as mock_client:
            with patch("src.builder.get_optimal_connection_pool_settings") as mock_pool:
                # Mock the pool settings
                mock_pool.return_value = {"max_connections": 200, "max_keepalive_connections": 50}

                mock_client.return_value = Mock()
                client = configured_builder.build()

                # Verify high performance settings were applied
                call_kwargs = mock_client.call_args[1]

                # Check key settings that define high performance
                assert call_kwargs["http2"] == True
                assert call_kwargs["enable_cache"] == True
                assert call_kwargs["adaptive_timeout"] == True
                assert call_kwargs["max_connections"] == 200  # From mocked pool settings
                assert call_kwargs["max_keepalive_connections"] == 50  # From mocked pool settings

    def test_reliability_profile(self):
        """Test the reliability profile configuration."""
        builder = ReqxClientBuilder()

        # Apply reliability profile
        configured_builder = builder.for_reliability()

        # Build the client
        with patch("src.client.ReqxClient") as mock_client:
            mock_client.return_value = Mock()
            client = configured_builder.build()

            # Verify reliability settings were applied
            call_kwargs = mock_client.call_args[1]

            # Check key settings that define reliability
            assert call_kwargs["timeout"] == 60.0  # Longer timeout
            assert call_kwargs["max_retries"] == 5  # More retries
            assert call_kwargs["retry_backoff"] == 1.0  # Longer backoff
            assert call_kwargs["adaptive_timeout"] == True  # Adaptive timeouts

    def test_auto_optimize_settings(self):
        """Test auto-optimization of connection pool settings."""
        builder = ReqxClientBuilder()

        # Enable auto optimization
        configured_builder = builder.auto_optimize()

        # Build the client
        with patch("src.client.ReqxClient") as mock_client:
            with patch("src.builder.get_optimal_connection_pool_settings") as mock_pool:
                # Mock the pool settings
                mock_pool.return_value = {
                    "max_connections": 150,
                    "max_keepalive_connections": 40,
                    "keepalive_expiry": 90,
                }

                mock_client.return_value = Mock()
                client = configured_builder.build()

                # Verify pool settings were applied
                call_kwargs = mock_client.call_args[1]

                assert call_kwargs["max_connections"] == 150
                assert call_kwargs["max_keepalive_connections"] == 40
                assert call_kwargs["keepalive_expiry"] == 90

    def test_manual_settings_override_auto_optimize(self):
        """Test that manual settings take precedence over auto optimization."""
        builder = ReqxClientBuilder()

        # Configure manual connection pool settings first
        configured_builder = builder.with_connection_pool(
            max_connections=50, max_keepalive=10, keepalive_expiry=30
        ).auto_optimize()  # Enable auto optimization after manual settings

        # Build the client
        with patch("src.client.ReqxClient") as mock_client:
            with patch("src.builder.get_optimal_connection_pool_settings") as mock_pool:
                # Mock the pool settings
                mock_pool.return_value = {
                    "max_connections": 150,
                    "max_keepalive_connections": 40,
                    "keepalive_expiry": 90,
                }

                mock_client.return_value = Mock()
                client = configured_builder.build()

                # Verify manual settings were preserved
                call_kwargs = mock_client.call_args[1]

                assert call_kwargs["max_connections"] == 50  # Manual setting
                assert call_kwargs["max_keepalive_connections"] == 10  # Manual setting
                assert call_kwargs["keepalive_expiry"] == 30  # Manual setting

    def test_http2_disables_aiohttp(self):
        """Test that enabling HTTP/2 disables aiohttp for consistency."""
        builder = ReqxClientBuilder()

        # Enable HTTP/2
        configured_builder = builder.with_http2(True)

        # Build the client
        with patch("src.client.ReqxClient") as mock_client:
            mock_client.return_value = Mock()
            client = configured_builder.build()

            # Verify HTTP/2 was enabled and aiohttp was disabled
            call_kwargs = mock_client.call_args[1]

            assert call_kwargs["http2"] == True
            assert call_kwargs["use_aiohttp"] == False

    def test_http3_disables_aiohttp(self):
        """Test that enabling HTTP/3 disables aiohttp for consistency."""
        builder = ReqxClientBuilder()

        # Enable HTTP/3
        configured_builder = builder.with_http3(True)

        # Build the client
        with patch("src.client.ReqxClient") as mock_client:
            mock_client.return_value = Mock()
            client = configured_builder.build()

            # Verify HTTP/3 was enabled and aiohttp was disabled
            call_kwargs = mock_client.call_args[1]

            assert call_kwargs["enable_http3"] == True
            assert call_kwargs["use_aiohttp"] == False

    def test_method_chaining(self):
        """Test that all builder methods properly return self for chaining."""
        builder = ReqxClientBuilder()

        # Create a long chain of method calls
        result = (
            builder.with_base_url("https://example.com")
            .with_timeout(10.0)
            .with_headers({"User-Agent": "Test"})
            .add_header("X-Test", "Value")
            .with_cookies({"session": "123"})
            .add_cookie("test", "value")
            .with_retry(max_retries=3, backoff=0.5)
            .with_http2(True)
            .with_debug(True)
            .with_cache(enabled=True, ttl=60)
            .with_adaptive_timeout(True)
            .with_rate_limit(100)
        )

        # Verify the result is still the builder
        assert result is builder

    def test_integration_with_client(self):
        """Test actual integration between builder and ReqxClient."""
        builder = ReqxClientBuilder()

        # Configure the client
        client = (
            builder.with_base_url("https://example.com")
            .with_timeout(10.0)
            .with_adaptive_timeout(True)
            .build()
        )

        # Verify client was created with correct type
        assert isinstance(client, ReqxClient)

        # Verify key settings were transferred
        assert client.base_url == "https://example.com"
        assert client.timeout == 10.0
        assert client.adaptive_timeout == True
