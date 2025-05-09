"""
Tests for the adaptive timeout management feature.
"""

import pytest

from src.client import AdaptiveTimeoutManager


class TestAdaptiveTimeoutManager:
    """Tests for the AdaptiveTimeoutManager class."""

    def test_initialization(self):
        """Test that the timeout manager initializes with correct defaults."""
        manager = AdaptiveTimeoutManager(default_timeout=30.0)

        assert manager.default_timeout == 30.0
        assert manager.min_timeout == 1.0
        assert manager.max_timeout == 120.0
        assert manager.host_metrics == {}

    def test_record_request(self):
        """Test recording request metrics."""
        manager = AdaptiveTimeoutManager()

        # Record successful request
        manager.record_request("example.com", 0.5, success=True)

        # Verify host was added to metrics
        assert "example.com" in manager.host_metrics
        assert manager.host_metrics["example.com"]["successes"] == 1
        assert manager.host_metrics["example.com"]["failures"] == 0
        assert manager.host_metrics["example.com"]["durations"] == [0.5]

        # Record failed request
        manager.record_request("example.com", 1.5, success=False)

        # Verify metrics were updated
        assert manager.host_metrics["example.com"]["successes"] == 1
        assert manager.host_metrics["example.com"]["failures"] == 1
        assert manager.host_metrics["example.com"]["durations"] == [0.5, 1.5]

    def test_record_timeout(self):
        """Test recording timeout events."""
        manager = AdaptiveTimeoutManager()

        # Record a timeout
        manager.record_timeout("example.com")

        # Verify timeout was recorded
        assert "example.com" in manager.host_metrics
        assert manager.host_metrics["example.com"]["timeouts"] == 1

        # Record another timeout
        manager.record_timeout("example.com")
        assert manager.host_metrics["example.com"]["timeouts"] == 2

    def test_get_timeout_insufficient_data(self):
        """Test that default timeout is used when there's not enough data."""
        manager = AdaptiveTimeoutManager(default_timeout=5.0)

        # Record just a few samples (below threshold)
        for i in range(10):
            manager.record_request("example.com", 0.2, success=True)

        # Should use current_timeout since we don't have enough samples
        timeout = manager.get_timeout("example.com")
        assert timeout == 5.0

        # For an unknown host, should return default timeout
        timeout = manager.get_timeout("unknown.com")
        assert timeout == 5.0

    def test_get_timeout_calculates_percentile(self):
        """Test that timeout calculation uses percentile of request durations."""
        manager = AdaptiveTimeoutManager(default_timeout=1.0)
        manager.sample_size = 10  # Lower threshold for testing

        # Add some varied durations
        durations = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 5.0]
        for d in durations:
            manager.record_request("example.com", d, success=True)

        # Get calculated timeout
        timeout = manager.get_timeout("example.com")

        # The 95th percentile of durations is 5.0, multiplied by safety factor 1.5 = 7.5
        expected_timeout = 5.0 * 1.5  # 95th percentile * safety factor

        # Allow for small floating point differences
        assert abs(timeout - expected_timeout) < 0.1

    def test_get_timeout_respects_limits(self):
        """Test that calculated timeouts respect min and max limits."""
        # Test minimum limit
        min_manager = AdaptiveTimeoutManager(default_timeout=2.0, min_timeout=0.5)
        min_manager.sample_size = 10  # Lower threshold for testing

        # Add samples with very small durations
        for _ in range(10):
            min_manager.record_request("example.com", 0.1, success=True)

        # Calculated raw timeout would be 0.1 * 1.5 = 0.15, but min is 0.5
        assert min_manager.get_timeout("example.com") == 0.5

        # Test maximum limit
        max_manager = AdaptiveTimeoutManager(default_timeout=10.0, max_timeout=15.0)
        max_manager.sample_size = 10  # Lower threshold for testing

        # Add samples with large durations
        for _ in range(10):
            max_manager.record_request("example.com", 20.0, success=True)

        # Calculated raw timeout would be 20.0 * 1.5 = 30.0, but max is 15.0
        assert max_manager.get_timeout("example.com") == 15.0

    def test_timeout_factors_in_timeouts(self):
        """Test that timeout history influences calculation."""
        manager = AdaptiveTimeoutManager(default_timeout=5.0)
        manager.sample_size = 10  # Lower threshold for testing

        # Add some baseline data
        for _ in range(10):
            manager.record_request("example.com", 1.0, success=True)

        # Get baseline timeout
        baseline = manager.get_timeout("example.com")

        # Now add some timeouts
        for _ in range(5):
            manager.record_timeout("example.com")

        # Get new timeout - should be higher due to timeout history
        new_timeout = manager.get_timeout("example.com")

        # New timeout should be higher
        assert new_timeout > baseline

    def test_get_statistics(self):
        """Test retrieving timeout statistics."""
        manager = AdaptiveTimeoutManager()

        # Add data for multiple hosts
        manager.record_request("example.com", 0.5, success=True)
        manager.record_request("example.com", 1.0, success=True)
        manager.record_timeout("example.com")

        manager.record_request("google.com", 0.2, success=True)
        manager.record_request("google.com", 0.1, success=False)

        # Get statistics
        stats = manager.get_statistics()

        # Verify structure
        assert "example.com" in stats
        assert "google.com" in stats

        # Verify example.com metrics
        assert stats["example.com"]["samples"] == 2
        assert stats["example.com"]["avg_duration"] == 0.75
        assert stats["example.com"]["min_duration"] == 0.5
        assert stats["example.com"]["max_duration"] == 1.0
        assert stats["example.com"]["success_rate"] > 0
        assert stats["example.com"]["timeout_rate"] > 0

        # Verify google.com metrics
        assert stats["google.com"]["samples"] == 2
        assert stats["google.com"]["avg_duration"] == 0.15
