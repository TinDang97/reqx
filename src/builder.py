"""
Builder pattern for Reqx client configuration.

This module provides a builder pattern for creating and configuring
ReqxClient instances with a fluent API.
"""

from typing import Any, Dict, List, Optional, Type, Union

from .client import ReqxClient
from .utils import get_optimal_connection_pool_settings


class ReqxClientBuilder:
    """Builder class for ReqxClient with fluent API."""

    def __init__(self):
        """Initialize the builder with default values."""
        self._config = {
            "base_url": "",
            "headers": {},
            "cookies": {},
            "timeout": 30.0,
            "max_connections": 100,
            "max_keepalive_connections": 20,
            "keepalive_expiry": 60,
            "follow_redirects": True,
            "verify_ssl": True,
            "max_retries": 3,
            "retry_backoff": 0.5,
            "http2": False,
            "enable_http3": False,
            "use_aiohttp": True,  # Default to use aiohttp for better performance
            "debug": False,
            "enable_cache": False,
            "cache_ttl": 300,
            "rate_limit": None,
            "rate_limit_max_tokens": 60,
            "certificate_pins": None,
            "adaptive_timeout": False,  # Adaptive timeout setting
            "persistence_enabled": False,  # Whether to persist optimized settings
            "persistence_path": None,  # Custom path for persisted settings
        }

        # Track whether connection pool has been manually configured
        self._connection_pool_configured = False
        # Track whether auto-optimization has been enabled
        self._auto_optimize_enabled = False

    def with_base_url(self, base_url: str) -> "ReqxClientBuilder":
        """Set the base URL for all requests."""
        self._config["base_url"] = base_url
        return self

    def with_headers(self, headers: Dict[str, str]) -> "ReqxClientBuilder":
        """Set default headers for all requests."""
        self._config["headers"] = headers
        return self

    def add_header(self, name: str, value: str) -> "ReqxClientBuilder":
        """Add a single header to the default headers."""
        if "headers" not in self._config:
            self._config["headers"] = {}
        self._config["headers"][name] = value
        return self

    def with_cookies(self, cookies: Dict[str, str]) -> "ReqxClientBuilder":
        """Set default cookies for all requests."""
        self._config["cookies"] = cookies
        return self

    def add_cookie(self, name: str, value: str) -> "ReqxClientBuilder":
        """Add a single cookie to the default cookies."""
        if "cookies" not in self._config:
            self._config["cookies"] = {}
        self._config["cookies"][name] = value
        return self

    def with_timeout(self, timeout: float) -> "ReqxClientBuilder":
        """Set the default timeout for all requests in seconds."""
        self._config["timeout"] = timeout
        return self

    def with_connection_pool(
        self, max_connections: int, max_keepalive: int, keepalive_expiry: int
    ) -> "ReqxClientBuilder":
        """Configure the connection pool parameters."""
        self._config["max_connections"] = max_connections
        self._config["max_keepalive_connections"] = max_keepalive
        self._config["keepalive_expiry"] = keepalive_expiry
        self._connection_pool_configured = True
        return self

    def auto_optimize(self, enable: bool = True) -> "ReqxClientBuilder":
        """
        Enable automatic optimization of client settings based on system resources.

        When enabled, this optimizes connection pool settings and other parameters
        for better performance based on the current system's capabilities.

        Args:
            enable: Whether to enable auto-optimization

        Returns:
            The builder instance for method chaining
        """
        self._auto_optimize_enabled = enable
        return self

    def follow_redirects(self, follow: bool = True) -> "ReqxClientBuilder":
        """Configure whether to follow redirects by default."""
        self._config["follow_redirects"] = follow
        return self

    def verify_ssl(self, verify: bool = True) -> "ReqxClientBuilder":
        """Configure whether to verify SSL certificates by default."""
        self._config["verify_ssl"] = verify
        return self

    def with_retry(self, max_retries: int, backoff: float) -> "ReqxClientBuilder":
        """Configure retry behavior for failed requests."""
        self._config["max_retries"] = max_retries
        self._config["retry_backoff"] = backoff
        return self

    def with_http2(self, enabled: bool = True) -> "ReqxClientBuilder":
        """Enable or disable HTTP/2 support."""
        self._config["http2"] = enabled
        # If HTTP/2 is explicitly enabled, we should use httpx
        if enabled:
            self._config["use_aiohttp"] = False
        return self

    def with_http3(self, enabled: bool = True) -> "ReqxClientBuilder":
        """Enable or disable HTTP/3 (QUIC) support."""
        self._config["enable_http3"] = enabled
        # If HTTP/3 is explicitly enabled, we should use httpx
        if enabled:
            self._config["use_aiohttp"] = False
        return self

    def use_aiohttp(self, enabled: bool = True) -> "ReqxClientBuilder":
        """Configure whether to use aiohttp for HTTP/1.1 requests."""
        self._config["use_aiohttp"] = enabled
        return self

    def with_debug(self, debug: bool = True) -> "ReqxClientBuilder":
        """Enable or disable debug logging."""
        self._config["debug"] = debug
        return self

    def with_cache(self, enabled: bool = True, ttl: int = 300) -> "ReqxClientBuilder":
        """Configure response caching behavior."""
        self._config["enable_cache"] = enabled
        self._config["cache_ttl"] = ttl
        return self

    def with_rate_limit(
        self, requests_per_second: float, max_tokens: int = 60
    ) -> "ReqxClientBuilder":
        """Configure rate limiting behavior."""
        self._config["rate_limit"] = requests_per_second
        self._config["rate_limit_max_tokens"] = max_tokens
        return self

    def with_certificate_pins(self, pins: Dict[str, List[str]]) -> "ReqxClientBuilder":
        """Configure certificate pinning for enhanced security."""
        self._config["certificate_pins"] = pins
        return self

    def with_adaptive_timeout(self, enabled: bool = True) -> "ReqxClientBuilder":
        """
        Enable or disable adaptive timeout management.

        When enabled, the client will automatically adjust timeout values for each host
        based on historical performance data, improving both reliability and performance.

        Args:
            enabled: Whether to enable adaptive timeouts

        Returns:
            The builder instance for method chaining
        """
        self._config["adaptive_timeout"] = enabled
        return self

    def with_persistence(
        self, enabled: bool = True, path: Optional[str] = None
    ) -> "ReqxClientBuilder":
        """
        Enable or disable persistence of optimized settings.

        When enabled, learned settings like adaptive timeouts and transport preferences
        will be persisted to disk and reused across application restarts, allowing
        the client to maintain its optimizations.

        Args:
            enabled: Whether to enable settings persistence
            path: Optional custom path for storing persisted settings

        Returns:
            The builder instance for method chaining
        """
        self._config["persistence_enabled"] = enabled
        if path is not None:
            self._config["persistence_path"] = path
        return self

    def with_transport_learning(self, enabled: bool = True) -> "ReqxClientBuilder":
        """
        Enable or disable intelligent transport selection learning.

        When enabled, the client will track performance metrics for different transport types
        (httpx vs aiohttp) for each host and automatically select the best-performing transport
        for future requests to that host.

        Args:
            enabled: Whether to enable transport learning

        Returns:
            The builder instance for method chaining
        """
        self._config["transport_learning"] = enabled
        return self

    def for_high_performance(self) -> "ReqxClientBuilder":
        """
        Configure the client for high-performance use cases.

        This is a convenience method that applies several optimizations:
        - Enables HTTP/2
        - Sets optimal connection pool settings
        - Enables response caching
        - Uses a more aggressive retry strategy
        - Enables adaptive timeouts for optimal performance

        Returns:
            The builder instance for method chaining
        """
        # Enable performance optimizations
        self._config["http2"] = True
        self._config["enable_cache"] = True
        self._config["cache_ttl"] = 120  # 2 minute cache
        self._config["max_retries"] = 2  # Fewer retries for faster failures
        self._config["adaptive_timeout"] = True  # Enable adaptive timeouts

        # Optimize connection pool if not manually configured
        if not self._connection_pool_configured:
            pool_settings = get_optimal_connection_pool_settings()
            self._config.update(pool_settings)
            self._connection_pool_configured = True

        return self

    def for_reliability(self) -> "ReqxClientBuilder":
        """
        Configure the client for high-reliability use cases.

        This is a convenience method that applies several reliability-focused settings:
        - More aggressive retry strategy
        - Longer timeouts
        - Modest connection pooling
        - Enables adaptive timeouts for better reliability

        Returns:
            The builder instance for method chaining
        """
        # Enable reliability optimizations
        self._config["timeout"] = 60.0  # Longer timeout
        self._config["max_retries"] = 5  # More retries
        self._config["retry_backoff"] = 1.0  # Longer backoff
        self._config["adaptive_timeout"] = True  # Enable adaptive timeouts

        # Use more conservative connection pooling
        if not self._connection_pool_configured:
            self._config["max_connections"] = 50
            self._config["max_keepalive_connections"] = 20
            self._config["keepalive_expiry"] = 120  # Longer keepalive

        return self

    def build(self) -> ReqxClient:
        """Build and return a configured ReqxClient instance."""
        # Apply auto-optimization if enabled and connection pool not manually configured
        if self._auto_optimize_enabled and not self._connection_pool_configured:
            pool_settings = get_optimal_connection_pool_settings()
            self._config.update(pool_settings)

        return ReqxClient(**self._config)
