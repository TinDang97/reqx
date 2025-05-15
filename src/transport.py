"""
Transport layer implementations for Reqx.

This module provides different transport implementations:
- AiohttpTransport: Uses aiohttp for HTTP/1.0 and HTTP/1.1 connections (better performance)
- HttpxTransport: Uses httpx for HTTP/2 and HTTP/3 connections (better feature support)
- HybridTransport: Dynamically selects the appropriate transport based on the protocol
"""

import asyncio
import logging
import ssl
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypeVar
from urllib.parse import urlparse

import aiohttp
import httpx
from httpx import AsyncClient, Limits, Timeout, TransportError

from src.models import ReqxResponse

logger = logging.getLogger("reqx.transport")

T = TypeVar("T")


class BaseTransport(ABC):
    """Abstract base class for transport implementations."""

    @abstractmethod
    async def request(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> ReqxResponse:
        """Send an HTTP request and return a response."""
        pass

    @abstractmethod
    async def close(self):
        """Close the transport and release resources."""
        pass

    @abstractmethod
    def supports_http2(self) -> bool:
        """Check if this transport supports HTTP/2."""
        pass

    @abstractmethod
    def supports_http3(self) -> bool:
        """Check if this transport supports HTTP/3."""
        pass


class HttpxTransport(BaseTransport):
    """Transport implementation using httpx."""

    def __init__(
        self,
        base_url: str = "",
        headers: Dict[str, str] | None = None,
        cookies: Dict[str, str] | None = None,
        timeout: float = 30.0,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        keepalive_expiry: int = 60,
        follow_redirects: bool = True,
        verify_ssl: bool = True,
        http2: bool = False,
        enable_http3: bool = False,
        **kwargs,
    ):
        """Initialize the httpx transport with custom configuration."""
        # Configure connection pooling
        limits = Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            keepalive_expiry=keepalive_expiry,
        )

        # Configure SSL
        self.ssl_context = ssl.create_default_context()
        if not verify_ssl:
            self.ssl_context.check_hostname = False
            self.ssl_context.verify_mode = ssl.CERT_NONE

        # Initialize the client with HTTP/2 or HTTP/3 if enabled
        client_kwargs = {
            "base_url": base_url or "",
            "headers": headers,
            "cookies": cookies,
            "timeout": Timeout(timeout),
            "follow_redirects": follow_redirects,
            "verify": verify_ssl,
            "limits": limits,
            "http2": http2,
        }

        if enable_http3:
            try:
                # Import HTTP/3 support if available
                from httpx_h3 import H3Transport  # type: ignore[import]

                client_kwargs["transport"] = H3Transport()
                logger.debug("HTTP/3 (QUIC) support enabled")
                self._http3_enabled = True
            except ImportError:
                logger.warning(
                    "HTTP/3 requested but httpx_h3 not installed. Falling back to HTTP/1.1/HTTP/2"
                )
                self._http3_enabled = False
        else:
            self._http3_enabled = False

        self._http2_enabled = http2
        self.client = AsyncClient(**client_kwargs)

    async def request(self, method: str, url: str, **kwargs) -> ReqxResponse:
        """Send an HTTP request using httpx and return a response."""
        request_start_time = time.time()
        response = await self.client.request(method=method, url=url, **kwargs)
        request_end_time = time.time()

        # Create ReqxResponse from httpx response
        reqx_response = ReqxResponse.from_httpx_response(
            response,
            request_start_time=request_start_time,
            request_end_time=request_end_time,
            transport_info={
                "transport": "httpx",
                "http2": self._http2_enabled,
                "http3": self._http3_enabled,
                "base_url": self.client.base_url,
                "follow_redirects": self.client.follow_redirects,
                "timeout": self.client.timeout,
            },
        )
        return reqx_response

    async def close(self):
        """Close the httpx client session."""
        if hasattr(self, "client"):
            await self.client.aclose()

    def supports_http2(self) -> bool:
        """Check if this transport supports HTTP/2."""
        return self._http2_enabled

    def supports_http3(self) -> bool:
        """Check if this transport supports HTTP/3."""
        return self._http3_enabled

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of transport performance metrics.

        Returns:
            Dictionary with basic metrics (since HttpxTransport doesn't track metrics)
        """
        return {
            "transports": {
                "httpx": {
                    "requests": 0,
                    "avg_time": 0,
                    "min_time": None,
                    "max_time": 0,
                    "error_rate": 0,
                }
            },
            "hosts_analyzed": 0,
            "transport_preferences": {"httpx": 0, "aiohttp": 0, "unknown": 0},
        }

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get detailed metrics about transport usage and performance.

        Returns:
            Dictionary with minimal metrics information (since HttpxTransport doesn't track metrics)
        """
        return {
            "total_requests": 0,
            "transports": {"httpx": {"requests": 0, "errors": 0, "avg_time": 0}},
            "hosts_analyzed": 0,
        }


class AiohttpTransport(BaseTransport):
    """Transport implementation using aiohttp for better HTTP/1.1 performance."""

    def __init__(
        self,
        base_url: str | None = None,
        headers: Dict[str, str] | None = None,
        cookies: Dict[str, str] | None = None,
        timeout: float = 30.0,
        max_connections: int = 100,
        keepalive_expiry: int = 60,
        follow_redirects: bool = True,
        verify_ssl: bool = True,
        **kwargs,
    ):
        """Initialize the aiohttp transport with custom configuration."""
        # Configure SSL context
        if not verify_ssl:
            ssl_context = False  # aiohttp uses False to disable SSL verification
        else:
            ssl_context = True  # Use default SSL context

        # Create a cookie jar if cookies provided
        cookie_jar = None
        if cookies:
            cookie_jar = aiohttp.CookieJar()
            for name, value in cookies.items():
                cookie_jar.update_cookies({name: value})

        # Configure connection limits
        connector = aiohttp.TCPConnector(
            limit=max_connections,
            ssl=ssl_context,
            ttl_dns_cache=300,
            keepalive_timeout=keepalive_expiry,
        )

        # Create session
        self.session = aiohttp.ClientSession(
            base_url=base_url,
            headers=headers,
            cookie_jar=cookie_jar,
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=timeout),
        )

        self.follow_redirects = follow_redirects
        self.base_url = base_url

    async def request(self, method: str, url: str, **kwargs) -> ReqxResponse:
        """Send an HTTP request using aiohttp and return a response."""
        # Convert kwargs to aiohttp format
        aiohttp_kwargs = self._convert_kwargs_to_aiohttp(kwargs)

        try:
            request_start_time = time.time()
            # Make the request with aiohttp
            async with self.session.request(method, url, **aiohttp_kwargs) as aiohttp_response:
                request_end_time = time.time()
                # Read the content
                content = await aiohttp_response.read()

                # Create ReqxResponse directly
                reqx_response = ReqxResponse(
                    status_code=aiohttp_response.status,
                    headers=dict(aiohttp_response.headers),
                    content=content,
                    request=httpx.Request(method, url),
                    request_start_time=request_start_time,
                    request_end_time=request_end_time,
                    url=url,
                    transport_info={
                        "transport": "aiohttp",
                        "base_url": self.base_url,
                        "follow_redirects": self.follow_redirects,
                        "timeout": aiohttp_kwargs.get("timeout"),
                    },
                )

                return reqx_response
        except aiohttp.ClientError as e:
            # Convert aiohttp exceptions to httpx exceptions
            raise TransportError(f"Transport error: {str(e)}") from e

    async def close(self):
        """Close the aiohttp session."""
        if hasattr(self, "session") and not self.session.closed:
            await self.session.close()

    def supports_http2(self) -> bool:
        """Check if this transport supports HTTP/2."""
        return False

    def supports_http3(self) -> bool:
        """Check if this transport supports HTTP/3."""
        return False

    def _convert_kwargs_to_aiohttp(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert httpx kwargs to aiohttp kwargs."""
        result = {}

        # Convert basic parameters
        if "params" in kwargs:
            result["params"] = kwargs["params"]

        if "headers" in kwargs:
            result["headers"] = kwargs["headers"]

        if "cookies" in kwargs:
            result["cookies"] = kwargs["cookies"]

        if "timeout" in kwargs:
            # Convert httpx Timeout to aiohttp timeout
            if isinstance(kwargs["timeout"], Timeout):
                timeout = kwargs["timeout"].read or kwargs["timeout"].connect
                result["timeout"] = aiohttp.ClientTimeout(total=timeout)
            else:
                result["timeout"] = aiohttp.ClientTimeout(total=kwargs["timeout"])

        # Handle json data
        if "json" in kwargs:
            result["json"] = kwargs["json"]

        # Handle form data or raw request body
        if "data" in kwargs:
            result["data"] = kwargs["data"]

        # Handle file uploads
        if "files" in kwargs:
            # Convert httpx file format to aiohttp format
            form = aiohttp.FormData()
            for field_name, file_info in (kwargs["files"] or {}).items():
                if isinstance(file_info, tuple):
                    filename, content = file_info
                    form.add_field(field_name, content, filename=filename)
                else:
                    form.add_field(field_name, file_info)
            result["data"] = form

        # Handle follow_redirects
        if "follow_redirects" in kwargs:
            result["allow_redirects"] = kwargs["follow_redirects"]
        else:
            result["allow_redirects"] = self.follow_redirects

        # Handle SSL verification
        if "verify" in kwargs:
            if not kwargs["verify"]:
                # Disable SSL verification
                # This would need the connector to be recreated which isn't feasible here
                # So we log a warning instead
                logger.warning("Cannot dynamically change SSL verification in aiohttp transport")

        return result


class TransportMetrics:
    """Collect and analyze performance metrics for different transports."""

    def __init__(self):
        """Initialize the metrics collector."""
        self.metrics = {
            "httpx": {
                "requests": 0,
                "errors": 0,
                "total_time": 0.0,
                "min_time": float("inf"),
                "max_time": 0.0,
                "status_codes": {},
            },
            "aiohttp": {
                "requests": 0,
                "errors": 0,
                "total_time": 0.0,
                "min_time": float("inf"),
                "max_time": 0.0,
                "status_codes": {},
            },
        }
        self.host_performance = {}  # Track per-host performance

    def record_request(
        self, transport_name: str, host: str, duration: float, status_code: int, error: bool = False
    ):
        """
        Record metrics for a request.

        Args:
            transport_name: Name of the transport used ('httpx' or 'aiohttp')
            host: Hostname that was queried
            duration: Request duration in seconds
            status_code: HTTP status code returned
            error: Whether the request resulted in an error
        """
        # Update global transport metrics
        transport_metrics = self.metrics[transport_name]
        transport_metrics["requests"] += 1
        transport_metrics["total_time"] += duration

        if error:
            transport_metrics["errors"] += 1

        # Update min/max times
        if duration < transport_metrics["min_time"]:
            transport_metrics["min_time"] = duration
        if duration > transport_metrics["max_time"]:
            transport_metrics["max_time"] = duration

        # Record status code
        status_str = str(status_code)
        transport_metrics["status_codes"][status_str] = (
            transport_metrics["status_codes"].get(status_str, 0) + 1
        )

        # Update host-specific metrics
        if host not in self.host_performance:
            self.host_performance[host] = {
                "httpx": {"count": 0, "total_time": 0.0, "errors": 0},
                "aiohttp": {"count": 0, "total_time": 0.0, "errors": 0},
            }

        host_metrics = self.host_performance[host][transport_name]
        host_metrics["count"] += 1
        host_metrics["total_time"] += duration
        if error:
            host_metrics["errors"] += 1

    def get_preferred_transport(self, host: str) -> Optional[str]:
        """
        Get the preferred transport for a host based on collected metrics.

        Args:
            host: Hostname to check

        Returns:
            'httpx', 'aiohttp', or None if insufficient data
        """
        if host not in self.host_performance:
            return None

        host_data = self.host_performance[host]
        httpx_data = host_data["httpx"]
        aiohttp_data = host_data["aiohttp"]

        # Need at least 5 requests with each transport for meaningful comparison
        if httpx_data["count"] < 5 or aiohttp_data["count"] < 5:
            return None

        # Calculate average request time for each transport
        httpx_avg = (
            httpx_data["total_time"] / httpx_data["count"]
            if httpx_data["count"] > 0
            else float("inf")
        )
        aiohttp_avg = (
            aiohttp_data["total_time"] / aiohttp_data["count"]
            if aiohttp_data["count"] > 0
            else float("inf")
        )

        # Calculate error rates
        httpx_error_rate = (
            httpx_data["errors"] / httpx_data["count"] if httpx_data["count"] > 0 else 1.0
        )
        aiohttp_error_rate = (
            aiohttp_data["errors"] / aiohttp_data["count"] if aiohttp_data["count"] > 0 else 1.0
        )

        # Factor in both speed and reliability (weighted)
        httpx_score = (0.7 * httpx_avg) + (0.3 * httpx_error_rate * 10)  # Higher score is worse
        aiohttp_score = (0.7 * aiohttp_avg) + (
            0.3 * aiohttp_error_rate * 10
        )  # Higher score is worse

        # Choose the transport with the better score (lower is better)
        return "httpx" if httpx_score <= aiohttp_score else "aiohttp"

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of collected metrics.

        Returns:
            Dictionary with transport performance metrics
        """
        summary = {
            "transports": {},
            "hosts_analyzed": len(self.host_performance),
            "transport_preferences": {},
        }

        # Calculate transport averages
        for name, metrics in self.metrics.items():
            if metrics["requests"] > 0:
                avg_time = metrics["total_time"] / metrics["requests"]
                error_rate = (
                    metrics["errors"] / metrics["requests"] if metrics["requests"] > 0 else 0
                )

                summary["transports"][name] = {
                    "requests": metrics["requests"],
                    "avg_time": avg_time,
                    "min_time": (
                        metrics["min_time"] if metrics["min_time"] != float("inf") else None
                    ),
                    "max_time": metrics["max_time"],
                    "error_rate": error_rate,
                }

        # Count hosts preferring each transport
        preferences = {"httpx": 0, "aiohttp": 0, "unknown": 0}

        for host in self.host_performance:
            preferred = self.get_preferred_transport(host)
            if preferred == "httpx":
                preferences["httpx"] += 1
            elif preferred == "aiohttp":
                preferences["aiohttp"] += 1
            else:
                preferences["unknown"] += 1

        summary["transport_preferences"] = preferences
        return summary


class HybridTransport(BaseTransport):
    """
    Transport that selects the appropriate implementation based on the protocol.

    Uses aiohttp for HTTP/1.0 and HTTP/1.1 (better performance)
    Uses httpx for HTTP/2 and HTTP/3 (better feature support)
    """

    def __init__(
        self,
        base_url: str | None = None,
        headers: Dict[str, str] | None = None,
        cookies: Dict[str, str] | None = None,
        timeout: float = 30.0,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        keepalive_expiry: int = 60,
        follow_redirects: bool = True,
        verify_ssl: bool = True,
        http2: bool = False,
        enable_http3: bool = False,
        protocol_detection_cache_ttl: int = 3600,  # Cache protocol detection for 1 hour
        enable_metrics: bool = True,  # Enable performance metrics
        transport_learning: bool = False,  # Enable intelligent transport selection
        **kwargs,
    ):
        """Initialize the hybrid transport with custom configuration."""
        # Common configuration for both transports
        self.base_args = {
            "base_url": base_url,
            "headers": headers,
            "cookies": cookies,
            "timeout": timeout,
            "max_connections": max_connections,
            "keepalive_expiry": keepalive_expiry,
            "follow_redirects": follow_redirects,
            "verify_ssl": verify_ssl,
        }

        # Initialize both transports
        self.aiohttp = AiohttpTransport(**self.base_args)

        # Add httpx-specific args
        httpx_args = {
            **self.base_args,
            "max_keepalive_connections": max_keepalive_connections,
            "http2": http2,
            "enable_http3": enable_http3,
        }
        self.httpx = HttpxTransport(**httpx_args)

        # Configuration flags
        self.http2 = http2
        self.http3 = enable_http3
        self.transport_learning = transport_learning

        # Protocol cache to avoid repeated analysis of the same URL
        self._protocol_cache = {}

        # Protocol detection enhancements
        self.protocol_detection_cache_ttl = protocol_detection_cache_ttl
        self._protocol_cache_timestamp = {}
        self._protocol_capabilities = {}  # Store detected capabilities by host

        # Performance metrics tracking
        self.enable_metrics = (
            enable_metrics or transport_learning
        )  # Always enable metrics when learning
        self.metrics = TransportMetrics() if self.enable_metrics else None

        # Track transport preferences when learning is enabled
        self._transport_preferences = {} if transport_learning else None

    async def request(self, method: str, url: str, **kwargs) -> ReqxResponse:
        """
        Send an HTTP request using the appropriate transport and return a response.
        """
        parsed_url = urlparse(url)
        host = parsed_url.netloc.split(":")[0]
        force_http2 = kwargs.pop("force_http2", False)

        transport = self._select_transport_for_url(url, force_http2)
        transport_name = "httpx" if isinstance(transport, HttpxTransport) else "aiohttp"

        start_time = time.time()
        error = False
        status_code = 0

        try:
            response = await transport.request(method, url, **kwargs)
            status_code = response.status_code
            return response
        except Exception as e:
            error = True
            # If one transport fails, try the other one as fallback
            fallback_transport = self.httpx if transport_name == "aiohttp" else self.aiohttp
            logger.warning(
                f"Request with {transport_name} failed: {str(e)}. " f"Trying fallback transport..."
            )

            # Record the failed attempt in metrics
            if self.enable_metrics and self.metrics:
                duration = time.time() - start_time
                self.metrics.record_request(transport_name, host, duration, 0, error=True)

            # Try the fallback transport
            start_time = time.time()
            try:
                response = await fallback_transport.request(method, url, **kwargs)

                # Record successful fallback attempt
                if self.enable_metrics and self.metrics:
                    duration = time.time() - start_time
                    fallback_name = (
                        "httpx" if isinstance(fallback_transport, HttpxTransport) else "aiohttp"
                    )
                    self.metrics.record_request(
                        fallback_name, host, duration, response.status_code, error=False
                    )

                    # Update our hostname-based preferences since the fallback worked better
                    self._protocol_capabilities[host] = {
                        "http2": isinstance(fallback_transport, HttpxTransport),
                        "http3": False,
                        "supports_aiohttp_optimizations": isinstance(
                            fallback_transport, AiohttpTransport
                        ),
                    }

                return response
            except Exception:
                # If both transports fail, re-raise the original exception
                raise
        finally:
            # Record metrics for successful requests
            if not error and self.enable_metrics and self.metrics:
                duration = time.time() - start_time
                self.metrics.record_request(transport_name, host, duration, status_code)

    async def close(self):
        """Close both transport sessions."""
        await asyncio.gather(self.aiohttp.close(), self.httpx.close())

    def supports_http2(self) -> bool:
        """Check if this transport supports HTTP/2."""
        return self.httpx.supports_http2()

    def supports_http3(self) -> bool:
        """Check if this transport supports HTTP/3."""
        return self.httpx.supports_http3()

    async def _detect_protocol_capabilities(self, host: str) -> Dict[str, bool]:
        """
        Actively detect protocol capabilities for a given host.

        Args:
            host: The hostname to check

        Returns:
            Dictionary of protocol capabilities
        """
        capabilities = {
            "http2": False,
            "http3": False,
            "supports_aiohttp_optimizations": True,  # Default to True for aiohttp
        }

        # First, try an HTTP/2 connection with httpx
        try:
            # Create a temporary client with HTTP/2
            import asyncio

            from httpx import AsyncClient

            async with AsyncClient(http2=True) as client:
                # Set a shorter timeout for detection
                response = await asyncio.wait_for(client.get(f"https://{host}/"), timeout=2.0)

                # Check if HTTP/2 was used
                if response.http_version == "HTTP/2":
                    capabilities["http2"] = True
                    # Sites with HTTP/2 often benefit less from aiohttp
                    capabilities["supports_aiohttp_optimizations"] = False
        except Exception:
            # If connection fails, continue with current capabilities
            pass

        # Cache the results
        self._protocol_capabilities[host] = capabilities
        self._protocol_cache_timestamp[host] = time.time()

        return capabilities

    def _select_transport_for_url(self, url: str, force_http2: bool = False) -> BaseTransport:
        """
        Select the appropriate transport based on URL analysis or cache.

        Args:
            url: The URL to analyze
            force_http2: Whether to force HTTP/2 regardless of URL

        Returns:
            The appropriate transport for the given URL
        """
        # Return cached decision if available and not expired
        parsed_url = urlparse(url)
        host = parsed_url.netloc.split(":")[0]

        # Check if we need to refresh the protocol cache
        should_refresh_cache = False
        if host in self._protocol_cache_timestamp:
            cache_age = time.time() - self._protocol_cache_timestamp.get(host, 0)
            if cache_age > self.protocol_detection_cache_ttl:
                should_refresh_cache = True

        # Force HTTP/2 if requested or HTTP/2 is globally enabled
        if force_http2 or self.http2 or self.http3:
            return self.httpx

        # Use metrics-based decision if available
        if self.enable_metrics and self.metrics:
            preferred = self.metrics.get_preferred_transport(host)
            if preferred == "httpx":
                return self.httpx
            elif preferred == "aiohttp":
                return self.aiohttp

        # If we have cached capabilities, use them to decide
        if host in self._protocol_capabilities and not should_refresh_cache:
            capabilities = self._protocol_capabilities[host]

            # Use HTTP/2 if supported or H3 is configured
            if capabilities["http2"] or self.http3:
                return self.httpx

            # Use aiohttp if it's particularly well-suited
            if capabilities["supports_aiohttp_optimizations"]:
                return self.aiohttp

        # List of hosts known to benefit from HTTP/2
        http2_hosts = {
            "google.com",
            "www.google.com",
            "youtube.com",
            "www.youtube.com",
            "facebook.com",
            "www.facebook.com",
            "twitter.com",
            "www.twitter.com",
            "x.com",
            "www.x.com",  # Twitter's new domain
            "github.com",
            "www.github.com",
            "cdn.",
            "static.",
            "media.",  # Common CDN prefixes
            "amazonaws.com",
            "cloudfront.net",
            "akamaized.net",  # Popular CDNs
            "fastly.net",
            "cloudflare.com",  # More CDNs
        }

        # Check if the host matches any HTTP/2 hosts
        if any(host.endswith(h) or host.startswith(h) for h in http2_hosts):
            # Schedule protocol detection in the background for future optimization
            asyncio.create_task(self._detect_protocol_capabilities(host))
            return self.httpx
        else:
            # Default to aiohttp for better HTTP/1.1 performance
            # Also schedule protocol detection in the background
            asyncio.create_task(self._detect_protocol_capabilities(host))
            return self.aiohttp

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of transport performance metrics.

        Returns:
            Dictionary with collected metrics or empty dict if metrics are disabled
        """
        if self.enable_metrics and self.metrics:
            return self.metrics.get_summary()
        return {}

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get detailed metrics about transport usage and performance.

        Returns:
            Dictionary with detailed transport metrics and host-specific information
        """
        if not self.enable_metrics or not self.metrics:
            return {}

        # Get base summary metrics
        metrics = self.metrics.get_summary()

        # Add total requests across all transports
        total_requests = sum(t.get("requests", 0) for t in metrics.get("transports", {}).values())
        metrics["total_requests"] = total_requests

        # Add detailed host metrics
        host_metrics = {}

        for host, host_data in self.metrics.host_performance.items():
            # Calculate average durations
            httpx_data = host_data["httpx"]
            aiohttp_data = host_data["aiohttp"]

            httpx_avg = (
                httpx_data["total_time"] / httpx_data["count"] if httpx_data["count"] > 0 else None
            )

            aiohttp_avg = (
                aiohttp_data["total_time"] / aiohttp_data["count"]
                if aiohttp_data["count"] > 0
                else None
            )

            # Determine preferred transport
            preferred = self.metrics.get_preferred_transport(host)

            host_metrics[host] = {
                "preferred": preferred or "unknown",
                "httpx": {
                    "count": httpx_data["count"],
                    "avg_duration": httpx_avg,
                    "errors": httpx_data["errors"],
                    "error_rate": (
                        httpx_data["errors"] / httpx_data["count"] if httpx_data["count"] > 0 else 0
                    ),
                },
                "aiohttp": {
                    "count": aiohttp_data["count"],
                    "avg_duration": aiohttp_avg,
                    "errors": aiohttp_data["errors"],
                    "error_rate": (
                        aiohttp_data["errors"] / aiohttp_data["count"]
                        if aiohttp_data["count"] > 0
                        else 0
                    ),
                },
            }

        metrics["host_metrics"] = host_metrics
        return metrics
