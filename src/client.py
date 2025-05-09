import asyncio
import base64
import hashlib
import logging
import os
import ssl
import time
from pydantic import TypeAdapter

from typing import Any, Awaitable, Callable, Dict, List, Optional, Type, TypeVar, Union
from urllib.parse import urljoin, urlparse

from httpx import Response, Timeout, TransportError

from .exceptions import MiddlewareError, RequestError, ResponseError, SecurityError
from .transport import AiohttpTransport, BaseTransport, HttpxTransport, HybridTransport
from .utils import log_request, log_response

logger = logging.getLogger("reqx")
# Set up logging
logging.basicConfig(level=logging.WARNING)

T = TypeVar("T")

try:
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    logger.warning("uvloop not installed. Using default asyncio event loop.")


# Define middleware types
RequestMiddleware = Callable[[str, str, Dict[str, Any]], Awaitable[Dict[str, Any]]]
ResponseMiddleware = Callable[[Response], Awaitable[Response]]

# Define batch request types
BatchRequestItem = TypeVar("BatchRequestItem", bound=dict)
BatchResponseItem = TypeVar("BatchResponseItem")


class BatchRequestError(Exception):
    """Represents an error in a batch request item."""

    def __init__(self, index: int, request: Dict[str, Any], error: Exception):
        self.index = index
        self.request = request
        self.error = error
        super().__init__(f"Error in batch request item {index}: {str(error)}")


class CacheEntry:
    """Represents a cached response."""

    __slots__ = ("response", "expires_at")

    def __init__(self, response: Response, expires_at: float):
        self.response = response
        self.expires_at = expires_at

    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return time.time() > self.expires_at


class RateLimiter:
    """Simple token bucket rate limiter."""

    def __init__(self, rate: float, max_tokens: int):
        """
        Initialize a rate limiter.

        Args:
            rate: Rate at which tokens are refilled per second
            max_tokens: Maximum number of tokens in the bucket
        """
        self.rate = rate
        self.max_tokens = max_tokens
        self.tokens = max_tokens
        self.last_refill = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens from the bucket, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            Time spent waiting in seconds
        """
        start_time = time.time()

        async with self.lock:
            await self._refill()
            wait_time = 0.0

            # If we don't have enough tokens, calculate wait time
            if self.tokens < tokens:
                wait_time = (tokens - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                await self._refill()

            self.tokens -= tokens
            return time.time() - start_time

    async def _refill(self):
        """Refill tokens based on time elapsed."""
        now = time.time()
        elapsed = now - self.last_refill
        self.last_refill = now

        # Add new tokens based on time elapsed
        new_tokens = elapsed * self.rate
        self.tokens = min(self.max_tokens, self.tokens + new_tokens)


class CertificatePinner:
    """Certificate pinning for enhanced security against MITM attacks."""

    def __init__(self, pins: Dict[str, List[str]] | None = None):
        """
        Initialize a certificate pinner.

        Args:
            pins: Dictionary mapping hostnames to lists of base64-encoded SHA-256 certificate hashes
                 Example: {"api.example.com": ["sha256/ABC123...", "sha256/DEF456..."]}
        """
        self.pins = pins or {}

    def add_pin(self, hostname: str, pin: str):
        """
        Add a certificate pin for a hostname.

        Args:
            hostname: The hostname to pin
            pin: Base64-encoded SHA-256 hash of the certificate public key,
                 prefixed with 'sha256/'
        """
        if hostname not in self.pins:
            self.pins[hostname] = []

        if pin not in self.pins[hostname]:
            self.pins[hostname].append(pin)

    def remove_pin(self, hostname: str, pin: str | None = None):
        """
        Remove certificate pins for a hostname.

        Args:
            hostname: The hostname to remove pins for
            pin: Specific pin to remove, or None to remove all pins for the hostname
        """
        if hostname not in self.pins:
            return

        if pin is None:
            del self.pins[hostname]
        elif pin in self.pins[hostname]:
            self.pins[hostname].remove(pin)

    def verify_certificate(self, hostname: str, cert: dict) -> bool:
        """
        Verify a certificate against pinned certificates.

        Args:
            hostname: The hostname that was connected to
            cert: The certificate information from the server

        Returns:
            True if the certificate matches a pin or no pins exist for the hostname

        Raises:
            SecurityError: If certificate pinning fails
        """
        if hostname not in self.pins or not self.pins[hostname]:
            return True

        # Extract the public key
        if not cert or "subject_public_key_info" not in cert:
            raise SecurityError("Certificate missing public key information")

        # Hash the public key info
        digest = hashlib.sha256(cert["subject_public_key_info"]).digest()
        pin_hash = f"sha256/{base64.b64encode(digest).decode('ascii')}"

        # Check if the hash matches any of our pins
        if pin_hash in self.pins[hostname]:
            return True

        raise SecurityError(
            f"Certificate pin verification failed for {hostname}. "
            f"Expected one of {self.pins[hostname]}, got {pin_hash}"
        )


class AdaptiveTimeoutManager:
    """
    Manages timeout settings adaptively based on host performance history.

    This class tracks request durations for different hosts and adjusts
    timeout settings accordingly to optimize reliability and performance.
    """

    def __init__(
        self,
        default_timeout: float = 30.0,
        min_timeout: float = 1.0,
        max_timeout: float = 120.0,
        persistence_enabled: bool = False,
        persistence_path: Optional[str] = None,
    ):
        """
        Initialize the adaptive timeout manager.

        Args:
            default_timeout: Default timeout in seconds for new hosts
            min_timeout: Minimum allowed timeout value
            max_timeout: Maximum allowed timeout value
            persistence_enabled: Whether to persist settings to disk
            persistence_path: Optional custom path for saving settings
        """
        self.default_timeout = default_timeout
        self.min_timeout = min_timeout
        self.max_timeout = max_timeout
        self.persistence_enabled = persistence_enabled

        # Host performance tracking
        self.host_metrics = {}

        # Statistical parameters
        self.percentile_threshold = 95  # Use 95th percentile for timeout calculation
        self.sample_size = 50  # Minimum samples needed for adaptation
        self.safety_factor = 1.5  # Multiply the percentile by this factor for safety

        # Initialize persistence if enabled
        if persistence_enabled:
            from .persistence import SettingsPersistence

            self.persistence = SettingsPersistence(storage_path=persistence_path)
            self._load_persisted_settings()
        else:
            self.persistence = None

    def _load_persisted_settings(self) -> None:
        """Load saved timeout settings from disk if available."""
        if not self.persistence_enabled or not self.persistence:
            return

        loaded_settings = self.persistence.load_timeout_settings()
        if loaded_settings:
            # Merge loaded settings with our current settings
            for host, metrics in loaded_settings.items():
                if host not in self.host_metrics:
                    self.host_metrics[host] = metrics

    def _save_settings(self) -> None:
        """Save current timeout settings to disk."""
        if not self.persistence_enabled or not self.persistence:
            return

        self.persistence.save_timeout_settings(self.host_metrics)

    def record_request(self, host: str, duration: float, success: bool):
        """
        Record the performance of a request to a given host.

        Args:
            host: Hostname of the request
            duration: Time taken for the request to complete (in seconds)
            success: Whether the request completed successfully
        """
        if host not in self.host_metrics:
            self.host_metrics[host] = {
                "durations": [],
                "timeout_history": [self.default_timeout],
                "successes": 0,
                "failures": 0,
                "timeouts": 0,
                "current_timeout": self.default_timeout,
            }

        metrics = self.host_metrics[host]

        # Update durations list (keep last 100 samples)
        metrics["durations"].append(duration)
        if len(metrics["durations"]) > 100:
            metrics["durations"].pop(0)

        # Update success/failure counts
        if success:
            metrics["successes"] += 1
        else:
            metrics["failures"] += 1

        # Periodically save settings if we have enough data
        if self.persistence_enabled and (metrics["successes"] + metrics["failures"]) % 20 == 0:
            self._save_settings()

    def record_timeout(self, host: str):
        """
        Record a timeout for a specific host.

        Args:
            host: Hostname that experienced the timeout
        """
        if host not in self.host_metrics:
            self.host_metrics[host] = {
                "durations": [],
                "timeout_history": [self.default_timeout],
                "successes": 0,
                "failures": 0,
                "timeouts": 0,
                "current_timeout": self.default_timeout,
            }

        self.host_metrics[host]["timeouts"] += 1

        # Save settings immediately after a timeout
        if self.persistence_enabled:
            self._save_settings()

    def get_timeout(self, host: str) -> float:
        """
        Get the optimal timeout for a given host based on historical performance.

        Args:
            host: Hostname to get timeout for

        Returns:
            Recommended timeout in seconds
        """
        # If we don't have data for this host, use default timeout
        if host not in self.host_metrics:
            return self.default_timeout

        metrics = self.host_metrics[host]

        # If we don't have enough samples, use current timeout
        if len(metrics["durations"]) < self.sample_size:
            return metrics["current_timeout"]

        # Calculate the percentile of request durations
        durations = sorted(metrics["durations"])
        idx = int(len(durations) * (self.percentile_threshold / 100))
        percentile_duration = durations[idx]

        # Calculate timeout based on percentile and safety factor
        calculated_timeout = percentile_duration * self.safety_factor

        # Factor in timeout history
        if metrics["timeouts"] > 0:
            # If we've had timeouts, be more conservative
            timeout_ratio = metrics["timeouts"] / (metrics["successes"] + 1)
            if timeout_ratio > 0.05:  # More than 5% of requests are timing out
                calculated_timeout *= 1 + timeout_ratio

        # Enforce minimum and maximum timeouts
        final_timeout = max(self.min_timeout, min(calculated_timeout, self.max_timeout))

        # Update the timeout history
        metrics["timeout_history"].append(final_timeout)
        if len(metrics["timeout_history"]) > 10:
            metrics["timeout_history"].pop(0)

        # Update current timeout (using exponential moving average to smooth changes)
        metrics["current_timeout"] = 0.7 * final_timeout + 0.3 * metrics["current_timeout"]

        # Periodically save settings when we calculate a new timeout
        if self.persistence_enabled and len(metrics["timeout_history"]) % 5 == 0:
            self._save_settings()

        return metrics["current_timeout"]

    def get_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics about managed timeouts.

        Returns:
            Dictionary with timeout statistics per host
        """
        stats = {}
        for host, metrics in self.host_metrics.items():
            if len(metrics["durations"]) > 0:
                stats[host] = {
                    "current_timeout": metrics["current_timeout"],
                    "avg_duration": sum(metrics["durations"]) / len(metrics["durations"]),
                    "max_duration": max(metrics["durations"]),
                    "min_duration": min(metrics["durations"]),
                    "samples": len(metrics["durations"]),
                    "success_rate": metrics["successes"]
                    / (metrics["successes"] + metrics["failures"] + 0.001),
                    "timeout_rate": metrics["timeouts"]
                    / (metrics["successes"] + metrics["failures"] + metrics["timeouts"] + 0.001),
                }
        return stats

    def save_settings(self) -> None:
        """Explicitly save current settings to disk."""
        if self.persistence_enabled:
            self._save_settings()


class ReqxClient:
    """
    A high-performance HTTP client with automatic protocol selection.

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
        max_retries: int = 3,
        retry_backoff: float = 0.5,
        http2: bool = False,
        enable_http3: bool = False,
        debug: bool = False,
        enable_cache: bool = False,
        cache_ttl: int = 300,
        rate_limit: Optional[float] = None,
        rate_limit_max_tokens: int = 60,
        certificate_pins: Dict[str, List[str]] | None = None,
        use_aiohttp: bool = True,
        adaptive_timeout: bool = False,  # Whether to use adaptive timeouts
        persistence_enabled: bool = False,  # Whether to persist settings
        persistence_path: Optional[str] = None,  # Custom path for persisted settings
        transport_learning: bool = False,  # Whether to enable intelligent transport selection
    ):
        """
        Initialize the ReqxClient with custom configuration.

        Args:
            base_url: Base URL for all requests
            headers: Default headers for all requests
            cookies: Default cookies for all requests
            timeout: Default timeout for all requests in seconds
            max_connections: Maximum number of connections
            max_keepalive_connections: Maximum number of idle keepalive connections
            keepalive_expiry: Keepalive connection expiry in seconds
            follow_redirects: Whether to follow redirects by default
            verify_ssl: Whether to verify SSL certificates by default
            max_retries: Maximum number of retries for failed requests
            retry_backoff: Exponential backoff factor for retries
            http2: Whether to use HTTP/2
            enable_http3: Whether to enable HTTP/3 (QUIC) if available
            debug: Whether to enable debug logging
            enable_cache: Whether to enable response caching
            cache_ttl: Default cache TTL in seconds
            rate_limit: Optional rate limit in requests per second
            rate_limit_max_tokens: Maximum tokens for rate limiting
            certificate_pins:
                Dictionary mapping hostnames to lists of base64-encoded SHA-256 certificate hashes
            use_aiohttp:
                Whether to use aiohttp for HTTP/1.1 requests (better performance)
            adaptive_timeout: Whether to dynamically adjust timeouts based on host performance
            persistence_enabled: Whether to persist settings
            persistence_path: Custom path for persisted settings
            transport_learning: Whether to enable intelligent transport selection
        """
        self.base_url = base_url
        self.default_headers = headers or {}
        self.default_cookies = cookies or {}
        self.timeout = timeout
        self.follow_redirects = follow_redirects
        self.verify_ssl = verify_ssl
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.debug = debug
        self.http3 = enable_http3
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl

        # Initialize cache if enabled
        self.cache: Dict[str, CacheEntry] = {}

        # Initialize rate limiter if enabled
        self.rate_limiter = None
        if rate_limit is not None and rate_limit > 0:
            self.rate_limiter = RateLimiter(rate_limit, rate_limit_max_tokens)

        # Initialize metrics
        self.metrics = {
            "requests_sent": 0,
            "request_errors": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "retry_attempts": 0,
            "total_request_time": 0.0,
            "rate_limit_wait_time": 0.0,
        }

        # Initialize middleware lists
        self.request_middlewares: List[RequestMiddleware] = []
        self.response_middlewares: List[ResponseMiddleware] = []

        if debug:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)

        # Create the appropriate transport based on configuration
        transport_args = {
            "base_url": base_url,
            "headers": headers,
            "cookies": cookies,
            "timeout": timeout,
            "max_connections": max_connections,
            "max_keepalive_connections": max_keepalive_connections,
            "keepalive_expiry": keepalive_expiry,
            "follow_redirects": follow_redirects,
            "verify_ssl": verify_ssl,
            "http2": http2,
            "enable_http3": enable_http3,
        }

        # Choose the transport implementation
        if http2 or enable_http3:
            # For HTTP/2 or HTTP/3, always use httpx
            self.transport = HttpxTransport(**transport_args)
        elif use_aiohttp:
            # Use the hybrid transport which will select the best option per request
            hybrid_args = {**transport_args, "transport_learning": transport_learning}
            self.transport = HybridTransport(**hybrid_args)
        else:
            # Use httpx for everything if specifically requested
            self.transport = HttpxTransport(**transport_args)

        # Initialize certificate pinner if pins are provided
        self.certificate_pinner = CertificatePinner(certificate_pins)

        # Initialize adaptive timeout manager if enabled
        self.adaptive_timeout = adaptive_timeout
        if adaptive_timeout:
            self.timeout_manager = AdaptiveTimeoutManager(
                default_timeout=timeout,
                persistence_enabled=persistence_enabled,
                persistence_path=persistence_path,
            )
        else:
            self.timeout_manager = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def _prepare_url(self, url: str) -> str:
        """Prepare the URL by joining it with the base_url if it's not absolute."""
        if not url.startswith(("http://", "https://")):
            return urljoin(self.base_url, url)
        return url

    def _get_cache_key(self, method: str, url: str, **kwargs) -> str:
        """Generate a unique cache key for a request."""
        # Create a unique cache key based on the request details
        key_parts = [method.upper(), url]

        # Add query params to the key
        if kwargs.get("params"):
            key_parts.append(str(sorted(kwargs["params"].items())))

        # Add request body to the key if it exists and method allows body
        if method.upper() in ("POST", "PUT", "PATCH") and (
            kwargs.get("json") or kwargs.get("data")
        ):
            body = kwargs.get("json") or kwargs.get("data") or ""
            if isinstance(body, dict):
                body = str(sorted(body.items()))
            key_parts.append(str(body))

        # Create a hash of the key parts
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    async def _get_from_cache(self, cache_key: str) -> Optional[Response]:
        """Get a response from the cache if it exists and is not expired."""
        if not self.enable_cache or cache_key not in self.cache:
            return None

        cache_entry = self.cache[cache_key]
        if cache_entry.is_expired():
            # Remove expired entry
            del self.cache[cache_key]
            return None

        self.metrics["cache_hits"] += 1
        # Create a new Response object with the same data
        return Response(
            status_code=cache_entry.response.status_code,
            headers=cache_entry.response.headers,
            content=cache_entry.response.content,
            request=cache_entry.response.request,
        )

    def _store_in_cache(
        self, cache_key: str, response: Response, ttl: Optional[int] = None
    ) -> None:
        """Store a response in the cache."""
        if not self.enable_cache:
            return

        # Only cache successful responses
        if response.status_code < 200 or response.status_code >= 300:
            return

        # Use specified TTL or default
        ttl = ttl if ttl is not None else self.cache_ttl
        expires_at = time.time() + ttl

        # copy the response to avoid modifying the original
        copied_response = Response(
            status_code=response.status_code,
            headers=response.headers,
            content=response.content,
            request=response.request,
        )

        # Create a copy of the response before caching
        self.cache[cache_key] = CacheEntry(copied_response, expires_at)
        self.metrics["cache_misses"] += 1

    def add_request_middleware(self, middleware: RequestMiddleware) -> None:
        """
        Add middleware to process requests before they are sent.

        Args:
            middleware: An async function that takes method, url, and kwargs and returns
                       modified kwargs for the request
        """
        self.request_middlewares.append(middleware)

    def add_response_middleware(self, middleware: ResponseMiddleware) -> None:
        """
        Add middleware to process responses after they are received.

        Args:
            middleware: An async function that takes a Response and returns a
                       potentially modified Response
        """
        self.response_middlewares.append(middleware)

    async def _apply_request_middlewares(
        self, method: str, url: str, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply all request middlewares in order."""
        current_kwargs = kwargs
        for middleware in self.request_middlewares:
            try:
                current_kwargs = await middleware(method, url, current_kwargs)
            except Exception as e:
                raise MiddlewareError(f"Request middleware error: {str(e)}") from e
        return current_kwargs

    async def _apply_response_middlewares(self, response: Response) -> Response:
        """Apply all response middlewares in order."""
        current_response = response
        for middleware in self.response_middlewares:
            try:
                current_response = await middleware(current_response)
            except Exception as e:
                raise MiddlewareError(f"Response middleware error: {str(e)}") from e
        return current_response

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Dict[str, str] | None = None,
        cookies: Dict[str, str] | None = None,
        params: Dict[str, Any] | None = None,
        json: Dict[str, Any] | None = None,
        data: Any | None = None,
        files: Dict[str, Any] | None = None,
        timeout: float | None = None,
        follow_redirects: bool | None = None,
        response_model: Type[T] | None = None,
        stream: bool = False,
        cache: bool | None = None,
        cache_ttl: int | None = None,
        force_http2: bool = False,
    ) -> Union[Response, T, None]:
        """
        Send an HTTP request with retry logic and proper error handling.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: URL to send the request to
            headers: Custom headers for this request
            cookies: Custom cookies for this request
            params: Query parameters
            json: JSON body for the request
            data: Form data or raw request body
            files: Files to upload
            timeout: Request timeout in seconds
            follow_redirects: Whether to follow redirects
            verify_ssl: Whether to verify SSL certificates
            response_model: Optional Pydantic model to parse the response into
            stream: Whether to enable streaming response (for large downloads)
            cache: Whether to use cache for this request (overrides client setting)
            cache_ttl: Custom TTL for this cached response in seconds
            force_http2: Force the use of HTTP/2 for this request

        Returns:
            Response object or parsed model instance if response_model is provided
        """
        start_time = time.time()
        self.metrics["requests_sent"] += 1

        # Extract host from URL for adaptive timeouts
        parsed_url = urlparse(url)
        host = parsed_url.netloc.split(":")[0]

        # Apply adaptive timeout if enabled and no specific timeout was provided
        if self.adaptive_timeout and self.timeout_manager and timeout is None:
            adaptive_timeout = self.timeout_manager.get_timeout(host)
            timeout = adaptive_timeout

        # Apply rate limiting if enabled
        if self.rate_limiter:
            wait_time = await self.rate_limiter.acquire()
            self.metrics["rate_limit_wait_time"] += wait_time

        # Determine if caching should be used for this request
        use_cache = cache if cache is not None else self.enable_cache

        # Check if the request is cacheable
        cacheable = use_cache and method.upper() == "GET" and not stream

        # Merge headers and cookies with defaults
        merged_headers = {**self.default_headers, **(headers or {})}
        merged_cookies = {**self.default_cookies, **(cookies or {})}

        # Use parameters if provided or fall back to defaults
        _timeout = Timeout(timeout or self.timeout)
        _follow_redirects = (
            follow_redirects if follow_redirects is not None else self.follow_redirects
        )

        # Prepare the URL
        full_url = self._prepare_url(url)

        if self.debug:
            log_request(full_url, method, merged_headers, json or data)

        # Prepare request kwargs
        kwargs = {
            "headers": merged_headers,
            "cookies": merged_cookies,
            "params": params,
            "json": json,
            "data": data,
            "files": files,
            "timeout": _timeout,
            "follow_redirects": _follow_redirects,
            "force_http2": force_http2,  # Pass through the HTTP/2 flag
        }

        # Generate cache key if the request is cacheable
        cache_key = None
        if cacheable:
            cache_key = self._get_cache_key(method, full_url, **kwargs)
            cached_response = await self._get_from_cache(cache_key)
            if cached_response:
                if self.debug:
                    logger.debug(f"Cache hit for {method} {full_url}")

                # Apply response middlewares to cached response
                cached_response = await self._apply_response_middlewares(cached_response)

                # Parse to response model if one was provided
                if response_model:
                    try:
                        json_data = cached_response.json()
                        # Use TypeAdapter for Pydantic v2 compatibility
                        adapter = TypeAdapter(response_model)
                        return adapter.validate_python(json_data)
                    except Exception as e:
                        raise ResponseError(
                            "Failed to parse cached response into model "
                            f"{response_model.__name__}: {str(e)}"
                        ) from e

                return cached_response

        # Apply request middlewares
        kwargs = await self._apply_request_middlewares(method, full_url, kwargs)

        # Initialize retry counter
        retry_count = 0

        # Retry loop
        while retry_count <= self.max_retries:
            request_start_time = time.time()
            try:
                # Make the request (with or without streaming)
                # Use our transport layer which handles HTTP protocol selection
                response = await self.transport.request(method=method, url=full_url, **kwargs)

                request_duration = time.time() - request_start_time

                # Record successful request for adaptive timeout
                if self.adaptive_timeout and self.timeout_manager:
                    self.timeout_manager.record_request(host, request_duration, success=True)

                if self.debug:
                    log_response(response)

                # Apply response middlewares
                response = await self._apply_response_middlewares(response)

                # Verify the server certificate if certificate pinning is enabled
                if self.certificate_pinner:
                    cert = response.extensions.get("cert")
                    if cert:
                        self.certificate_pinner.verify_certificate(response.url.host, cert)

                # Check if the response indicates an error
                response.raise_for_status()

                # Store in cache if cacheable and not streaming
                if cacheable and not stream and cache_key is not None:
                    self._store_in_cache(cache_key, response, cache_ttl)

                # Parse to response model if one was provided
                if response_model and not stream:
                    try:
                        json_data = response.json()
                        # Use TypeAdapter for Pydantic v2 compatibility
                        adapter = TypeAdapter(response_model)
                        return adapter.validate_python(json_data)
                    except Exception as e:
                        raise ResponseError(
                            "Failed to parse response into model"
                            f"{response_model.__name__}: {str(e)}"
                        ) from e

                # Update metrics
                self.metrics["total_request_time"] += time.time() - start_time
                return response

            except TransportError as e:
                # Check if it's a timeout
                is_timeout = "timeout" in str(e).lower() or "timed out" in str(e).lower()
                request_duration = time.time() - request_start_time

                # Record metrics for adaptive timeout
                if self.adaptive_timeout and self.timeout_manager:
                    if is_timeout:
                        self.timeout_manager.record_timeout(host)
                    else:
                        self.timeout_manager.record_request(host, request_duration, success=False)

                # Network-related errors are retryable
                retry_count += 1
                self.metrics["retry_attempts"] += 1
                if retry_count <= self.max_retries:
                    # Exponential backoff
                    wait_time = self.retry_backoff * (2 ** (retry_count - 1))
                    logger.debug(
                        "Request failed with error: "
                        f"{str(e)}. Retrying in {wait_time:.2f} seconds..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self.metrics["request_errors"] += 1
                    raise RequestError(
                        f"Request failed after {self.max_retries} retries: {str(e)}"
                    ) from e

            except Exception as e:
                # Other exceptions are not retried
                self.metrics["request_errors"] += 1
                if isinstance(e, ResponseError) or isinstance(e, MiddlewareError):
                    raise
                raise RequestError(f"Request failed: {str(e)}") from e

    # Convenience methods for common HTTP methods
    async def get(self, url: str, **kwargs) -> Union[Response, Any]:
        """Send a GET request."""
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> Union[Response, Any]:
        """Send a POST request."""
        return await self.request("POST", url, **kwargs)

    async def put(self, url: str, **kwargs) -> Union[Response, Any]:
        """Send a PUT request."""
        return await self.request("PUT", url, **kwargs)

    async def patch(self, url: str, **kwargs) -> Union[Response, Any]:
        """Send a PATCH request."""
        return await self.request("PATCH", url, **kwargs)

    async def delete(self, url: str, **kwargs) -> Union[Response, Any]:
        """Send a DELETE request."""
        return await self.request("DELETE", url, **kwargs)

    async def head(self, url: str, **kwargs) -> Union[Response, Any]:
        """Send a HEAD request."""
        return await self.request("HEAD", url, **kwargs)

    async def options(self, url: str, **kwargs) -> Union[Response, Any]:
        """Send an OPTIONS request."""
        return await self.request("OPTIONS", url, **kwargs)

    async def stream(self, method: str, url: str, **kwargs) -> Response | None:
        """
        Send a request and return a streaming response for handling large files.

        Example usage:
            async with client.stream('GET', 'https://example.com/large-file') as response:
                async for chunk in response.aiter_bytes():
                    # process chunk
        """
        kwargs["stream"] = True
        return await self.request(method, url, **kwargs)

    async def download_file(
        self, url: str, file_path: str, chunk_size: int = 8192, **kwargs
    ) -> None:
        """
        Download a file to the given path.

        Args:
            url: URL of the file to download
            file_path: Path to save the file to
            chunk_size: Size of chunks to download at a time
            **kwargs: Additional arguments to pass to the request method
        """

        # Get the streaming response
        response = await self.stream("GET", url, **kwargs)
        if response is None:
            raise RequestError(f"Failed to download file: {url}")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        # Open the file for writing
        with open(file_path, "wb") as f:
            # Download the file in chunks
            async for chunk in response.aiter_bytes(chunk_size):
                f.write(chunk)

    async def close(self):
        """Close the client session."""
        if hasattr(self, "transport"):
            await self.transport.close()

    def clear_cache(self, url_pattern: Optional[str] = None):
        """
        Clear the cache, optionally filtering by URL pattern.

        Args:
            url_pattern: Optional string pattern to match URLs to clear
        """
        if not self.enable_cache:
            return

        if url_pattern is None:
            # Clear entire cache
            self.cache.clear()
        else:
            # Clear only matching entries
            keys_to_remove = [
                key for key, entry in self.cache.items() if url_pattern in entry.response.url.path
            ]
            for key in keys_to_remove:
                del self.cache[key]

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get the current metrics for this client.

        Returns:
            Dictionary of metrics
        """
        # Calculate derived metrics
        metrics = dict(self.metrics)

        # Calculate success rate
        if metrics["requests_sent"] > 0:
            metrics["success_rate"] = (
                metrics["requests_sent"] - metrics["request_errors"]
            ) / metrics["requests_sent"]
        else:
            metrics["success_rate"] = 0

        # Calculate cache hit rate
        cache_requests = metrics["cache_hits"] + metrics["cache_misses"]
        if cache_requests > 0:
            metrics["cache_hit_rate"] = metrics["cache_hits"] / cache_requests
        else:
            metrics["cache_hit_rate"] = 0

        # Calculate average request time
        if metrics["requests_sent"] > 0:
            metrics["avg_request_time"] = metrics["total_request_time"] / metrics["requests_sent"]
        else:
            metrics["avg_request_time"] = 0

        return metrics

    async def batch(
        self,
        requests: List[Dict[str, Any]],
        max_concurrency: int = 10,
        raise_exceptions: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Union[Response, Exception, None]]:
        """
        Send multiple requests in parallel with controlled concurrency.

        Args:
            requests: List of request specifications, each containing at minimum 'method' and 'url',
                     and optionally other request parameters
            max_concurrency: Maximum number of concurrent requests
            raise_exceptions:
                Whether to raise the first exception encountered or return them as results
            progress_callback:
                Optional callback function that receives (completed, total) as arguments

        Returns:
            List of responses or exceptions in the same order as the requests
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        results: List[Optional[Union[Response, Exception]]] = [None] * len(requests)
        tasks = []
        completed = 0
        total = len(requests)

        async def _send_request(index: int, req_data: Dict[str, Any]):
            nonlocal completed

            async with semaphore:
                try:
                    method = req_data.pop("method", "GET")
                    url = req_data.pop("url")
                    response = await self.request(method, url, **req_data)
                    results[index] = response
                except Exception as e:
                    if raise_exceptions:
                        raise BatchRequestError(index, req_data, e) from e
                    results[index] = e

                completed += 1
                if progress_callback:
                    progress_callback(completed, total)

        # Create tasks for each request
        for i, req_data in enumerate(requests):
            tasks.append(asyncio.create_task(_send_request(i, req_data.copy())))

        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=not raise_exceptions)
        return results

    async def batch_with_model(
        self,
        requests: List[Dict[str, Any]],
        response_model: Type[T],
        max_concurrency: int = 10,
        raise_exceptions: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Union[T, Exception]]:
        """
        Send multiple requests in parallel and parse responses into the given model.

        Args:
            requests: List of request specifications
            response_model: Pydantic model to parse responses into
            max_concurrency: Maximum number of concurrent requests
            raise_exceptions: Whether to raise exceptions or return them as results
            progress_callback:
                Optional callback function that receives (completed, total) as arguments

        Returns:
            List of parsed model instances or exceptions
        """
        responses = await self.batch(requests, max_concurrency, raise_exceptions, progress_callback)
        results = []
        adapter = TypeAdapter(response_model)

        for response in responses:
            if isinstance(response, Exception):
                results.append(response)
            else:
                try:
                    if response is None:
                        results.append(None)
                        continue

                    json_data = response.json()
                    results.append(adapter.validate_python(json_data))
                except Exception as e:
                    if raise_exceptions:
                        raise ResponseError(
                            "Failed to parse response into model "
                            f"{response_model.__name__}: {str(e)}"
                        ) from e
                    results.append(e)

        return results

    async def parallel_get(
        self, urls: List[str], max_concurrency: int = 10, **kwargs
    ) -> List[Union[Response, Exception, None]]:
        """
        Convenience method to send multiple GET requests in parallel.

        Args:
            urls: List of URLs to request
            max_concurrency: Maximum number of concurrent requests
            **kwargs: Additional arguments to pass to all requests

        Returns:
            List of responses in the same order as the URLs
        """
        requests = [{"method": "GET", "url": url, **kwargs} for url in urls]
        return await self.batch(requests, max_concurrency=max_concurrency)

    async def batch_request(
        self,
        requests: List[Dict[str, Any]],
        concurrency_limit: int = 10,
        response_model: Type[T] | None = None,
        raise_exceptions: bool = False,
    ) -> List[Union[T, Response, Exception, None]]:
        """
        Execute multiple requests in parallel with concurrency control.

        Args:
            requests: List of request configurations, each a dict with keys:
                     method, url, and optional kwargs like headers, json, etc.
            concurrency_limit: Maximum number of concurrent requests
            response_model: Optional Pydantic model to parse responses into
            raise_exceptions: Whether to raise exceptions or return them in the results list

        Returns:
            List of responses or exceptions in the same order as the requests
        """
        semaphore = asyncio.Semaphore(concurrency_limit)
        results: List[None | Response | Exception | Any] = [None] * len(
            requests
        )  # Pre-allocate the results list

        async def _process_request(index: int, req_config: dict):
            req_method = req_config.pop("method", "GET")
            req_url = req_config.pop("url")

            try:
                # Use the same URL object if possible to reduce memory
                if isinstance(req_url, str) and req_url.startswith(("http://", "https://")):
                    full_url = req_url
                else:
                    full_url = self._prepare_url(req_url)

                # Apply request middleware with minimal dict copying
                kwargs = await self._apply_request_middlewares(req_method, full_url, req_config)

                async with semaphore:
                    # Acquire rate limiting tokens if needed
                    if self.rate_limiter:
                        await self.rate_limiter.acquire()

                    # Make the actual request using our transport layer
                    response = await self.transport.request(
                        method=req_method, url=full_url, **kwargs
                    )

                    # Process response middleware
                    if self.response_middlewares:
                        response = await self._apply_response_middlewares(response)

                    # Store the response directly in the results list
                    results[index] = response
            except Exception as e:
                if raise_exceptions:
                    raise e
                results[index] = e

        # Create tasks for all requests
        tasks = []
        for i, req in enumerate(requests):
            # Make a shallow copy to avoid modifying the original
            req_copy = {k: v for k, v in req.items()}
            tasks.append(_process_request(i, req_copy))

        # Execute all requests with concurrency control
        await asyncio.gather(*tasks)

        # Apply response model if provided
        if response_model:
            adapter = TypeAdapter(response_model)
            for i, result in enumerate(results):
                if isinstance(result, Response):
                    try:
                        json_data = result.json()
                        # Replace the response with the model instance
                        results[i] = adapter.validate_python(json_data)
                    except Exception as e:
                        if raise_exceptions:
                            raise ResponseError(
                                f"Failed to parse response into model {response_model.__name__}: "
                                f"{str(e)}"
                            ) from e
                        results[i] = e

        return results

    async def batch_get(
        self,
        urls: List[str],
        concurrency_limit: int = 10,
        response_model: Type[T] | None = None,
        **kwargs,
    ) -> List[Union[T, Response, Exception, None]]:
        """
        Execute multiple GET requests in parallel.

        Args:
            urls: List of URLs to send GET requests to
            concurrency_limit: Maximum number of concurrent requests
            response_model: Optional Pydantic model to parse responses into
            **kwargs: Additional kwargs to apply to all requests

        Returns:
            List of responses in the same order as the URLs
        """
        requests = [{"method": "GET", "url": url, **kwargs} for url in urls]
        return await self.batch_request(requests, concurrency_limit, response_model)

    def get_timeout_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about adaptive timeouts if enabled.

        Returns:
            Dictionary with timeout statistics or empty dict if adaptive timeouts are disabled
        """
        if self.adaptive_timeout and self.timeout_manager:
            return self.timeout_manager.get_statistics()
        return {}

    def get_transport_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about transport usage and performance.

        Returns:
            Dictionary with transport metrics or empty dict if the transport doesn't support metrics
        """
        if hasattr(self.transport, "get_metrics"):
            return self.transport.get_metrics()
        return {}
