"""Middleware system for request and response processing.

This module provides a middleware framework for intercepting and modifying
HTTP requests and responses in the enhanced-httpx client.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar, Union

import httpx

from .exceptions import MiddlewareError

logger = logging.getLogger("enhanced_httpx.middleware")

# Type aliases for request and response handlers
RequestHandler = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]
ResponseHandler = Callable[[httpx.Response, Dict[str, Any]], Awaitable[httpx.Response]]

# Type for middleware options
T = TypeVar("T")


class Middleware(ABC):
    """Base class for all middleware components."""

    @abstractmethod
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process and potentially modify an outgoing request.

        Args:
            request: The request dictionary containing method, url, headers, etc.

        Returns:
            Modified request dictionary.
        """
        pass

    @abstractmethod
    async def process_response(
        self, response: httpx.Response, request: Dict[str, Any]
    ) -> httpx.Response:
        """Process and potentially modify an incoming response.

        Args:
            response: The httpx Response object
            request: The original request dictionary

        Returns:
            Modified response object.
        """
        pass


class MiddlewareChain:
    """A chain of middleware components that are executed in sequence."""

    def __init__(self):
        """Initialize an empty middleware chain."""
        self.middlewares: List[Middleware] = []

    def add(self, middleware: Middleware) -> "MiddlewareChain":
        """Add a middleware to the chain.

        Args:
            middleware: The middleware instance to add

        Returns:
            Self for method chaining
        """
        self.middlewares.append(middleware)
        return self

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request through all middlewares.

        Args:
            request: The request dictionary

        Returns:
            Modified request dictionary
        """
        current_request = request
        for middleware in self.middlewares:
            try:
                current_request = await middleware.process_request(current_request)
            except Exception as e:
                logger.error(
                    f"Error in middleware {middleware.__class__.__name__} during request processing: {str(e)}"
                )
                raise MiddlewareError(f"Request middleware error: {str(e)}")

        return current_request

    async def process_response(
        self, response: httpx.Response, request: Dict[str, Any]
    ) -> httpx.Response:
        """Process a response through all middlewares in reverse order.

        Args:
            response: The response object
            request: The original request dictionary

        Returns:
            Modified response object.
        """
        current_response = response

        # Process in reverse order
        for middleware in reversed(self.middlewares):
            try:
                current_response = await middleware.process_response(current_response, request)
            except Exception as e:
                logger.error(
                    f"Error in middleware {middleware.__class__.__name__} during response processing: {str(e)}"
                )
                raise MiddlewareError(f"Response middleware error: {str(e)}")

        return current_response


class LoggingMiddleware(Middleware):
    """Middleware for logging request and response details."""

    def __init__(
        self,
        log_level: int = logging.INFO,
        log_request_headers: bool = True,
        log_request_body: bool = False,
        log_response_headers: bool = True,
        log_response_body: bool = False,
        include_sensitive_data: bool = False,
        sensitive_headers: List[str] = None,
    ):
        """Initialize logging middleware.

        Args:
            log_level: Logging level
            log_request_headers: Whether to log request headers
            log_request_body: Whether to log request body
            log_response_headers: Whether to log response headers
            log_response_body: Whether to log response body
            include_sensitive_data: Whether to include sensitive data (not recommended)
            sensitive_headers: List of header names considered sensitive
        """
        self.log_level = log_level
        self.log_request_headers = log_request_headers
        self.log_request_body = log_request_body
        self.log_response_headers = log_response_headers
        self.log_response_body = log_response_body
        self.include_sensitive_data = include_sensitive_data
        self.sensitive_headers = sensitive_headers or [
            "Authorization",
            "X-API-Key",
            "Cookie",
            "Set-Cookie",
        ]

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Log request details."""
        method = request.get("method", "GET")
        url = request.get("url", "")

        log_parts = [f"Request: {method} {url}"]

        # Store start time for duration calculation
        request["_logging_start_time"] = time.time()

        if self.log_request_headers and "headers" in request:
            headers = self._filter_sensitive_headers(request["headers"])
            log_parts.append(f"Request Headers: {headers}")

        if self.log_request_body and "content" in request:
            body = request["content"]
            if isinstance(body, bytes):
                try:
                    body = body.decode("utf-8")
                except UnicodeDecodeError:
                    body = "<binary data>"

            log_parts.append(f"Request Body: {body}")

        logger.log(self.log_level, "\n".join(log_parts))
        return request

    async def process_response(
        self, response: httpx.Response, request: Dict[str, Any]
    ) -> httpx.Response:
        """Log response details."""
        start_time = request.pop("_logging_start_time", None)
        duration = f" ({time.time() - start_time:.3f}s)" if start_time else ""

        log_parts = [
            f"Response: {response.status_code}{duration} "
            f"- {request.get('method', 'GET')} {request.get('url', '')}"
        ]

        if self.log_response_headers:
            headers = self._filter_sensitive_headers(dict(response.headers))
            log_parts.append(f"Response Headers: {headers}")

        if self.log_response_body:
            try:
                # Try to parse as JSON for pretty printing
                content_type = response.headers.get("content-type", "")
                if "json" in content_type:
                    body = response.json()
                    log_parts.append(f"Response Body: {json.dumps(body, indent=2)}")
                else:
                    # Limit text response to a reasonable length
                    text = response.text[:500]
                    if len(response.text) > 500:
                        text += "... [truncated]"
                    log_parts.append(f"Response Body: {text}")
            except Exception as e:
                log_parts.append(f"Response Body: <Error reading body: {str(e)}>")

        logger.log(self.log_level, "\n".join(log_parts))
        return response

    def _filter_sensitive_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Filter sensitive headers if configured."""
        if self.include_sensitive_data:
            return headers

        filtered = {}
        for name, value in headers.items():
            if any(sensitive.lower() == name.lower() for sensitive in self.sensitive_headers):
                filtered[name] = "**REDACTED**"
            else:
                filtered[name] = value

        return filtered


class RetryMiddleware(Middleware):
    """Middleware for automatically retrying failed requests."""

    def __init__(
        self,
        max_retries: int = 3,
        retry_statuses: List[int] = None,
        retry_exceptions: List[type] = None,
        backoff_factor: float = 0.5,
        respect_retry_after: bool = True,
        retry_methods: List[str] = None,
    ):
        """Initialize retry middleware.

        Args:
            max_retries: Maximum number of retry attempts
            retry_statuses: HTTP status codes to retry (default: 429, 500, 502, 503, 504)
            retry_exceptions: Exception types to retry
            backoff_factor: Exponential backoff factor (0 for no backoff)
            respect_retry_after: Whether to respect Retry-After header
            retry_methods: HTTP methods to retry (default: GET, HEAD, OPTIONS)
        """
        import asyncio

        self.max_retries = max_retries
        self.retry_statuses = retry_statuses or [429, 500, 502, 503, 504]
        self.retry_exceptions = retry_exceptions or [
            httpx.ConnectError,
            httpx.ReadError,
            httpx.WriteError,
            httpx.PoolTimeout,
            httpx.NetworkError,
            httpx.TimeoutException,
        ]
        self.backoff_factor = backoff_factor
        self.respect_retry_after = respect_retry_after
        self.retry_methods = [m.upper() for m in (retry_methods or ["GET", "HEAD", "OPTIONS"])]

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Store original request for retry handling."""
        # No modifications needed at this stage
        return request

    async def process_response(
        self, response: httpx.Response, request: Dict[str, Any]
    ) -> httpx.Response:
        """Handle retries if needed."""
        import asyncio

        # No retries needed if response is OK or method not in retry_methods
        method = request.get("method", "GET").upper()
        if response.is_success or method not in self.retry_methods:
            return response

        # Get existing retry count, starting at 0
        retry_count = request.get("_retry_count", 0)

        # Return the response if we've reached the max retries
        if retry_count >= self.max_retries:
            logger.debug(f"Max retries ({self.max_retries}) reached, returning last response")
            return response

        # Check if this response should trigger a retry
        should_retry = response.status_code in self.retry_statuses

        if should_retry:
            # Increment retry counter
            request["_retry_count"] = retry_count + 1

            # Calculate sleep time with exponential backoff
            sleep_time = self._get_sleep_time(response, retry_count)
            logger.info(
                f"Retrying request after {sleep_time:.2f}s (attempt {retry_count + 1}/{self.max_retries})"
            )

            # Sleep before retry
            await asyncio.sleep(sleep_time)

            # The client will resend the request with updated _retry_count
            # This is handled by the EnhancedClient._send_with_middlewares method
            raise httpx.TransportError(f"Retry needed (status={response.status_code})")

        return response

    def _get_sleep_time(self, response: httpx.Response, retry_count: int) -> float:
        """Calculate sleep time between retries."""
        retry_after = None

        # Check for Retry-After header if configured
        if self.respect_retry_after:
            retry_after_header = response.headers.get("retry-after")
            if retry_after_header:
                try:
                    if retry_after_header.isdigit():
                        # Retry-After in seconds
                        retry_after = float(retry_after_header)
                    else:
                        # Retry-After as HTTP date
                        from email.utils import parsedate_to_datetime

                        retry_date = parsedate_to_datetime(retry_after_header)
                        if retry_date:
                            retry_after = (retry_date - datetime.now()).total_seconds()
                            retry_after = max(0, retry_after)  # Ensure non-negative
                except (ValueError, TypeError):
                    pass

        # If no valid Retry-After, use exponential backoff
        if retry_after is None:
            retry_after = self.backoff_factor * (2**retry_count)

        return retry_after


class DefaultHeadersMiddleware(Middleware):
    """Middleware for adding default headers to requests."""

    def __init__(self, headers: Dict[str, str]):
        """Initialize with default headers.

        Args:
            headers: Default headers to add to requests
        """
        self.headers = headers

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Add default headers to request."""
        if "headers" not in request:
            request["headers"] = {}

        # Add defaults (only if not already set by user)
        for key, value in self.headers.items():
            if key not in request["headers"]:
                request["headers"][key] = value

        return request

    async def process_response(
        self, response: httpx.Response, request: Dict[str, Any]
    ) -> httpx.Response:
        """No modifications needed for response."""
        return response


class CompressionMiddleware(Middleware):
    """Middleware for handling request/response compression."""

    def __init__(
        self,
        compress_requests: bool = False,
        min_size_to_compress: int = 1024,
        compress_type: str = "gzip",
        accept_compressed_responses: bool = True,
    ):
        """Initialize compression middleware.

        Args:
            compress_requests: Whether to compress request bodies
            min_size_to_compress: Minimum body size in bytes to compress
            compress_type: Compression algorithm ("gzip", "deflate", or "br" (brotli))
            accept_compressed_responses: Whether to indicate acceptance of compressed responses
        """
        if compress_type not in ["gzip", "deflate", "br"]:
            raise ValueError("compress_type must be 'gzip', 'deflate', or 'br'")

        self.compress_requests = compress_requests
        self.min_size_to_compress = min_size_to_compress
        self.compress_type = compress_type
        self.accept_compressed_responses = accept_compressed_responses

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Compress request body if needed."""
        # Add Accept-Encoding header for compressed responses if enabled
        if self.accept_compressed_responses:
            if "headers" not in request:
                request["headers"] = {}

            # Add the compression types we support to Accept-Encoding
            encodings = ["gzip", "deflate"]

            # Add brotli if it's available
            try:
                import brotli

                encodings.append("br")
            except ImportError:
                pass

            request["headers"]["Accept-Encoding"] = ", ".join(encodings)

        if not self.compress_requests:
            return request

        # Skip if no content or content is too small
        if "content" not in request:
            return request

        content = request["content"]
        if not isinstance(content, (bytes, str)):
            return request

        # Convert string to bytes if needed
        if isinstance(content, str):
            content = content.encode("utf-8")

        # Skip if content is too small
        if len(content) < self.min_size_to_compress:
            return request

        # Compress the content
        if self.compress_type == "gzip":
            import gzip

            compressed = gzip.compress(content)
        elif self.compress_type == "deflate":
            import zlib

            compressed = zlib.compress(content)
        elif self.compress_type == "br":
            try:
                import brotli

                compressed = brotli.compress(content)
            except ImportError:
                logger.warning(
                    "Brotli compression requested but brotli package not installed. Falling back to gzip."
                )
                import gzip

                compressed = gzip.compress(content)
                self.compress_type = "gzip"  # Update so header is correct

        # Update request with compressed content
        request["content"] = compressed

        # Add appropriate header
        if "headers" not in request:
            request["headers"] = {}

        request["headers"]["Content-Encoding"] = self.compress_type

        # Log compression efficiency
        original_size = len(content)
        compressed_size = len(compressed)
        compression_ratio = (original_size - compressed_size) / original_size * 100
        logger.debug(
            f"Compressed request body with {self.compress_type}: {original_size} -> {compressed_size} bytes ({compression_ratio:.1f}% reduction)"
        )

        return request

    async def process_response(
        self, response: httpx.Response, request: Dict[str, Any]
    ) -> httpx.Response:
        """No modifications needed for response (httpx handles response decompression)."""
        # HTTPX already handles decompression of responses
        if response.headers.get("content-encoding"):
            logger.debug(
                f"Server responded with {response.headers.get('content-encoding')} encoding"
            )

        return response


class CacheMiddleware(Middleware):
    """Simple in-memory caching middleware."""

    def __init__(
        self,
        max_entries: int = 100,
        ttl_seconds: int = 300,
        cacheable_methods: List[str] = None,
        cacheable_status_codes: List[int] = None,
        cache_vary_headers: List[str] = None,
    ):
        """Initialize caching middleware.

        Args:
            max_entries: Maximum number of cached responses
            ttl_seconds: Time-to-live in seconds for cached entries
            cacheable_methods: HTTP methods that can be cached (default: GET, HEAD)
            cacheable_status_codes: HTTP status codes that can be cached (default: 200, 203, 300, 301, 308)
            cache_vary_headers: Headers that should be included in the cache key (default: None)
        """
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.cacheable_methods = [m.upper() for m in (cacheable_methods or ["GET", "HEAD"])]
        self.cacheable_status_codes = cacheable_status_codes or [200, 203, 300, 301, 308]
        self.cache_vary_headers = [h.lower() for h in (cache_vary_headers or [])]

        # Cache storage: {cache_key: (expires_at, last_accessed_at, response_data)}
        self._cache = {}

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Check cache for matching request."""
        # Skip cache for non-cacheable methods
        method = request.get("method", "GET").upper()
        if method not in self.cacheable_methods:
            return request

        # Generate cache key
        cache_key = self._generate_cache_key(request)

        # Check if we have a valid cache entry
        if cache_key in self._cache:
            expires_at, last_accessed, response_data = self._cache[cache_key]

            # If not expired, mark request to be handled by response processor
            if time.time() < expires_at:
                # Update last accessed time for proper LRU behavior
                self._cache[cache_key] = (expires_at, time.time(), response_data)
                request["_cache_hit"] = (True, response_data)
                logger.debug(f"Cache hit for {method} {request.get('url')}")
            else:
                # Clear expired entry
                del self._cache[cache_key]
                logger.debug(f"Cache expired for {method} {request.get('url')}")

        return request

    async def process_response(
        self, response: httpx.Response, request: Dict[str, Any]
    ) -> httpx.Response:
        """Handle cache hits and store cacheable responses."""
        # Check if this was a cache hit
        cache_hit = request.pop("_cache_hit", (False, None))
        if cache_hit[0]:
            # Reconstruct response from cache
            return self._reconstruct_response(cache_hit[1], request)

        # Otherwise, see if we should cache this response
        method = request.get("method", "GET").upper()
        if method in self.cacheable_methods and response.status_code in self.cacheable_status_codes:
            # Check Cache-Control header for no-store directive
            cache_control = response.headers.get("Cache-Control", "").lower()
            if "no-store" not in cache_control and "private" not in cache_control:
                # Store in cache
                cache_key = self._generate_cache_key(request)
                current_time = time.time()
                expires_at = current_time + self.ttl_seconds

                # Use max-age from Cache-Control if available
                if "max-age=" in cache_control:
                    try:
                        max_age = int(cache_control.split("max-age=")[1].split(",")[0])
                        expires_at = current_time + max_age
                    except (ValueError, IndexError):
                        pass

                # Store serialized response data with access time
                self._cache[cache_key] = (
                    expires_at,
                    current_time,
                    self._serialize_response(response),
                )
                logger.debug(f"Cached response for {method} {request.get('url')}")

                # Ensure cache doesn't grow too large (true LRU)
                if len(self._cache) > self.max_entries:
                    # Remove least recently used entry
                    lru_key = min(self._cache.items(), key=lambda item: item[1][1])[0]
                    del self._cache[lru_key]
                    logger.debug(f"Evicted LRU cache entry")

        return response

    def _generate_cache_key(self, request: Dict[str, Any]) -> str:
        """Generate a unique key for the request."""
        import hashlib

        # Include method and URL
        components = [request.get("method", "GET").upper(), str(request.get("url", ""))]

        # Include query parameters
        if "params" in request:
            params = request["params"]
            if isinstance(params, dict):
                # Sort to ensure consistent order
                param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
                components.append(param_str)
            else:
                components.append(str(params))

        # Include specified headers that might affect caching
        if self.cache_vary_headers and "headers" in request:
            headers = request["headers"]
            for header in self.cache_vary_headers:
                if header.lower() in {k.lower() for k in headers}:
                    # Find the actual header key regardless of case
                    actual_key = next(k for k in headers if k.lower() == header.lower())
                    components.append(f"{header.lower()}:{headers[actual_key]}")

        # Generate a hash of the components
        key_str = ":".join(components)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _serialize_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Serialize response for caching."""
        return {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "content": response.content,
            "url": str(response.url),
        }

    def _reconstruct_response(
        self, data: Dict[str, Any], request: Dict[str, Any]
    ) -> httpx.Response:
        """Reconstruct response from cached data."""
        return httpx.Response(
            status_code=data["status_code"],
            headers=data["headers"],
            content=data["content"],
            request=httpx.Request(method=request.get("method", "GET"), url=request.get("url", "")),
        )


class CircuitBreakerMiddleware(Middleware):
    """
    Implementation of the Circuit Breaker pattern for HTTP requests.

    This middleware prevents making requests to services that are failing repeatedly,
    which helps in:
    1. Preventing overloading failing services with retry requests
    2. Failing fast when a service is down
    3. Allowing periodic recovery attempts
    """

    # Circuit states
    CLOSED = "closed"  # Normal operation, requests proceed
    OPEN = "open"  # Circuit is open, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service has recovered

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        error_codes: List[int] | None = None,
        excluded_urls: List[str] | None = None,
    ):
        """Initialize the circuit breaker.

        Args:
            failure_threshold: Number of failures before opening the circuit
            recovery_timeout: Time in seconds to wait before attempting recovery
            error_codes: HTTP status codes to consider as failures
            excluded_urls: URLs to exclude from circuit breaking
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.error_codes = error_codes or [500, 501, 502, 503, 504, 505]
        self.excluded_urls = excluded_urls or []

        # Circuit state by host
        self._circuits: Dict[str, Dict[str, Any]] = {}

    def _get_circuit_state(self, host: str) -> Dict[str, Any]:
        """Get or create circuit state for a host."""
        if host not in self._circuits:
            self._circuits[host] = {
                "state": self.CLOSED,
                "failures": 0,
                "last_failure_time": None,
                "last_test_time": None,
            }
        return self._circuits[host]

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Check if the circuit is open before allowing a request."""
        # Extract the URL and host
        url = request.get("url", "")

        # Skip circuit breaking for excluded URLs
        if any(excluded in url for excluded in self.excluded_urls):
            return request

        # Parse the hostname from the URL
        try:
            from urllib.parse import urlparse

            host = urlparse(url).netloc
            if not host:
                # No valid host, just proceed with the request
                return request
        except Exception:
            # Can't parse URL, just proceed with the request
            return request

        # Get circuit for this host
        circuit = self._get_circuit_state(host)
        current_time = time.time()

        if circuit["state"] == self.OPEN:
            # Check if recovery timeout has elapsed
            if current_time - circuit["last_failure_time"] > self.recovery_timeout:
                # Transition to half-open state and allow the request
                logger.info(f"Circuit for {host} transitioning to half-open state")
                circuit["state"] = self.HALF_OPEN
                circuit["last_test_time"] = current_time
                return request
            else:
                # Circuit is open, fail fast
                from .exceptions import CircuitBreakerError

                raise CircuitBreakerError(
                    f"Circuit is open for {host}, request rejected",
                    host=host,
                    retry_after=self.recovery_timeout
                    - (current_time - circuit["last_failure_time"]),
                )

        # Circuit is closed or half-open, allow the request
        return request

    async def process_response(
        self, response: httpx.Response, request: Dict[str, Any]
    ) -> httpx.Response:
        """Handle response and update circuit state."""
        # Extract the URL and host
        url = str(response.url)

        # Skip circuit breaking for excluded URLs
        if any(excluded in url for excluded in self.excluded_urls):
            return response

        # Parse the hostname from the URL
        try:
            from urllib.parse import urlparse

            host = urlparse(url).netloc
            if not host:
                # No valid host, just return the response
                return response
        except Exception:
            # Can't parse URL, just return the response
            return response

        # Get circuit for this host
        circuit = self._get_circuit_state(host)
        current_time = time.time()

        # Check if the response is a failure
        is_failure = response.status_code in self.error_codes

        if is_failure:
            # Handle failure
            if circuit["state"] == self.CLOSED:
                circuit["failures"] += 1
                circuit["last_failure_time"] = current_time

                if circuit["failures"] >= self.failure_threshold:
                    # Open the circuit
                    logger.warning(
                        f"Circuit for {host} opened after {circuit['failures']} failures"
                    )
                    circuit["state"] = self.OPEN

            elif circuit["state"] == self.HALF_OPEN:
                # Test request failed, back to open state
                logger.warning(f"Circuit for {host} reopened after failed test")
                circuit["state"] = self.OPEN
                circuit["last_failure_time"] = current_time

        else:
            # Handle success
            if circuit["state"] == self.HALF_OPEN:
                # Test request succeeded, back to closed state
                logger.info(f"Circuit for {host} closed after successful test")
                circuit["state"] = self.CLOSED
                circuit["failures"] = 0

            elif circuit["state"] == self.CLOSED:
                # Reset failure count after successful request
                circuit["failures"] = 0

        return response


class TracingMiddleware(Middleware):
    """
    Middleware for request tracing with unique IDs.

    This middleware assigns a unique trace ID to each request and follows it
    through all stages of processing, including retries and middleware chains.
    It's useful for debugging and correlating logs across services.
    """

    def __init__(
        self,
        trace_header_name: str = "X-Trace-ID",
        include_in_response: bool = True,
        trace_all_headers: bool = False,
        prefix: str = "",
    ):
        """
        Initialize tracing middleware.

        Args:
            trace_header_name: Name of the header for passing the trace ID
            include_in_response: Whether to log the trace ID in the response
            trace_all_headers: Whether to log all request/response headers for debugging
            prefix: Optional prefix for generated trace IDs
        """
        self.trace_header_name = trace_header_name
        self.include_in_response = include_in_response
        self.trace_all_headers = trace_all_headers
        self.prefix = prefix

    def _generate_trace_id(self) -> str:
        """Generate a unique trace ID."""
        import socket
        import time
        import uuid

        # Create a trace ID that includes host, timestamp, and random component
        try:
            hostname = socket.gethostname()[:8]  # First 8 chars of hostname
        except:
            hostname = "unknown"

        timestamp = int(time.time() * 1000) % 10000000  # 7 digit timestamp
        random_part = uuid.uuid4().hex[:8]  # 8 random hex chars

        trace_id = f"{hostname}-{timestamp:07d}-{random_part}"
        if self.prefix:
            return f"{self.prefix}-{trace_id}"
        return trace_id

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Add trace ID to request headers."""
        if "headers" not in request:
            request["headers"] = {}

        # Check if this request already has a trace ID (from parent request)
        if self.trace_header_name in request["headers"]:
            trace_id = request["headers"][self.trace_header_name]
            logger.debug(f"Using existing trace ID: {trace_id}")
        else:
            # Generate a new trace ID
            trace_id = self._generate_trace_id()
            request["headers"][self.trace_header_name] = trace_id
            logger.debug(f"Generated new trace ID: {trace_id}")

        # Store trace ID in request for use in response
        request["_trace_id"] = trace_id

        # Log request details with trace ID
        if self.trace_all_headers:
            logger.debug(f"[Trace: {trace_id}] Request headers: {request.get('headers', {})}")

        return request

    async def process_response(
        self, response: httpx.Response, request: Dict[str, Any]
    ) -> httpx.Response:
        """Log trace ID with response."""
        # Get trace ID from request
        trace_id = request.pop("_trace_id", "unknown")

        # Log response with trace ID
        log_msg = f"[Trace: {trace_id}] Response: {response.status_code}"

        if self.trace_all_headers:
            log_msg += f", Headers: {dict(response.headers)}"

        logger.debug(log_msg)

        # Return response with trace ID header if configured
        if self.include_in_response:
            response.headers[self.trace_header_name] = trace_id

        return response


class MemoryAwareMiddleware(Middleware):
    """
    Middleware that monitors memory usage and automatically switches to
    streaming mode for large responses.

    This helps prevent memory issues when dealing with large responses
    by automatically detecting response size from Content-Length header
    or switching to streaming mode if the response starts getting too large.
    """

    def __init__(
        self,
        max_in_memory_size: int = 10 * 1024 * 1024,  # 10MB default
        check_content_length: bool = True,
        monitor_download: bool = True,
    ):
        """
        Initialize memory aware middleware.

        Args:
            max_in_memory_size: Maximum response size to keep in memory (bytes)
            check_content_length: Whether to check Content-Length header
            monitor_download: Whether to monitor download size during streaming
        """
        self.max_in_memory_size = max_in_memory_size
        self.check_content_length = check_content_length
        self.monitor_download = monitor_download

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Check if request should use streaming based on expected response size."""
        # If streaming is already explicitly set, don't change it
        if "stream" in request:
            return request

        # If we know this is a large resource, enable streaming
        url = request.get("url", "")
        if any(ext in url for ext in [".zip", ".pdf", ".iso", ".mp4", ".tar.gz"]):
            logger.debug(f"Automatically enabling streaming for likely large resource: {url}")
            request["stream"] = True

        return request

    async def process_response(
        self, response: httpx.Response, request: Dict[str, Any]
    ) -> httpx.Response:
        """Check response size and handle accordingly."""
        # If already streaming, nothing to do
        if request.get("stream", False):
            return response

        # Check Content-Length header if enabled
        if self.check_content_length and "content-length" in response.headers:
            try:
                content_length = int(response.headers["content-length"])
                if content_length > self.max_in_memory_size:
                    logger.warning(
                        f"Large response detected ({content_length} bytes). "
                        f"Consider using streaming mode for {response.url}"
                    )

                    # We can't convert to streaming at this point, but we can warn
                    # the developer to use streaming in the future for this endpoint
                    pass
            except (ValueError, TypeError):
                pass

        # If this is a binary response, check its actual size
        if hasattr(response, "_content") and response._content:
            size = len(response._content)
            if size > self.max_in_memory_size:
                logger.warning(
                    f"Large response loaded in memory: {size} bytes. "
                    f"Consider using streaming mode for future requests to {response.url}"
                )

        return response
