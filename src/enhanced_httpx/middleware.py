"""Middleware system for request and response processing.

This module provides a middleware framework for intercepting and modifying
HTTP requests and responses in the enhanced-httpx client.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, Awaitable, List, Union, TypeVar
import logging
import json
import time
from datetime import datetime

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
            Modified response object
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
            f"Response: {response.status_code}{duration} - {request.get('method', 'GET')} {request.get('url', '')}"
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
    ):
        """Initialize compression middleware.

        Args:
            compress_requests: Whether to compress request bodies
            min_size_to_compress: Minimum body size in bytes to compress
            compress_type: Compression algorithm ("gzip" or "deflate")
        """
        if compress_type not in ["gzip", "deflate"]:
            raise ValueError("compress_type must be 'gzip' or 'deflate'")

        self.compress_requests = compress_requests
        self.min_size_to_compress = min_size_to_compress
        self.compress_type = compress_type

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Compress request body if needed."""
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
        import gzip
        import zlib

        if self.compress_type == "gzip":
            compressed = gzip.compress(content)
        else:  # deflate
            compressed = zlib.compress(content)

        # Update request with compressed content
        request["content"] = compressed

        # Add appropriate header
        if "headers" not in request:
            request["headers"] = {}

        request["headers"]["Content-Encoding"] = self.compress_type

        return request

    async def process_response(
        self, response: httpx.Response, request: Dict[str, Any]
    ) -> httpx.Response:
        """No modifications needed for response (httpx handles response decompression)."""
        return response


class CacheMiddleware(Middleware):
    """Simple in-memory caching middleware."""

    def __init__(
        self,
        max_entries: int = 100,
        ttl_seconds: int = 300,
        cacheable_methods: List[str] = None,
        cacheable_status_codes: List[int] = None,
    ):
        """Initialize caching middleware.

        Args:
            max_entries: Maximum number of cached responses
            ttl_seconds: Time-to-live in seconds for cached entries
            cacheable_methods: HTTP methods that can be cached (default: GET, HEAD)
            cacheable_status_codes: HTTP status codes that can be cached (default: 200, 203, 300, 301, 308)
        """
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.cacheable_methods = [m.upper() for m in (cacheable_methods or ["GET", "HEAD"])]
        self.cacheable_status_codes = cacheable_status_codes or [200, 203, 300, 301, 308]

        # Simple cache storage: {cache_key: (expires_at, response_data)}
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
            expires_at, response_data = self._cache[cache_key]

            # If not expired, mark request to be handled by response processor
            if time.time() < expires_at:
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

            # Store in cache
            cache_key = self._generate_cache_key(request)
            expires_at = time.time() + self.ttl_seconds

            # Store serialized response data
            self._cache[cache_key] = (expires_at, self._serialize_response(response))
            logger.debug(f"Cached response for {method} {request.get('url')}")

            # Ensure cache doesn't grow too large (simple LRU)
            if len(self._cache) > self.max_entries:
                # Remove oldest entry
                oldest_key = min(self._cache.items(), key=lambda item: item[1][0])[0]
                del self._cache[oldest_key]

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
