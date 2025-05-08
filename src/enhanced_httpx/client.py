from httpx import AsyncClient, Response, Timeout, Limits, TransportError
from pydantic import BaseModel, Field, TypeAdapter
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    TypeVar,
    Generic,
    Type,
    cast,
    overload,
    Callable,
    Awaitable,
    Set,
)
import orjson
import asyncio
import uvloop
from urllib.parse import urljoin
import ssl
import logging
import time
import hashlib
from datetime import datetime, timedelta
from .exceptions import RequestError, ResponseError, SessionError, MiddlewareError, RateLimitError
from .utils import log_request, log_response, serialize_json, deserialize_json

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
logger = logging.getLogger("enhanced_httpx")

T = TypeVar("T")

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


class RequestModel(BaseModel):
    url: str
    method: str = Field(default="GET")
    headers: Optional[Dict[str, str]] = Field(default_factory=dict)
    cookies: Optional[Dict[str, str]] = Field(default_factory=dict)
    params: Optional[Dict[str, Any]] = Field(default_factory=dict)
    body: Optional[Any] = None
    json: Optional[Dict[str, Any]] = None
    timeout: Optional[float] = None
    follow_redirects: bool = True
    verify_ssl: bool = True


class EnhancedClient:
    """
    An enhanced HTTP client built on top of httpx with additional features
    for performance, security, and usability.
    """

    def __init__(
        self,
        base_url: str = "",
        headers: Dict[str, str] = None,
        cookies: Dict[str, str] = None,
        timeout: float = 30.0,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        keepalive_expiry: int = 60,
        follow_redirects: bool = True,
        verify_ssl: bool = True,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
        http2: bool = False,
        enable_http3: bool = False,  # New HTTP/3 support option
        debug: bool = False,
        enable_cache: bool = False,  # Enable response caching
        cache_ttl: int = 300,  # Default cache TTL in seconds
        rate_limit: Optional[float] = None,  # Requests per second
        rate_limit_max_tokens: int = 60,  # Maximum rate limit tokens
    ):
        """
        Initialize the EnhancedClient with custom configuration.

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

        # Configure SSL
        self.ssl_context = ssl.create_default_context()
        if not verify_ssl:
            self.ssl_context.check_hostname = False
            self.ssl_context.verify_mode = ssl.CERT_NONE

        # Configure connection pooling
        limits = Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            keepalive_expiry=keepalive_expiry,
        )

        # Initialize the client with HTTP/2 or HTTP/3 if enabled
        client_kwargs = {
            "base_url": base_url,
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
                from httpx_h3 import H3Transport

                client_kwargs["transport"] = H3Transport()
                logger.debug("HTTP/3 (QUIC) support enabled")
            except ImportError:
                logger.warning(
                    "HTTP/3 requested but httpx_h3 not installed. Falling back to HTTP/1.1/HTTP/2"
                )

        self.client = AsyncClient(**client_kwargs)

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
        # Return a copy of the cached response
        return cache_entry.response.copy()

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

        # Create a copy of the response before caching
        self.cache[cache_key] = CacheEntry(response.copy(), expires_at)
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
        headers: Dict[str, str] = None,
        cookies: Dict[str, str] = None,
        params: Dict[str, Any] = None,
        json: Dict[str, Any] = None,
        data: Any = None,
        files: Dict[str, Any] = None,
        timeout: float = None,
        follow_redirects: bool = None,
        verify_ssl: bool = None,
        response_model: Type[T] = None,
        stream: bool = False,
        cache: bool = None,
        cache_ttl: int = None,
    ) -> Union[Response, T]:
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

        Returns:
            Response object or parsed model instance if response_model is provided
        """
        start_time = time.time()
        self.metrics["requests_sent"] += 1

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
        _verify_ssl = verify_ssl if verify_ssl is not None else self.verify_ssl

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
            "verify": _verify_ssl,
        }

        # Check if we can use a cached response
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
                            f"Failed to parse cached response into model {response_model.__name__}: {str(e)}"
                        )

                return cached_response

        # Apply request middlewares
        kwargs = await self._apply_request_middlewares(method, full_url, kwargs)

        # Initialize retry counter
        retry_count = 0
        last_exception = None

        # Retry loop
        while retry_count <= self.max_retries:
            try:
                # Make the request (with or without streaming)
                response = await self.client.request(method=method, url=full_url, **kwargs)

                if self.debug:
                    log_response(response)

                # Apply response middlewares
                response = await self._apply_response_middlewares(response)

                # Check if the response indicates an error
                response.raise_for_status()

                # Store in cache if cacheable and not streaming
                if cacheable and not stream:
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
                            f"Failed to parse response into model {response_model.__name__}: {str(e)}"
                        ) from e

                # Update metrics
                self.metrics["total_request_time"] += time.time() - start_time
                return response

            except TransportError as e:
                # Network-related errors are retryable
                last_exception = e
                retry_count += 1
                self.metrics["retry_attempts"] += 1
                if retry_count <= self.max_retries:
                    # Exponential backoff
                    wait_time = self.retry_backoff * (2 ** (retry_count - 1))
                    logger.debug(
                        f"Request failed with error: {str(e)}. Retrying in {wait_time:.2f} seconds..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self.metrics["request_errors"] += 1
                    raise RequestError(f"Request failed after {self.max_retries} retries: {str(e)}")

            except Exception as e:
                # Other exceptions are not retried
                self.metrics["request_errors"] += 1
                if isinstance(e, ResponseError) or isinstance(e, MiddlewareError):
                    raise
                raise RequestError(f"Request failed: {str(e)}")

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

    async def stream(self, method: str, url: str, **kwargs) -> Response:
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
        import os

        async with self.stream("GET", url, **kwargs) as response:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

            # Open the file for writing
            with open(file_path, "wb") as f:
                # Download the file in chunks
                async for chunk in response.aiter_bytes(chunk_size):
                    f.write(chunk)

    async def close(self):
        """Close the client session."""
        if self.client:
            await self.client.aclose()

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
            raise_exceptions: Whether to raise the first exception encountered or return them as results
            progress_callback: Optional callback function that receives (completed, total) as arguments

        Returns:
            List of responses or exceptions in the same order as the requests
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        results: List[None | Response | Exception] = [None] * len(requests)
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
            progress_callback: Optional callback function that receives (completed, total) as arguments

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
                            f"Failed to parse response into model {response_model.__name__}: {str(e)}"
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
        response_model: Type[T] = None,
        raise_exceptions: bool = False
    ) -> List[Union[T, Response, Exception]]:
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
        results = [None] * len(requests)

        async def _process_request(index: int, req_config: Dict[str, Any]):
            async with semaphore:
                method = req_config.pop("method", "GET")
                url = req_config.pop("url")
                
                try:
                    response = await self.request(method, url, **req_config)
                    results[index] = response
                except Exception as e:
                    if raise_exceptions:
                        raise e
                    results[index] = e

        # Create tasks for all requests
        tasks = [
            _process_request(i, req) 
            for i, req in enumerate(requests)
        ]
        
        # Execute all requests with concurrency control
        await asyncio.gather(*tasks)
        return results
        
    async def batch_get(
        self, 
        urls: List[str], 
        concurrency_limit: int = 10,
        response_model: Type[T] = None,
        **kwargs
    ) -> List[Union[T, Response, Exception]]:
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
        requests = [
            {"method": "GET", "url": url, **kwargs}
            for url in urls
        ]
        return await self.batch_request(requests, concurrency_limit, response_model)
