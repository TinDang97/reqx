from httpx import AsyncClient, Response, Timeout, Limits, TransportError
from pydantic import BaseModel, Field, TypeAdapter
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic, Type, cast, overload
import orjson
import asyncio
import uvloop
from urllib.parse import urljoin
import ssl
import logging
from .exceptions import RequestError, ResponseError, SessionError
from .utils import log_request, log_response, serialize_json, deserialize_json

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
logger = logging.getLogger("enhanced_httpx")

T = TypeVar("T")


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
        debug: bool = False,
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
            debug: Whether to enable debug logging
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

        # Initialize the client
        self.client = AsyncClient(
            base_url=base_url,
            headers=headers,
            cookies=cookies,
            timeout=Timeout(timeout),
            follow_redirects=follow_redirects,
            verify=verify_ssl,
            limits=limits,
            http2=http2,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def _prepare_url(self, url: str) -> str:
        """Prepare the URL by joining it with the base_url if it's not absolute."""
        if not url.startswith(("http://", "https://")):
            return urljoin(self.base_url, url)
        return url

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

        Returns:
            Response object or parsed model instance if response_model is provided
        """
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

        # Initialize retry counter
        retry_count = 0
        last_exception = None

        # Retry loop
        while retry_count <= self.max_retries:
            try:
                response = await self.client.request(
                    method=method,
                    url=full_url,
                    headers=merged_headers,
                    cookies=merged_cookies,
                    params=params,
                    json=json,
                    data=data,
                    files=files,
                    timeout=_timeout,
                    follow_redirects=_follow_redirects,
                    verify=_verify_ssl,
                )

                if self.debug:
                    log_response(response)

                # Check if the response indicates an error
                response.raise_for_status()

                # Parse to response model if one was provided
                if response_model:
                    try:
                        json_data = response.json()
                        # Use TypeAdapter for Pydantic v2 compatibility
                        adapter = TypeAdapter(response_model)
                        return adapter.validate_python(json_data)
                    except Exception as e:
                        raise ResponseError(
                            f"Failed to parse response into model {response_model.__name__}: {str(e)}"
                        )

                return response

            except TransportError as e:
                # Network-related errors are retryable
                last_exception = e
                retry_count += 1
                if retry_count <= self.max_retries:
                    # Exponential backoff
                    wait_time = self.retry_backoff * (2 ** (retry_count - 1))
                    logger.debug(
                        f"Request failed with error: {str(e)}. Retrying in {wait_time:.2f} seconds..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise RequestError(f"Request failed after {self.max_retries} retries: {str(e)}")

            except Exception as e:
                # Other exceptions are not retried
                if isinstance(e, ResponseError):
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

    async def close(self):
        """Close the client session."""
        if self.client:
            await self.client.aclose()
