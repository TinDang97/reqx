"""
Data models for the enhanced-httpx library.
"""

import json
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional, Type, TypeVar, Union

import httpx
from pydantic import BaseModel, Field, HttpUrl, validator

T = TypeVar("T", bound=BaseModel)


class HttpMethod(str, Enum):
    """HTTP methods supported by the client."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class RequestModel(BaseModel):
    """
    Model for HTTP requests with comprehensive validation.
    """

    url: HttpUrl
    method: HttpMethod = Field(default=HttpMethod.GET, description="HTTP method to use")
    headers: Dict[str, str] = Field(
        default_factory=dict, description="Custom headers for the request"
    )
    cookies: Dict[str, str] = Field(
        default_factory=dict, description="Cookies to send with the request"
    )
    params: Dict[str, Any] = Field(default_factory=dict, description="Query parameters")
    body: Optional[Union[str, Dict[str, Any], List[Any]]] = Field(
        default=None, description="Body of the request, can be a string, dict, or list"
    )
    timeout: Optional[float] = Field(default=None, description="Request timeout in seconds")
    follow_redirects: bool = Field(default=True, description="Whether to follow redirects")
    verify_ssl: bool = Field(default=True, description="Whether to verify SSL certificates")

    class Config:
        use_enum_values = True
        extra = "forbid"  # Prevent extra attributes
        # Enable slots for memory optimization
        slots = True

    @validator("url")
    def validate_url(cls, v):
        """Validate URL format."""
        if not str(v).startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v

    @validator("headers", "cookies", "params")
    def validate_dict_values(cls, v):
        """Ensure all dictionary values are strings."""
        if v is None:
            return {}
        return {k: str(val) if val is not None else "" for k, val in v.items()}

    @validator("body")
    def validate_body(cls, v):
        """Validate request body format."""
        if v is None:
            return None

        # If it's already a string, try to make sure it's valid JSON if it looks like JSON
        if isinstance(v, str):
            if v.strip().startswith("{") or v.strip().startswith("["):
                try:
                    json.loads(v)  # Just validate it's proper JSON
                except json.JSONDecodeError:
                    raise ValueError(
                        "Body string appears to be JSON but is not valid JSON"
                    ) from None
            return v

        # Otherwise, leave it as a dict or list
        return v


class ResponseModel(BaseModel):
    """
    Model for HTTP responses with proper validation.
    """

    status_code: int = Field(..., description="HTTP status code of the response")
    headers: Dict[str, str] = Field(
        default_factory=dict, description="Headers returned in the response"
    )
    body: Any = Field(default=None, description="Body of the response")
    elapsed: Optional[float] = Field(default=None, description="Request elapsed time in seconds")
    url: Optional[HttpUrl] = Field(default=None, description="Final URL after any redirects")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When the response was received"
    )

    class Config:
        # Enable slots for memory optimization
        slots = True

    @validator("status_code")
    def validate_status_code(cls, v):
        """Validate HTTP status code range."""
        if not (100 <= v <= 599):
            raise ValueError(f"Invalid HTTP status code: {v}")
        return v

    @property
    def is_success(self) -> bool:
        """Check if the response was successful (2xx status code)."""
        return 200 <= self.status_code < 300

    @property
    def is_redirect(self) -> bool:
        """Check if the response is a redirect (3xx status code)."""
        return 300 <= self.status_code < 400

    @property
    def is_client_error(self) -> bool:
        """Check if the response is a client error (4xx status code)."""
        return 400 <= self.status_code < 500

    @property
    def is_server_error(self) -> bool:
        """Check if the response is a server error (5xx status code)."""
        return 500 <= self.status_code < 600


class ReqxResponse:
    """
    Response container class for all responses from the transport layer.

    This class provides:
    1. Request information tracking
    2. Easy conversion to Pydantic models
    3. Improved error handling
    4. Performance metrics
    """

    def __init__(
        self,
        status_code: int,
        headers: Dict[str, str],
        content: bytes,
        request: Optional[httpx.Request] = None,
        extensions: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.status_code = status_code
        self.headers = headers
        self.content = content
        self._text = None
        self._json = None
        self.request = request
        self.extensions = extensions or {}

        # Request metadata
        self.request_start_time = kwargs.get("request_start_time")
        self.request_end_time = kwargs.get("request_end_time")
        self.request_attempt = kwargs.get("request_attempt", 1)
        self.request_retries = kwargs.get("request_retries", 0)
        self.cache_hit = kwargs.get("cache_hit", False)
        self.transport_info = kwargs.get("transport_info", {})

        # Extract URL from request or use provided URL
        self.url = kwargs.get("url")
        if self.url is None and self.request is not None:
            self.url = self.request.url

    @classmethod
    def from_httpx_response(cls, response: httpx.Response, **kwargs):
        """Create a ReqxResponse from an httpx Response."""
        return cls(
            status_code=response.status_code,
            headers=response.headers,
            content=response.content,
            request=response.request,
            extensions=response.extensions,
            url=response.url,
            **kwargs,
        )

    def text(self) -> str:
        """Return the response content as a string."""
        if self._text is None:
            self._text = self.content.decode("utf-8", errors="replace")
        return self._text

    def json(self) -> Any:
        """Parse the response content as JSON."""
        if self._json is None:
            self._json = json.loads(self.text())
        return self._json

    def to_model(self, model_cls: Type[T]) -> T:
        """Convert response data to a Pydantic model."""
        try:
            json_data = self.json()
            return model_cls.model_validate(json_data)
        except Exception as e:
            from .exceptions import ResponseError

            raise ResponseError(
                f"Failed to parse response into model {model_cls.__name__}: {str(e)}",
                status_code=self.status_code,
                response=self,
            ) from e

    def to_response_model(self) -> ResponseModel:
        """Convert to a ResponseModel instance with complete metadata."""
        try:
            body = self.json()
        except Exception:
            body = self.text()

        elapsed = None
        if self.request_start_time and self.request_end_time:
            elapsed = self.request_end_time - self.request_start_time

        return ResponseModel(
            status_code=self.status_code,
            headers={k: v for k, v in self.headers.items()},
            body=body,
            elapsed=elapsed,
            url=str(self.url),
            timestamp=datetime.now(),
        )

    @property
    def request_time(self) -> Optional[float]:
        """Return the time taken for the request in seconds."""
        if self.request_start_time and self.request_end_time:
            return self.request_end_time - self.request_start_time
        return None

    @property
    def is_success(self) -> bool:
        """Check if the response indicates success."""
        return 200 <= self.status_code < 300

    @property
    def is_error(self) -> bool:
        """Check if the response indicates an error."""
        return self.status_code >= 400

    @property
    def is_client_error(self) -> bool:
        """Check if the response indicates a client error."""
        return 400 <= self.status_code < 500

    @property
    def is_server_error(self) -> bool:
        """Check if the response indicates a server error."""
        return self.status_code >= 500

    def raise_for_status(self) -> None:
        """Raise an exception if the response contains an HTTP error status code."""
        from .exceptions import ResponseError

        if self.is_error:
            error_msg = f"HTTP error occurred: {self.status_code}"
            if self.status_code >= 500:
                error_msg = f"Server error: {self.status_code}"
            elif self.status_code >= 400:
                error_msg = f"Client error: {self.status_code}"

            raise ResponseError(
                message=error_msg,
                status_code=self.status_code,
                response=self,
            )

    def dict(self) -> Dict[str, Any]:
        """Convert response to a dictionary representation."""
        try:
            body = self.json()
        except Exception:
            body = self.text()

        return {
            "status_code": self.status_code,
            "url": str(self.url) if self.url else None,
            "headers": {k: v for k, v in self.headers.items()},
            "body": body,
            "elapsed": self.request_time,
            "ok": self.is_success,
            "request_info": {
                "method": self.request.method if self.request else None,
                "url": str(self.request.url) if self.request else None,
                "attempt": self.request_attempt,
                "retries": self.request_retries,
                "cache_hit": self.cache_hit,
            },
            "transport_info": self.transport_info,
        }


class GenericResponse(Generic[T], BaseModel):
    """
    Generic response model that can be used to parse API responses into specific types.
    """

    data: Optional[T] = None
    meta: Optional[Dict[str, Any]] = None
    errors: Optional[List[Dict[str, Any]]] = None

    class Config:
        # Enable slots for memory optimization
        slots = True

    @property
    def has_errors(self) -> bool:
        """Check if the response contains errors."""
        return self.errors is not None and len(self.errors) > 0


class RequestBatch(BaseModel):
    """A batch of HTTP requests to be executed together."""

    requests: List[Dict[str, Any]] = Field(default_factory=list)
    max_connections: int = 10
    timeout: float = 30.0

    class Config:
        # Enable slots for memory optimization
        slots = True

    def add_request(self, method: HttpMethod, url: str, **kwargs) -> int:
        """
        Add a request to the batch.

        Args:
            method: HTTP method
            url: URL
            **kwargs: Additional arguments to pass to httpx.request

        Returns:
            The index of the added request in the batch
        """
        request = {"method": method, "url": url, **kwargs}
        self.requests.append(request)
        return len(self.requests) - 1

    async def execute(
        self, client: httpx.AsyncClient
    ) -> List[Union[httpx.Response, BaseException]]:
        """
        Execute all requests in the batch concurrently.

        Args:
            client: HTTP client to use

        Returns:
            List of responses or exceptions in the same order as the requests
        """
        import asyncio

        # Use a semaphore to limit the number of concurrent connections
        semaphore = asyncio.Semaphore(self.max_connections)

        async def fetch(request_args):
            async with semaphore:
                return await client.request(**request_args)

        # Create tasks for all requests
        tasks = [fetch(request) for request in self.requests]

        # Execute all tasks concurrently and gather results
        return await asyncio.gather(*tasks, return_exceptions=True)

    def clear(self) -> None:
        """Clear all requests from the batch."""
        self.requests.clear()


class RequestHook(BaseModel):
    """Hook for request interception and modification."""

    callback: Optional[Callable] = None
    async_callback: Optional[Callable[[str, str, Dict[str, Any]], Awaitable[Dict[str, Any]]]] = None

    class Config:
        # Enable slots for memory optimization
        slots = True


class ResponseHook(BaseModel):
    """Hook for response interception and modification."""

    callback: Optional[Callable] = None
    async_callback: Optional[Callable[[ReqxResponse], Awaitable[ReqxResponse]]] = None

    class Config:
        # Enable slots for memory optimization
        slots = True
