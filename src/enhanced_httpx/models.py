"""
Data models for the enhanced-httpx library.
"""

from pydantic import BaseModel, HttpUrl, Field, validator, root_validator, TypeAdapter
from pydantic.version import VERSION as PYDANTIC_VERSION
from typing import Optional, Any, Dict, Union, List, TypeVar, Generic, Type, Callable, Awaitable
from enum import Enum
import json
import re
from datetime import datetime
import httpx

T = TypeVar("T")


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
                    raise ValueError("Body string appears to be JSON but is not valid JSON")
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


class GenericResponse(Generic[T], BaseModel):
    """
    Generic response model that can be used to parse API responses into specific types.
    """

    data: Optional[T] = None
    meta: Optional[Dict[str, Any]] = None
    errors: Optional[List[Dict[str, Any]]] = None

    @property
    def has_errors(self) -> bool:
        """Check if the response contains errors."""
        return self.errors is not None and len(self.errors) > 0


class RequestBatch(BaseModel):
    """A batch of HTTP requests to be executed together."""

    requests: List[Dict[str, Any]] = Field(default_factory=list)
    max_connections: int = 10
    timeout: float = 30.0

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

    async def execute(self, client: httpx.AsyncClient) -> List[httpx.Response]:
        """
        Execute all requests in the batch concurrently.

        Args:
            client: HTTP client to use

        Returns:
            List of responses in the same order as the requests
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


class ResponseHook(BaseModel):
    """Hook for response interception and modification."""

    callback: Optional[Callable] = None
    async_callback: Optional[Callable[[httpx.Response], Awaitable[httpx.Response]]] = None
