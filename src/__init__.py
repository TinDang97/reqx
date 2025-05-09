"""
Reqx Package

This package provides an enhanced HTTP client built on top of httpx, allowing for custom HTTP
requests with user-defined headers, cookies, and body in an asynchronous manner.
It includes Pydantic models for request and response validation, custom exceptions,
and a command-line interface for easy usage.

Features:
- Asynchronous HTTP requests with httpx
- Automatic retries with exponential backoff
- Connection pooling for performance
- JSON path selectors for response data extraction
- Comprehensive error handling
- Command-line interface
- Type validation with Pydantic
- Fast JSON serialization with orjson
- SSL/TLS security features
"""

__version__ = "0.1.0"

# CLI entry point
from . import cli as cli_app

# Core client
from .client import ReqxClient

# Exceptions
from .exceptions import (
    AuthenticationError,
    ClientError,
    ConfigurationError,
    ForbiddenError,
    HTTPStatusError,
    HTTPXError,
    NetworkError,
    NotFoundError,
    RequestError,
    ResponseError,
    ServerError,
    SessionError,
    SSLError,
    TimeoutError,
    ValidationError,
)

# Models
from .models import GenericResponse, HttpMethod, RequestModel, ResponseModel

# Utils
from .utils import deserialize_json, format_curl_command, select_json_path, serialize_json

__all__ = [
    # Client
    "ReqxClient",
    # Models
    "HttpMethod",
    "RequestModel",
    "ResponseModel",
    "GenericResponse",
    # Exceptions
    "HTTPXError",
    "RequestError",
    "ResponseError",
    "NetworkError",
    "TimeoutError",
    "SSLError",
    "ClientError",
    "ServerError",
    "ValidationError",
    "AuthenticationError",
    "ForbiddenError",
    "NotFoundError",
    "SessionError",
    "ConfigurationError",
    "HTTPStatusError",
    # Utils
    "serialize_json",
    "deserialize_json",
    "select_json_path",
    "format_curl_command",
    # CLI
    "cli_app",
]
