from typing import Optional, Dict, Any, Union
from httpx import Response


class HTTPXError(Exception):
    """Base class for exceptions in the enhanced_httpx library."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class RequestError(HTTPXError):
    """Exception raised for errors in the request preparation or transmission."""

    def __init__(self, message: str, request_info: Optional[Dict[str, Any]] = None):
        self.request_info = request_info
        super().__init__(message)


class NetworkError(RequestError):
    """Exception raised for network connectivity issues."""

    pass


class TimeoutError(RequestError):
    """Exception raised when a request times out."""

    pass


class SSLError(RequestError):
    """Exception raised for SSL/TLS related errors."""

    pass


class ResponseError(HTTPXError):
    """Exception raised for errors in the response."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response: Optional[Response] = None,
        response_body: Optional[Any] = None,
    ):
        self.status_code = status_code
        self.response = response
        self.response_body = response_body
        super().__init__(message)


class ClientError(ResponseError):
    """Exception raised for 4xx client error responses."""

    pass


class ServerError(ResponseError):
    """Exception raised for 5xx server error responses."""

    pass


class RedirectError(ResponseError):
    """Exception raised for redirect-related issues."""

    pass


class AuthenticationError(ClientError):
    """Exception raised for authentication failures (401)."""

    pass


class ForbiddenError(ClientError):
    """Exception raised for authorization failures (403)."""

    pass


class NotFoundError(ClientError):
    """Exception raised for resource not found errors (404)."""

    pass


class ValidationError(ClientError):
    """Exception raised for invalid request data or response parsing."""

    def __init__(self, message: str, errors: Optional[Dict[str, Any]] = None, **kwargs):
        self.errors = errors
        super().__init__(message, **kwargs)


class SessionError(HTTPXError):
    """Exception raised for session-related errors."""

    pass


class ConfigurationError(HTTPXError):
    """Exception raised for configuration issues."""

    pass


class HTTPStatusError(ResponseError):
    """Exception that maps HTTP status codes to specific error classes."""

    @classmethod
    def from_response(cls, response: Response) -> "HTTPStatusError":
        """Create an appropriate error instance based on the HTTP status code."""
        status_code = response.status_code
        message = f"HTTP Error {status_code}: {response.reason_phrase}"

        try:
            response_body = response.json()
        except Exception:
            response_body = response.text

        error_kwargs = {
            "message": message,
            "status_code": status_code,
            "response": response,
            "response_body": response_body,
        }

        if 400 <= status_code < 500:
            if status_code == 401:
                return AuthenticationError(**error_kwargs)
            elif status_code == 403:
                return ForbiddenError(**error_kwargs)
            elif status_code == 404:
                return NotFoundError(**error_kwargs)
            elif status_code == 422:
                return ValidationError(**error_kwargs)
            else:
                return ClientError(**error_kwargs)
        elif 500 <= status_code < 600:
            return ServerError(**error_kwargs)
        elif 300 <= status_code < 400:
            return RedirectError(**error_kwargs)
        else:
            return cls(**error_kwargs)
