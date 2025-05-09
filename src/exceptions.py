from typing import Any, Dict, List, Optional

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


class MiddlewareError(HTTPXError):
    """Exception raised when a middleware encounters an error."""

    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        self.original_exception = original_exception
        super().__init__(message)


class SecurityError(HTTPXError):
    """
    Error related to security checks.

    This is raised when security-related checks fail, such as certificate
    pinning validation, HSTS validation, etc.
    """

    pass


class CircuitBreakerError(HTTPXError):
    """
    Error when a circuit breaker prevents a request.

    This is raised when a circuit breaker is open for a particular host,
    indicating that the service is currently considered unavailable.
    """

    def __init__(self, message: str, host: str | None = None, retry_after: float | None = None):
        self.host = host
        self.retry_after = retry_after
        super().__init__(message)


class GraphQLError(HTTPXError):
    """
    Error related to GraphQL request execution.

    This is raised when a GraphQL request fails, either due to HTTP errors
    or errors in the GraphQL response itself.
    """

    def __init__(
        self,
        message: str,
        http_status: int | None = None,
        errors: List[Dict[str, Any]] | None = None,
        data: Dict[str, Any] | None = None,
        response_data: Dict[str, Any] | None = None,
    ):
        self.http_status = http_status
        self.errors = errors or []
        self.data = data
        self.response_data = response_data
        super().__init__(message)


class HTTPStatusError(ResponseError):
    """Exception that maps HTTP status codes to specific error classes."""

    @classmethod
    def from_response(cls, response: Response) -> "ResponseError":
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
