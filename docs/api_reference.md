# Enhanced HTTPX API Reference

This document provides detailed API documentation for the enhanced-httpx library.

## Table of Contents

- [EnhancedClient](#enhancedclient)
  - [Constructor](#constructor)
  - [Basic Request Methods](#basic-request-methods)
  - [Middleware Support](#middleware-support)
  - [Caching](#caching)
  - [Rate Limiting](#rate-limiting)
  - [Batch Requests](#batch-requests)
- [Models](#models)
  - [RequestModel](#requestmodel)
  - [ResponseModel](#responsemodel)
  - [GenericResponse](#genericresponse)
  - [RequestBatch](#requestbatch)
- [Exceptions](#exceptions)
- [Utilities](#utilities)

## EnhancedClient

The `EnhancedClient` class is the main interface for making HTTP requests.

```python
from enhanced_httpx import EnhancedClient
```

### Constructor

```python
EnhancedClient(
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
    enable_http3: bool = False,
    debug: bool = False,
    enable_cache: bool = False,
    cache_ttl: int = 300,
    rate_limit: Optional[float] = None,
    rate_limit_max_tokens: int = 60,
)
```

Parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | `str` | `""` | Base URL for all requests |
| `headers` | `Dict[str, str]` | `None` | Default headers for all requests |
| `cookies` | `Dict[str, str]` | `None` | Default cookies for all requests |
| `timeout` | `float` | `30.0` | Default timeout for all requests in seconds |
| `max_connections` | `int` | `100` | Maximum number of connections |
| `max_keepalive_connections` | `int` | `20` | Maximum number of idle keepalive connections |
| `keepalive_expiry` | `int` | `60` | Keepalive connection expiry in seconds |
| `follow_redirects` | `bool` | `True` | Whether to follow redirects by default |
| `verify_ssl` | `bool` | `True` | Whether to verify SSL certificates by default |
| `max_retries` | `int` | `3` | Maximum number of retries for failed requests |
| `retry_backoff` | `float` | `0.5` | Exponential backoff factor for retries |
| `http2` | `bool` | `False` | Whether to use HTTP/2 |
| `enable_http3` | `bool` | `False` | Whether to enable HTTP/3 (QUIC) if available |
| `debug` | `bool` | `False` | Whether to enable debug logging |
| `enable_cache` | `bool` | `False` | Whether to enable response caching |
| `cache_ttl` | `int` | `300` | Default cache TTL in seconds |
| `rate_limit` | `Optional[float]` | `None` | Optional rate limit in requests per second |
| `rate_limit_max_tokens` | `int` | `60` | Maximum tokens for rate limiting |

### Basic Request Methods

#### GET Request

```python
async def get(
    self,
    url: str,
    *,
    params: Dict[str, Any] = None,
    headers: Dict[str, str] = None,
    cookies: Dict[str, str] = None,
    follow_redirects: bool = None,
    timeout: float = None,
    response_model: Type[T] = None,
    force_refresh: bool = False,
) -> Union[Response, T]:
    """
    Send an HTTP GET request.
    
    Args:
        url: URL to request
        params: Query parameters to append to the URL
        headers: Custom headers for this request
        cookies: Custom cookies for this request
        follow_redirects: Whether to follow redirects
        timeout: Request timeout in seconds
        response_model: Optional Pydantic model to parse the response into
        force_refresh: Whether to ignore cache and force a fresh request
        
    Returns:
        Response object or parsed model instance if response_model is provided
    """
```

#### POST Request

```python
async def post(
    self,
    url: str,
    *,
    data: Any = None,
    json: Dict[str, Any] = None,
    params: Dict[str, Any] = None,
    headers: Dict[str, str] = None,
    cookies: Dict[str, str] = None,
    follow_redirects: bool = None,
    timeout: float = None,
    response_model: Type[T] = None,
) -> Union[Response, T]:
    """
    Send an HTTP POST request.
    
    Args:
        url: URL to request
        data: Form data or request body
        json: JSON data to include in the request body
        params: Query parameters to append to the URL
        headers: Custom headers for this request
        cookies: Custom cookies for this request
        follow_redirects: Whether to follow redirects
        timeout: Request timeout in seconds
        response_model: Optional Pydantic model to parse the response into
        
    Returns:
        Response object or parsed model instance if response_model is provided
    """
```

Similar methods are available for `put()`, `patch()`, `delete()`, `head()`, and `options()`.

#### Request Method

```python
async def request(
    self,
    method: str,
    url: str,
    **kwargs
) -> Response:
    """
    Send an HTTP request with the given method.
    
    Args:
        method: HTTP method (GET, POST, etc.)
        url: URL to request
        **kwargs: Additional arguments to pass to httpx.request
        
    Returns:
        Response object
    """
```

### Middleware Support

```python
def add_request_middleware(self, middleware: RequestMiddleware) -> None:
    """
    Add a middleware function that will be called before each request.
    
    Args:
        middleware: A function that takes (method, url, request_kwargs) and returns modified request_kwargs
    """
```

```python
def add_response_middleware(self, middleware: ResponseMiddleware) -> None:
    """
    Add a middleware function that will be called after each response.
    
    Args:
        middleware: A function that takes a Response object and returns a modified Response object
    """
```

### Caching

Caching is configured through the constructor options `enable_cache` and `cache_ttl`. 
Additionally, you can use the `force_refresh` parameter on request methods to bypass the cache.

```python
async def clear_cache(self) -> None:
    """
    Clear the response cache.
    """

async def get_cache_stats(self) -> Dict[str, int]:
    """
    Get cache statistics.
    
    Returns:
        Dictionary with cache hit/miss counts
    """
```

### Rate Limiting

Rate limiting is configured through the constructor options `rate_limit` and `rate_limit_max_tokens`.

```python
async def get_rate_limit_stats(self) -> Dict[str, float]:
    """
    Get rate limiting statistics.
    
    Returns:
        Dictionary with rate limiting statistics
    """
```

### Batch Requests

```python
def create_batch(self) -> RequestBatch:
    """
    Create a new request batch.
    
    Returns:
        A new RequestBatch instance
    """

async def execute_batch(
    self,
    batch: RequestBatch,
    max_connections: int = None
) -> List[Response]:
    """
    Execute a batch of requests concurrently.
    
    Args:
        batch: The RequestBatch to execute
        max_connections: Maximum number of concurrent connections
        
    Returns:
        List of Response objects in the same order as the requests
    """

async def execute_batch_with_model(
    self,
    batch: RequestBatch,
    response_model: Type[T],
    max_connections: int = None
) -> List[Union[T, Exception]]:
    """
    Execute a batch of requests and parse each response into the given model.
    
    Args:
        batch: The RequestBatch to execute
        response_model: Pydantic model to parse each response into
        max_connections: Maximum number of concurrent connections
        
    Returns:
        List of model instances or exceptions in the same order as the requests
    """
```

## Models

### RequestModel

```python
class RequestModel(BaseModel):
    """
    Model for HTTP requests with comprehensive validation.
    """
    url: HttpUrl
    method: HttpMethod = HttpMethod.GET
    headers: Dict[str, str] = Field(default_factory=dict)
    cookies: Dict[str, str] = Field(default_factory=dict)
    params: Dict[str, Any] = Field(default_factory=dict)
    body: Optional[Union[str, Dict[str, Any], List[Any]]] = None
    timeout: Optional[float] = None
    follow_redirects: bool = True
    verify_ssl: bool = True
```

### ResponseModel

```python
class ResponseModel(BaseModel):
    """
    Model for HTTP responses with proper validation.
    """
    status_code: int
    headers: Dict[str, str] = Field(default_factory=dict)
    body: Any = None
    elapsed: Optional[float] = None
    url: Optional[HttpUrl] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    @property
    def is_success(self) -> bool:
        """Check if the response was successful (2xx status code)."""
        
    @property
    def is_redirect(self) -> bool:
        """Check if the response is a redirect (3xx status code)."""
        
    @property
    def is_client_error(self) -> bool:
        """Check if the response is a client error (4xx status code)."""
        
    @property
    def is_server_error(self) -> bool:
        """Check if the response is a server error (5xx status code)."""
```

### GenericResponse

```python
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
```

### RequestBatch

```python
class RequestBatch(BaseModel):
    """A batch of HTTP requests to be executed together."""
    
    requests: List[Dict[str, Any]] = Field(default_factory=list)
    max_connections: int = 10
    timeout: float = 30.0
    
    def add_request(
        self,
        method: HttpMethod,
        url: str,
        **kwargs
    ) -> int:
        """
        Add a request to the batch.
        
        Args:
            method: HTTP method
            url: URL
            **kwargs: Additional arguments to pass to httpx.request
            
        Returns:
            The index of the added request in the batch
        """
    
    async def execute(self, client: httpx.AsyncClient) -> List[httpx.Response]:
        """
        Execute all requests in the batch concurrently.
        
        Args:
            client: HTTP client to use
            
        Returns:
            List of responses in the same order as the requests
        """
    
    def clear(self) -> None:
        """Clear all requests from the batch."""
```

## Exceptions

Enhanced HTTPX provides several specialized exception classes:

- `RequestError`: Base class for all request-related errors
- `ResponseError`: Base class for all response-related errors
- `TimeoutError`: Raised when a request times out
- `ConnectionError`: Raised when a connection cannot be established
- `RedirectError`: Raised when there are too many redirects
- `ClientError`: Raised for 4xx responses (has subclasses like `NotFoundError`, `UnauthorizedError`, etc.)
- `ServerError`: Raised for 5xx responses
- `ParseError`: Raised when there's an error parsing a response
- `ValidationError`: Raised when there's an error validating a response against a model
- `MiddlewareError`: Raised when there's an error in middleware processing
- `RateLimitError`: Raised when rate limit is exceeded
- `BatchRequestError`: Raised for errors in batch requests

## Utilities

### JSON Path Selection

```python
def select_json_path(data: Any, path: str) -> Any:
    """
    Extract data from a JSON structure using a JSONPath expression.
    
    Args:
        data: JSON data to extract from
        path: JSONPath expression
        
    Returns:
        Extracted data
    """
```

### JSON Serialization

```python
def serialize_json(data: Any) -> str:
    """
    Serialize data to JSON string using orjson.
    
    Args:
        data: Data to serialize
        
    Returns:
        JSON string
    """

def deserialize_json(data: str) -> Any:
    """
    Deserialize JSON string to Python object using orjson.
    
    Args:
        data: JSON string to deserialize
        
    Returns:
        Deserialized Python object
    """
```

### Logging

```python
def log_request(
    method: str,
    url: str,
    **kwargs
) -> None:
    """
    Log an HTTP request.
    
    Args:
        method: HTTP method
        url: URL
        **kwargs: Request arguments
    """

def log_response(
    response: Response
) -> None:
    """
    Log an HTTP response.
    
    Args:
        response: Response object
    """
```