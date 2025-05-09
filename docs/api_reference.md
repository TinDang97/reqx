# Enhanced-HTTPX API Reference

This document provides a detailed reference for the Enhanced-HTTPX library's public API.

## Table of Contents

- [Client Classes](#client-classes)
  - [ReqxClient](#reqxclient)
  - [ReqxClientBuilder](#reqxclientbuilder)
- [Request and Response](#request-and-response)
  - [Models](#models)
  - [Middleware](#middleware)
- [Advanced Features](#advanced-features)
  - [GraphQL](#graphql)
  - [Webhooks](#webhooks)
  - [Transport Management](#transport-management)
  - [Adaptive Timeouts](#adaptive-timeouts)
  - [System-Aware Optimization](#system-aware-optimization)

## Client Classes

### ReqxClient

The main client class for making HTTP requests.

#### Constructor

```python
ReqxClient(
    base_url: str = None,
    timeout: float = 30.0,
    headers: Dict[str, str] = None,
    cookies: Dict[str, str] = None,
    auth: Auth = None,
    follow_redirects: bool = True,
    max_redirects: int = 10,
    verify: bool = True,
    cert: Union[str, Tuple[str, str]] = None,
    http2: bool = False,
    http3: bool = False,
    use_aiohttp: bool = False,
    adaptive_timeout: bool = False,
    retry: bool = False,
    max_retries: int = 3,
    middleware: List[Middleware] = None,
    auto_optimize: bool = False
)
```

#### Basic Methods

```python
# Synchronous methods
get(url, params=None, headers=None, **kwargs) -> Response
post(url, data=None, json=None, headers=None, **kwargs) -> Response
put(url, data=None, json=None, headers=None, **kwargs) -> Response
patch(url, data=None, json=None, headers=None, **kwargs) -> Response
delete(url, headers=None, **kwargs) -> Response
head(url, headers=None, **kwargs) -> Response
options(url, headers=None, **kwargs) -> Response
request(method, url, **kwargs) -> Response

# Asynchronous methods
async get(url, params=None, headers=None, **kwargs) -> Response
async post(url, data=None, json=None, headers=None, **kwargs) -> Response
async put(url, data=None, json=None, headers=None, **kwargs) -> Response
async patch(url, data=None, json=None, headers=None, **kwargs) -> Response
async delete(url, headers=None, **kwargs) -> Response
async head(url, headers=None, **kwargs) -> Response
async options(url, headers=None, **kwargs) -> Response
async request(method, url, **kwargs) -> Response
```

#### Advanced Methods

```python
# Batch requests
async batch_request(requests: List[Dict]) -> List[Response]

# GraphQL support
async graphql(url, query, variables=None, operation_name=None, **kwargs) -> Response

# Webhook management
async webhook_send(url, event, payload, signature_key=None) -> Response
async webhook_verify(request_data, signature, signature_key) -> bool

# Transport and timeout management
get_timeout_statistics() -> Dict
get_transport_metrics() -> Dict
set_default_transport(transport_name: str)
```

#### Context Manager Support

```python
# Synchronous context manager
with ReqxClient() as client:
    response = client.get("https://example.com")

# Asynchronous context manager
async with ReqxClient() as client:
    response = await client.get("https://example.com")
```

### ReqxClientBuilder

A builder class for creating customized client instances.

```python
builder = ReqxClientBuilder()

# Base configuration
builder.with_base_url(url: str)
builder.with_timeout(timeout: float)
builder.with_headers(headers: Dict[str, str])
builder.with_cookies(cookies: Dict[str, str])
builder.with_auth(auth: Auth)
builder.with_verify(verify: bool)
builder.with_cert(cert: Union[str, Tuple[str, str]])
builder.with_follow_redirects(follow_redirects: bool, max_redirects: int = 10)

# Advanced features
builder.with_http2(enabled: bool = True)
builder.with_http3(enabled: bool = False)
builder.use_aiohttp(enabled: bool = False)
builder.with_adaptive_timeout(enabled: bool = True, initial_timeout: float = 10.0)
builder.with_middleware(middleware: Middleware)
builder.with_retry(enabled: bool = True, max_retries: int = 3, retry_statuses: List[int] = None)

# Preset profiles
builder.for_high_performance()
builder.for_reliability()
builder.for_low_resources()
builder.auto_optimize()

# Build the client
client = builder.build()
```

## Request and Response

### Models

#### Response

```python
class Response:
    # Properties
    status_code: int
    headers: Dict[str, str]
    content: bytes
    text: str
    encoding: str
    http_version: str
    is_redirect: bool
    is_error: bool
    cookies: Dict[str, str]
    url: str
    elapsed: float
    metadata: Dict  # Custom metadata added by middleware

    # Methods
    json() -> Dict
    raise_for_status() -> None
    close() -> None
```

### Middleware

Middleware allows you to intercept and modify requests and responses.

```python
class Middleware:
    async def process_request(self, ctx: RequestContext, next_middleware: Callable) -> ResponseContext:
        """
        Process a request and optionally modify it.
        
        Args:
            ctx: The request context with request details
            next_middleware: The next middleware in the chain
            
        Returns:
            A response context with response details
        """
        # Default implementation just passes to next middleware
        return await next_middleware(ctx)
```

#### RequestContext

```python
class RequestContext:
    method: str
    url: str
    params: Dict
    headers: Dict[str, str]
    cookies: Dict[str, str]
    data: Any
    json: Dict
    timeout: float
    metadata: Dict  # Custom metadata that middleware can use
```

#### ResponseContext

```python
class ResponseContext:
    status_code: int
    headers: Dict[str, str]
    content: bytes
    encoding: str
    cookies: Dict[str, str]
    url: str
    elapsed: float
    http_version: str
    metadata: Dict  # Custom metadata that middleware can use
```

## Advanced Features

### GraphQL

```python
# Basic GraphQL query
response = await client.graphql(
    "https://api.example.com/graphql",
    query="""
    query {
        users {
            id
            name
        }
    }
    """
)

# Query with variables
response = await client.graphql(
    "https://api.example.com/graphql",
    query="""
    query GetUser($id: ID!) {
        user(id: $id) {
            id
            name
            email
        }
    }
    """,
    variables={"id": "123"}
)
```

### Webhooks

```python
# Send a webhook
await client.webhook_send(
    "https://webhook.example.com/endpoint",
    event="user.created",
    payload={"user_id": "123", "email": "user@example.com"},
    signature_key="your_webhook_signing_key"
)

# Verify a webhook signature
is_valid = await client.webhook_verify(
    request_data=request.body,
    signature=request.headers.get("X-Webhook-Signature"),
    signature_key="your_webhook_signing_key"
)
```

### Transport Management

```python
# Get transport metrics
metrics = client.get_transport_metrics()
"""
{
    "total_requests": 100,
    "transports": {
        "httpx": {"requests": 75, "errors": 2, "avg_duration": 0.123},
        "aiohttp": {"requests": 25, "errors": 0, "avg_duration": 0.098}
    },
    "hosts_analyzed": 8,
    "host_metrics": {
        "api.example.com": {
            "httpx": {"count": 30, "avg_duration": 0.110},
            "aiohttp": {"count": 10, "avg_duration": 0.095},
            "preferred": "aiohttp"
        }
    }
}
"""

# Set default transport
client.set_default_transport("aiohttp")

# Make request with specific transport
response = await client.get("https://api.example.com", transport="httpx")
```

### Adaptive Timeouts

```python
# Get timeout statistics
stats = client.get_timeout_statistics()
"""
{
    "api.example.com": {
        "samples": 50,
        "avg_duration": 0.347,
        "p95_duration": 0.650,
        "current_timeout": 1.3,
        "timeouts": 2,
        "success_rate": 0.96
    }
}
"""

# Configure adaptive timeouts
client = (
    ReqxClientBuilder()
    .with_adaptive_timeout(
        enabled=True,
        initial_timeout=10.0,
        min_timeout=1.0,
        max_timeout=60.0,
        multiplier=2.0
    )
    .build()
)
```

### System-Aware Optimization

```python
# Get system resource metrics
from src.utils import get_system_resource_metrics, get_optimal_connection_pool_settings

metrics = get_system_resource_metrics()
"""
{
    "cpu_count": 8,
    "cpu_count_logical": 16,
    "memory_gb": 16.0,
    "memory_available_gb": 8.5,
    "os": "Linux"
}
"""

# Get optimal connection pool settings
settings = get_optimal_connection_pool_settings()
"""
{
    "max_connections": 64,
    "max_keepalive_connections": 19,
    "keepalive_expiry": 60
}
"""

# Create an auto-optimized client
client = ReqxClientBuilder().auto_optimize().build()
```
