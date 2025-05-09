# Enhanced HTTPX

An enhanced HTTP client library built on top of [httpx](https://www.python-httpx.org/) for making custom HTTP requests with async support. This library provides a convenient and powerful API for handling HTTP requests with features like connection pooling, automatic retries, JSON path selectors, and more.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Features

- ✅ **Fully asynchronous** HTTP client with `async`/`await` syntax
- ✅ **Automatic retries** with configurable backoff strategy
- ✅ **Connection pooling** for optimal performance
- ✅ **HTTP/2 and HTTP/3 support** for improved performance
- ✅ **Response caching** with configurable TTL
- ✅ **Rate limiting** with token bucket algorithm
- ✅ **Middleware support** for request/response processing
- ✅ **Batch requests** with concurrency controls
- ✅ **JSON path selectors** to extract specific data from responses
- ✅ **Type validation** with Pydantic models
- ✅ **Advanced error handling** with specific exception types
- ✅ **Fast JSON serialization** with orjson
- ✅ **CLI interface** for quick HTTP requests from the terminal
- ✅ **Security features** with proper SSL/TLS configuration
- ✅ **Performance optimizations** with uvloop
- ✅ **Metrics collection** for request performance analysis

## Installation

Enhanced HTTPX requires Python 3.8+ and can be installed using `uv`:

```bash
uv pip install enhanced-httpx
```

Or using pip:

```bash
pip install enhanced-httpx
```

### Optional Dependencies

To install with optional features:

```bash
# For Brotli compression support
pip install enhanced-httpx[compression]

# For development tools
pip install enhanced-httpx[dev]

# For multiple extras
pip install enhanced-httpx[compression,dev]
```

## Benchmark Results

| Client         | Requests | Duration (s) | Req/sec   | Avg Request (ms) | Memory (KB) |
|----------------|----------|--------------|-----------|------------------|-------------|
| enhanced_httpx | 100      | 0.004        | 24,736.7  | 0.04             | 2,486       |
| aiohttp        | 100      | 2.286        | 43.87     | 22.86            | 27,882      |
| httpx          | 100      | 2.599        | 39.67     | 25.99            | 3,416.33    |

**Relative Performance:**

- `enhanced_httpx` is 99.8% faster than `aiohttp`
- `enhanced_httpx` is 99.8% faster than `httpx`

**Memory Efficiency:**

- `enhanced_httpx` uses 1,021.6% less memory than `aiohttp`
- `enhanced_httpx` uses 37.4% less memory than `httpx`

_Results exported to `benchmark_results.json`_
_*note: no cache applied_

## Quick Start

### Basic Usage

```python
import asyncio
from enhanced_httpx import EnhancedClient

async def main():
    # Create a client
    async with EnhancedClient() as client:
        # Make a GET request
        response = await client.get("https://httpbin.org/get")
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.json()}")

        # Make a POST request with JSON body
        post_response = await client.post(
            "https://httpbin.org/post",
            json={"name": "John", "age": 30}
        )
        print(f"POST response: {post_response.json()}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Using Response Models

You can automatically parse responses into Pydantic models:

```python
import asyncio
from pydantic import BaseModel
from enhanced_httpx import EnhancedClient

class User(BaseModel):
    id: int
    name: str
    email: str

async def main():
    async with EnhancedClient() as client:
        # Parse response directly into a model
        user = await client.get(
            "https://jsonplaceholder.typicode.com/users/1",
            response_model=User
        )

        print(f"User: {user.name} ({user.email})")

if __name__ == "__main__":
    asyncio.run(main())
```

### JSON Path Selector

Extract specific data from JSON responses using JSONPath syntax:

```python
import asyncio
from enhanced_httpx import EnhancedClient, select_json_path

async def main():
    async with EnhancedClient() as client:
        response = await client.get("https://jsonplaceholder.typicode.com/users")
        data = response.json()

        # Extract all user emails
        emails = select_json_path(data, "$[*].email")
        print("User emails:", emails)

        # Extract nested data
        companies = select_json_path(data, "$[*].company.name")
        print("Company names:", companies)

if __name__ == "__main__":
    asyncio.run(main())
```

### Error Handling

The library provides detailed exceptions for different error cases:

```python
import asyncio
from enhanced_httpx import EnhancedClient
from enhanced_httpx.exceptions import (
    NotFoundError, ServerError, TimeoutError
)

async def main():
    async with EnhancedClient() as client:
        try:
            # This will raise a NotFoundError
            await client.get("https://httpbin.org/status/404")
        except NotFoundError as e:
            print(f"Resource not found: {e}")

        try:
            # This will raise a ServerError
            await client.get("https://httpbin.org/status/500")
        except ServerError as e:
            print(f"Server error: {e}")

        try:
            # This will likely raise a TimeoutError
            await client.get(
                "https://httpbin.org/delay/10",
                timeout=2.0
            )
        except TimeoutError as e:
            print(f"Request timed out: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Advanced Configuration

You can configure the client with various options:

```python
from enhanced_httpx import EnhancedClient

# Create a client with custom configuration
client = EnhancedClient(
    base_url="https://api.example.com",
    timeout=30.0,
    max_connections=100,
    max_keepalive_connections=20,
    keepalive_expiry=60,
    follow_redirects=True,
    verify_ssl=True,
    max_retries=3,
    retry_backoff=0.5,
    http2=True,
    enable_http3=False,  # Enable HTTP/3 support
    debug=False,
    enable_cache=True,   # Enable response caching
    cache_ttl=300,       # Cache TTL in seconds
    rate_limit=100,      # Rate limit (requests per second)
)
```

### Request Compression

Enhanced HTTPX supports request compression using various algorithms. For Brotli compression, install the optional dependency:

```bash
pip install enhanced-httpx[compression]
```

Then use the CompressionMiddleware:

```python
import asyncio
from enhanced_httpx import EnhancedClient
from enhanced_httpx.middleware import CompressionMiddleware, MiddlewareChain

async def main():
    # Create middleware chain with compression
    middleware = MiddlewareChain()
    middleware.add(CompressionMiddleware(
        compress_requests=True,
        min_size_to_compress=1024,  # Only compress bodies larger than 1KB
        compress_type="br",  # Use Brotli compression
        accept_compressed_responses=True
    ))

    # Create client with middleware
    async with EnhancedClient() as client:
        client.middleware = middleware

        # The request body will be automatically compressed
        response = await client.post(
            "https://httpbin.org/post",
            json={"large": "data" * 1000}  # Large enough to trigger compression
        )
        print(response.json())

if __name__ == "__main__":
    asyncio.run(main())
```

### Using Middleware

You can add middleware functions to modify requests before they are sent or responses before they are returned:

```python
import asyncio
from enhanced_httpx import EnhancedClient

# Define a request middleware
async def add_auth_header(method, url, request_kwargs):
    headers = request_kwargs.get("headers", {})
    headers["Authorization"] = "Bearer token123"
    request_kwargs["headers"] = headers
    return request_kwargs

# Define a response middleware
async def log_response_time(response):
    print(f"Request to {response.url} took {response.elapsed.total_seconds():.2f}s")
    return response

async def main():
    # Create a client with middleware
    async with EnhancedClient() as client:
        # Add middleware
        client.add_request_middleware(add_auth_header)
        client.add_response_middleware(log_response_time)

        # Make a request - middleware will be applied
        response = await client.get("https://httpbin.org/get")
        print(response.json())

if __name__ == "__main__":
    asyncio.run(main())
```

### Batch Requests

Process multiple requests concurrently with batch mode:

```python
import asyncio
from enhanced_httpx import EnhancedClient, BatchRequestItem

async def main():
    async with EnhancedClient() as client:
        # Create a batch of requests
        batch = client.create_batch()

        # Add requests to the batch
        batch.add_request("GET", "https://httpbin.org/get")
        batch.add_request("POST", "https://httpbin.org/post", json={"name": "John"})
        batch.add_request("GET", "https://httpbin.org/delay/1")

        # Execute all requests concurrently (with max_connections limit)
        responses = await client.execute_batch(batch, max_connections=5)

        # Process responses
        for i, response in enumerate(responses):
            print(f"Response {i+1} status: {response.status_code}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Caching Responses

Configure caching for repeated requests:

```python
import asyncio
from enhanced_httpx import EnhancedClient

async def main():
    # Enable caching with 60 second TTL
    async with EnhancedClient(enable_cache=True, cache_ttl=60) as client:
        # First request will be sent to server
        response1 = await client.get("https://httpbin.org/get")
        print("First request:", response1.elapsed.total_seconds())

        # Second request will use cached response (much faster)
        response2 = await client.get("https://httpbin.org/get")
        print("Second request (cached):", response2.elapsed.total_seconds())

        # Force refresh cache for this request
        response3 = await client.get("https://httpbin.org/get", force_refresh=True)
        print("Third request (forced refresh):", response3.elapsed.total_seconds())

if __name__ == "__main__":
    asyncio.run(main())
```

## Command Line Interface

Enhanced HTTPX comes with a CLI for making HTTP requests from the terminal:

```bash
# Basic GET request
enhanced-httpx get https://httpbin.org/get

# POST with JSON data
enhanced-httpx post https://httpbin.org/post -j '{"name": "John", "age": 30}'

# Custom headers and parameters
enhanced-httpx get https://httpbin.org/get -H "Authorization=Bearer token123" -p "query=test"

# Extract specific data with JSONPath
enhanced-httpx get https://jsonplaceholder.typicode.com/users --json-path "$[*].email"

# Save response to file
enhanced-httpx get https://httpbin.org/json --pretty --save response.json

# Debug mode
enhanced-httpx --debug get https://httpbin.org/headers
```

## Performance

Enhanced HTTPX is designed for performance:

- Uses `uvloop` for an optimized event loop implementation
- Fast JSON serialization/deserialization with `orjson`
- Efficient connection pooling
- Smart retry mechanisms

You can run the benchmark script to compare performance with other HTTP clients:

```bash
python scripts/benchmark.py
```

## Development

To set up the development environment:

```bash
# Clone the repository
git clone https://github.com/yourusername/enhanced-httpx.git
cd enhanced-httpx

# Install the package in development mode
uv pip install -e ".[dev]"

# Run tests
python scripts/run_tests.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
