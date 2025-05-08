# Enhanced HTTPX

An enhanced HTTP client library built on top of [httpx](https://www.python-httpx.org/) for making custom HTTP requests with async support. This library provides a convenient and powerful API for handling HTTP requests with features like connection pooling, automatic retries, JSON path selectors, and more.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Features

- ✅ **Fully asynchronous** HTTP client with `async`/`await` syntax
- ✅ **Automatic retries** with configurable backoff strategy
- ✅ **Connection pooling** for optimal performance
- ✅ **JSON path selectors** to extract specific data from responses
- ✅ **Type validation** with Pydantic models
- ✅ **Advanced error handling** with specific exception types
- ✅ **Fast JSON serialization** with orjson
- ✅ **CLI interface** for quick HTTP requests from the terminal
- ✅ **Security features** with proper SSL/TLS configuration
- ✅ **Performance optimizations** with uvloop

## Installation

Enhanced HTTPX requires Python 3.8+ and can be installed using `uv`:

```bash
uv pip install enhanced-httpx
```

Or using pip:

```bash
pip install enhanced-httpx
```

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
    debug=False
)
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