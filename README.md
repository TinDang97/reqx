# reqx: High-Performance HTTP Client Library

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

reqx is a modern, high-performance HTTP client library for Python, extending [httpx](https://github.com/encode/httpx) with advanced features and optimizations for demanding use cases.

## Key Features

- **Multiple Transport Support:** Switches between httpx and aiohttp for optimal protocol compatibility and speed.
- **Smart Protocol Selection:** Automatically chooses HTTP/1.1, HTTP/2, or HTTP/3 per request.
- **Adaptive Timeouts:** Dynamically adjusts timeouts based on request history.
- **System-Aware Configuration:** Tunes connection pools and networking for your system.
- **Enhanced Builder Pattern:** Fluent API for configuring clients with preset profiles.
- **Middleware Pipeline:** Extensible architecture for request/response processing.
- **Advanced Retries:** Exponential backoff, jitter, and status-based retry policies.
- **Batch Requests:** Parallel request execution with efficient resource management.
- **Streaming Support:** Handles large requests/responses efficiently.
- **GraphQL Integration:** Native support for GraphQL queries.
- **Webhook Management:** Simple webhook creation and management.
- **Settings Persistence:** Retains learned optimizations across restarts.

## Installation

```bash
pip install reqx
```

## Quick Start

### Basic Usage

```python
from reqx import ReqxClient

client = ReqxClient()

# Synchronous request
response = client.get("https://api.example.com/users")
print(response.json())

# Asynchronous request
async def fetch_data():
    async with ReqxClient() as client:
        response = await client.get("https://api.example.com/users")
        return response.json()
```

### Advanced Configuration

```python
from reqx import ReqxClientBuilder

client = (
    ReqxClientBuilder()
    .with_base_url("https://api.example.com")
    .with_http2(True)
    .with_adaptive_timeout(True)
    .for_high_performance()
    .with_retry(max_retries=3)
    .with_middleware(LoggingMiddleware())
    .with_persistence(True)
    .build()
)
```

### Preset Profiles

```python
client = ReqxClientBuilder().for_high_performance().build()
client = ReqxClientBuilder().for_reliability().build()
client = ReqxClientBuilder().auto_optimize().build()
```

## Advanced Features

### Hybrid Transport Selection

```python
client = (
    ReqxClientBuilder()
    .use_aiohttp(True)
    .with_http2(True)
    .auto_optimize()
    .build()
)
```
- HTTP/1.1 endpoints use aiohttp (faster for HTTP/1.1)
- HTTP/2 endpoints use httpx (faster for HTTP/2)

### Adaptive Timeout Management

```python
client = (
    ReqxClientBuilder()
    .with_adaptive_timeout(True)
    .build()
)
```
Timeouts adjust automatically per host based on performance.

### Settings Persistence

```python
client = (
    ReqxClientBuilder()
    .with_adaptive_timeout(True)
    .with_persistence(True)
    .build()
)
```
Settings are saved and reused across runs.

Custom persistence path:

```python
client = (
    ReqxClientBuilder()
    .with_adaptive_timeout(True)
    .with_persistence(True, "/path/to/settings")
    .build()
)
```

Benchmark persistence:

```bash
python -m scripts.benchmark_persistence --url https://api.example.com --iterations 5 --plot
```

### Middleware Pipeline

```python
from reqx import Middleware

class MyMiddleware(Middleware):
    async def process_request(self, ctx, next_middleware):
        ctx.headers["X-Custom"] = "Value"
        response = await next_middleware(ctx)
        response.metadata["custom_key"] = "processed"
        return response

client = ReqxClientBuilder().with_middleware(MyMiddleware()).build()
```

### Batch Requests

```python
responses = await client.batch_request([
    {"method": "GET", "url": "https://api.example.com/users/1"},
    {"method": "GET", "url": "https://api.example.com/users/2"},
    {"method": "POST", "url": "https://api.example.com/data", "json": {"key": "value"}}
])

for resp in responses:
    print(f"Status: {resp.status_code}, Data: {resp.json()}")
```

### GraphQL Integration

```python
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

user_data = response.json()["data"]["user"]
```

## Contributing

Contributions are welcome! See the [Contributing Guide](CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Experiment

Running batch benchmarks against http://localhost:80
Batch size: 100
HTTP method: GET
Concurrent requests: 10000
Runs per client: 3
Clients: reqx, httpx, aiohttp

| Client        | Requests | Duration (s) | Req/sec | Avg Request (ms) | Memory (KB) |
|---------------|----------|--------------|---------|------------------|-------------|
| aiohttp_batch |   10000  |    6.38      | 1567.53 | 0.64             | 3807.67     |
| reqx_batch    |   10000  |    7.915     | 1271.68 | 0.79             | 4872.33     |
| httpx_batch   |   10000  |   26.264     | 382.02  | 2.63             | 5777.67     |

