#!/usr/bin/env python
"""
Example demonstrating middleware capabilities in Enhanced-HTTPX.

This example shows how to create custom middleware for:
1. Request/response logging
2. Automatic retries with backoff
3. Authentication
4. Caching
5. Performance monitoring
"""

import asyncio
import time
import json
import logging
from datetime import datetime
from typing import Dict, Callable, Optional, Any

from src.builder import ReqxClientBuilder
from src.middleware import Middleware
from src.models import RequestContext, ResponseContext


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("middleware-example")


class LoggingMiddleware(Middleware):
    """Middleware that logs all requests and responses."""

    def __init__(self, log_level: int = logging.INFO):
        self.log_level = log_level
        self.logger = logging.getLogger("reqx.logging")

    async def process_request(
        self, ctx: RequestContext, next_middleware: Callable
    ) -> ResponseContext:
        """Process and log the request and response."""
        request_id = f"req-{int(time.time() * 1000)}"

        # Add request ID to context metadata
        if not ctx.metadata:
            ctx.metadata = {}
        ctx.metadata["request_id"] = request_id

        self.logger.log(self.log_level, f"[{request_id}] Request: {ctx.method} {ctx.url}")

        start_time = time.time()
        try:
            # Pass to next middleware
            response_ctx = await next_middleware(ctx)

            # Log response
            duration = time.time() - start_time
            self.logger.log(
                self.log_level,
                f"[{request_id}] Response: {response_ctx.status_code} " f"({duration:.3f}s)",
            )

            # Add timing info to response metadata
            if not response_ctx.metadata:
                response_ctx.metadata = {}
            response_ctx.metadata["duration"] = duration
            response_ctx.metadata["request_id"] = request_id

            return response_ctx

        except Exception as e:
            duration = time.time() - start_time
            self.logger.log(logging.ERROR, f"[{request_id}] Error: {str(e)} ({duration:.3f}s)")
            raise


class CachingMiddleware(Middleware):
    """Simple in-memory caching middleware."""

    def __init__(self, ttl: int = 60):
        """Initialize with time-to-live in seconds."""
        self.cache: Dict[str, Dict] = {}
        self.ttl = ttl

    def _cache_key(self, ctx: RequestContext) -> str:
        """Generate a cache key from the request."""
        # Only cache GET requests
        if ctx.method.upper() != "GET":
            return None

        params_str = ""
        if ctx.params:
            params_str = "&".join(f"{k}={v}" for k, v in sorted(ctx.params.items()))

        return f"{ctx.method}:{ctx.url}:{params_str}"

    async def process_request(
        self, ctx: RequestContext, next_middleware: Callable
    ) -> ResponseContext:
        """Process request with caching."""
        cache_key = self._cache_key(ctx)

        # Skip caching for non-GET requests
        if not cache_key:
            return await next_middleware(ctx)

        # Check if we have a cached response
        now = time.time()
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if now < entry["expires_at"]:
                # Return cached response
                logger.info(f"Cache hit for {ctx.method} {ctx.url}")

                response = entry["response"]
                if not response.metadata:
                    response.metadata = {}
                response.metadata["cached"] = True
                return response

        # Cache miss, proceed with the request
        response = await next_middleware(ctx)

        # Cache the response if it's successful
        if 200 <= response.status_code < 300:
            if not response.metadata:
                response.metadata = {}

            expires_at = now + self.ttl
            self.cache[cache_key] = {"response": response, "expires_at": expires_at}
            logger.info(
                f"Cached response for {ctx.method} {ctx.url} until {datetime.fromtimestamp(expires_at)}"
            )
            response.metadata["cached"] = False

        return response


class RetryMiddleware(Middleware):
    """Middleware that automatically retries failed requests."""

    def __init__(
        self,
        max_retries: int = 3,
        retry_status_codes: list = None,
        retry_exceptions: tuple = None,
        backoff_factor: float = 0.3,
    ):
        self.max_retries = max_retries
        self.retry_status_codes = retry_status_codes or [429, 500, 502, 503, 504]
        self.retry_exceptions = retry_exceptions or (ConnectionError, TimeoutError)
        self.backoff_factor = backoff_factor
        self.logger = logging.getLogger("reqx.retry")

    async def process_request(
        self, ctx: RequestContext, next_middleware: Callable
    ) -> ResponseContext:
        """Process request with automatic retries."""
        retries = 0

        while True:
            try:
                response = await next_middleware(ctx)

                # Check if we should retry based on status code
                if response.status_code in self.retry_status_codes and retries < self.max_retries:
                    retries += 1
                    wait_time = self.backoff_factor * (2 ** (retries - 1))
                    self.logger.info(
                        f"Retrying request ({retries}/{self.max_retries}) after "
                        f"{wait_time:.2f}s due to status {response.status_code}"
                    )
                    await asyncio.sleep(wait_time)
                    continue

                return response

            except self.retry_exceptions as e:
                if retries >= self.max_retries:
                    self.logger.error(f"Max retries ({self.max_retries}) exceeded: {str(e)}")
                    raise

                retries += 1
                wait_time = self.backoff_factor * (2 ** (retries - 1))
                self.logger.info(
                    f"Retrying request ({retries}/{self.max_retries}) after "
                    f"{wait_time:.2f}s due to error: {str(e)}"
                )
                await asyncio.sleep(wait_time)


class AuthenticationMiddleware(Middleware):
    """Middleware that handles authentication."""

    def __init__(
        self,
        auth_strategy: str = "bearer_token",
        token: str = None,
        token_provider: Callable = None,
    ):
        self.auth_strategy = auth_strategy
        self.token = token
        self.token_provider = token_provider

    async def _get_token(self) -> str:
        """Get the authentication token."""
        if self.token_provider:
            return await self.token_provider()
        return self.token

    async def process_request(
        self, ctx: RequestContext, next_middleware: Callable
    ) -> ResponseContext:
        """Add authentication to request."""
        token = await self._get_token()

        if not ctx.headers:
            ctx.headers = {}

        if self.auth_strategy == "bearer_token":
            ctx.headers["Authorization"] = f"Bearer {token}"
        elif self.auth_strategy == "api_key":
            ctx.headers["X-API-Key"] = token

        return await next_middleware(ctx)


class MetricsMiddleware(Middleware):
    """Middleware that collects performance metrics."""

    def __init__(self):
        self.metrics = {
            "requests": 0,
            "errors": 0,
            "request_times": [],
            "status_counts": {},
            "endpoints": {},
        }

    async def process_request(
        self, ctx: RequestContext, next_middleware: Callable
    ) -> ResponseContext:
        """Collect metrics for the request."""
        start_time = time.time()
        self.metrics["requests"] += 1

        # Extract endpoint from URL
        path = ctx.url.split("://", 1)[-1].split("/", 1)[-1] if "/" in ctx.url else ""
        endpoint = f"{ctx.method} /{path}"

        if endpoint not in self.metrics["endpoints"]:
            self.metrics["endpoints"][endpoint] = {"count": 0, "errors": 0, "times": []}

        try:
            response = await next_middleware(ctx)

            # Update metrics
            duration = time.time() - start_time
            self.metrics["request_times"].append(duration)
            self.metrics["endpoints"][endpoint]["count"] += 1
            self.metrics["endpoints"][endpoint]["times"].append(duration)

            # Count status codes
            status = str(response.status_code)
            if status not in self.metrics["status_counts"]:
                self.metrics["status_counts"][status] = 0
            self.metrics["status_counts"][status] += 1

            return response

        except Exception as e:
            # Count errors
            duration = time.time() - start_time
            self.metrics["errors"] += 1
            self.metrics["endpoints"][endpoint]["errors"] += 1
            self.metrics["endpoints"][endpoint]["times"].append(duration)
            raise

    def get_metrics(self) -> Dict:
        """Get the collected metrics."""
        total_requests = self.metrics["requests"]
        avg_time = (
            sum(self.metrics["request_times"]) / len(self.metrics["request_times"])
            if self.metrics["request_times"]
            else 0
        )

        # Calculate endpoint stats
        for endpoint, data in self.metrics["endpoints"].items():
            if data["times"]:
                data["avg_time"] = sum(data["times"]) / len(data["times"])
            else:
                data["avg_time"] = 0
            data["success_rate"] = 1 - (data["errors"] / data["count"]) if data["count"] > 0 else 0

        return {
            "total_requests": total_requests,
            "total_errors": self.metrics["errors"],
            "error_rate": self.metrics["errors"] / total_requests if total_requests > 0 else 0,
            "avg_request_time": avg_time,
            "status_code_distribution": self.metrics["status_counts"],
            "endpoints": self.metrics["endpoints"],
        }


async def demonstrate_middleware():
    """Demonstrate the various middleware capabilities."""
    print("===== Enhanced-HTTPX Middleware Example =====\n")

    # Create the metrics middleware to collect stats
    metrics_middleware = MetricsMiddleware()

    # Create a client with all middleware
    client = (
        ReqxClientBuilder()
        .with_middleware(LoggingMiddleware())
        .with_middleware(RetryMiddleware(max_retries=2, backoff_factor=0.5))
        .with_middleware(CachingMiddleware(ttl=30))
        .with_middleware(
            AuthenticationMiddleware(auth_strategy="bearer_token", token="dummy_token_for_demo")
        )
        .with_middleware(metrics_middleware)
        .build()
    )

    try:
        print("Making requests with middleware chain...\n")

        # Make a few requests to different endpoints
        urls = [
            "https://httpbin.org/get",
            "https://httpbin.org/status/429",  # Will trigger retry
            "https://httpbin.org/get?param=1",  # Will be cached
        ]

        # Make initial requests
        for url in urls:
            try:
                print(f"Requesting {url}")
                response = await client.get(url)
                print(f"  Status: {response.status_code}")

                # Show if the response was cached
                if response.metadata and response.metadata.get("cached") is not None:
                    print(f"  Cached: {response.metadata['cached']}")

                # Show request ID from logging middleware
                if response.metadata and "request_id" in response.metadata:
                    print(f"  Request ID: {response.metadata['request_id']}")

            except Exception as e:
                print(f"  Error: {str(e)}")
            print()

        # Make the same requests again to demonstrate caching
        print("Making the same requests again to demonstrate caching...")
        for url in urls:
            try:
                print(f"Requesting {url}")
                response = await client.get(url)
                print(f"  Status: {response.status_code}")

                # Show if the response was cached
                if response.metadata and response.metadata.get("cached") is not None:
                    print(f"  Cached: {response.metadata['cached']}")

            except Exception as e:
                print(f"  Error: {str(e)}")
            print()

        # Display collected metrics
        print("\n===== Request Metrics =====")
        metrics = metrics_middleware.get_metrics()
        print(f"Total requests: {metrics['total_requests']}")
        print(f"Error rate: {metrics['error_rate'] * 100:.1f}%")
        print(f"Average request time: {metrics['avg_request_time'] * 1000:.2f}ms")

        print("\nStatus code distribution:")
        for status, count in metrics["status_counts"].items():
            print(f"  {status}: {count}")

        print("\nEndpoint statistics:")
        for endpoint, data in metrics["endpoints"].items():
            print(f"  {endpoint}:")
            print(f"    Requests: {data['count']}")
            print(f"    Success rate: {data['success_rate'] * 100:.1f}%")
            print(f"    Avg time: {data['avg_time'] * 1000:.2f}ms")

    finally:
        await client.close()


async def main():
    """Run the middleware example."""
    await demonstrate_middleware()


if __name__ == "__main__":
    asyncio.run(main())
