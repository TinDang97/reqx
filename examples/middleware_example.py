#!/usr/bin/env python
"""
Example demonstrating the middleware system in enhanced-httpx.
This shows how to use request and response middleware for custom processing.
"""

import asyncio
import json
import os
import sys
import time
from typing import Any, Dict

# Add parent directory to path for importing reqx
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import ReqxClient, Response


async def auth_middleware(method: str, url: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Request middleware that adds authentication headers.
    This is a simple example of how you might add an auth token to all requests.
    """
    print(f"Auth Middleware: Adding authentication headers to {method} request to {url}")

    # Clone the kwargs to avoid modifying the original
    new_kwargs = kwargs.copy()

    # Get existing headers or create new ones
    headers = new_kwargs.get("headers", {}).copy()

    # Add your authentication header
    headers["Authorization"] = "Bearer sample-token-1234"

    # Update the kwargs with modified headers
    new_kwargs["headers"] = headers

    return new_kwargs


async def timing_middleware(method: str, url: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Request middleware that stores the start time in the kwargs.
    This will be used by a corresponding response middleware to calculate request duration.
    """
    print(f"Timing Middleware: Starting timer for {method} request to {url}")

    # Store the start time in the kwargs
    # Note: We're not modifying the actual request, just storing metadata
    kwargs["_start_time"] = time.time()

    return kwargs


async def logging_middleware(response: Response) -> Response:
    """
    Response middleware that logs the response status and time taken.
    """
    # Get the start time from the request
    start_time = getattr(response.request, "_start_time", None)

    if start_time:
        duration = time.time() - start_time
        print(f"Logging Middleware: {response.status_code} response received in {duration:.2f}s")
    else:
        print(f"Logging Middleware: {response.status_code} response received (no timing data)")

    return response


async def response_json_middleware(response: Response) -> Response:
    """
    Response middleware that pre-parses JSON responses.
    """
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type and response.status_code < 300:
        try:
            # Pre-parse the JSON (it will be cached for future calls)
            json_data = response.json()
            print(
                f"JSON Middleware: Successfully parsed JSON response with {len(json_data) if isinstance(json_data, dict) else 'non-dict'} fields"
            )
        except Exception as e:
            print(f"JSON Middleware: Failed to parse JSON response: {e}")

    return response


async def main():
    """Main function demonstrating middleware usage."""
    print("Enhanced HTTPX Middleware Example")
    print("===============================\n")

    # Create a client
    async with ReqxClient(debug=True) as client:
        # Add request middlewares (executed in the order they are added)
        client.add_request_middleware(auth_middleware)
        client.add_request_middleware(timing_middleware)

        # Add response middlewares (executed in the order they are added)
        client.add_response_middleware(logging_middleware)
        client.add_response_middleware(response_json_middleware)

        print("\nMaking a GET request to httpbin.org/get...\n")
        # Make a request - the middlewares will be applied automatically
        response = await client.get("https://httpbin.org/get")

        # Display the response
        print("\nResponse:")
        print(f"Status: {response.status_code}")
        print("Headers:")
        for name, value in response.headers.items():
            print(f"  {name}: {value}")

        # The auth header should be visible in the httpbin response
        print("\nRequest info echoed by httpbin:")
        data = response.json()
        print(json.dumps(data, indent=2))

        # Verify our auth middleware worked
        auth_header = data["headers"].get("Authorization")
        assert auth_header == "Bearer sample-token-1234", "Auth middleware didn't work!"
        print("\n✅ Auth middleware successfully added authentication header")

        # Try a POST request
        print("\nMaking a POST request to httpbin.org/post...\n")
        post_response = await client.post(
            "https://httpbin.org/post", json={"name": "Middleware Test", "value": 42}
        )

        print(f"\nPOST Status: {post_response.status_code}")
        post_data = post_response.json()

        # Verify the JSON was sent and the auth header was included
        assert post_data["json"]["name"] == "Middleware Test", "POST JSON data not sent correctly"
        assert (
            post_data["headers"]["Authorization"] == "Bearer sample-token-1234"
        ), "Auth middleware didn't work for POST!"

        print("✅ Middleware pipeline successfully processed both requests")


if __name__ == "__main__":
    asyncio.run(main())
