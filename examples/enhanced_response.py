#!/usr/bin/env python
"""
Example demonstrating the enhanced response functionality of the Enhanced-HTTPX client.

This example shows how to use the EnhancedResponse class to get additional request
information and easily convert responses to Pydantic models.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from src.builder import ReqxClientBuilder
from src.models import EnhancedResponse


class User(BaseModel):
    """Example user model for demonstration."""

    id: int
    name: str
    username: str
    email: str
    phone: Optional[str] = None
    website: Optional[str] = None


class UserApiResponse(BaseModel):
    """Example API response wrapper."""

    data: User
    meta: Dict = Field(default_factory=dict)


async def basic_enhanced_response():
    """Demonstrate basic EnhancedResponse properties."""
    print("\n=== Basic Enhanced Response Example ===\n")

    client = ReqxClientBuilder().build()

    try:
        # Make a request that will return an EnhancedResponse
        response = await client.get("https://jsonplaceholder.typicode.com/users/1")

        # EnhancedResponse is a subclass of httpx.Response with extra features
        print(f"Response type: {type(response)}")
        print(f"Status code: {response.status_code}")
        print(f"Content type: {response.headers.get('content-type')}")

        # Enhanced properties
        print(f"\nRequest time: {response.request_time:.4f} seconds")
        print(f"Is success: {response.is_success}")
        print(f"Is error: {response.is_error}")
        print(f"Cache hit: {response.cache_hit}")
        print(f"Request attempt: {response.request_attempt}")

        # Get transport info if available
        if response.transport_info:
            print(f"\nTransport info: {json.dumps(response.transport_info, indent=2)}")

        # Convert to a dictionary with all metadata
        print("\nResponse dictionary representation:")
        response_dict = response.dict()
        print(json.dumps({k: v for k, v in response_dict.items() if k != "body"}, indent=2))

        # Convert to ResponseModel
        resp_model = response.to_response_model()
        print(f"\nResponseModel.is_success: {resp_model.is_success}")
        print(f"ResponseModel.elapsed: {resp_model.elapsed}")
        print(f"ResponseModel.timestamp: {resp_model.timestamp}")

    finally:
        await client.close()


async def pydantic_model_conversion():
    """Demonstrate converting responses to Pydantic models."""
    print("\n=== Pydantic Model Conversion Example ===\n")

    client = ReqxClientBuilder().build()

    try:
        # Method 1: Convert an EnhancedResponse to a model using to_model
        response = await client.get("https://jsonplaceholder.typicode.com/users/1")
        user = response.to_model(User)
        print(f"User from to_model(): {user.name} ({user.email})")

        # Method 2: Use response_model parameter (client does the conversion)
        user2 = await client.get(
            "https://jsonplaceholder.typicode.com/users/2", response_model=User
        )
        print(f"User from response_model: {user2.name} ({user2.email})")

        # Method 3: Nested models with response wrappers
        wrapped_response = await client.get(
            "https://jsonplaceholder.typicode.com/users/3", response_model=UserApiResponse
        )
        print(f"Wrapped user: {wrapped_response.data.name} ({wrapped_response.data.email})")

    finally:
        await client.close()


async def error_handling():
    """Demonstrate error handling with EnhancedResponse."""
    print("\n=== Error Handling Example ===\n")

    client = ReqxClientBuilder().with_debug(True).build()

    try:
        # Make a request to a non-existent endpoint
        try:
            response = await client.get("https://jsonplaceholder.typicode.com/invalid-endpoint")
            print("This should not be reached due to 404 error")
        except Exception as e:
            print(f"Caught expected error: {type(e).__name__}: {str(e)}")

        # Make a request but disable automatic error raising
        client.transport.raise_for_status = False
        response = await client.get("https://jsonplaceholder.typicode.com/invalid-endpoint")

        print(f"Status code: {response.status_code}")
        print(f"Is error: {response.is_error}")
        print(f"Is client error: {response.is_client_error}")
        print(f"Is server error: {response.is_server_error}")

        # Convert error response to a model - this will raise an exception
        try:
            user = response.to_model(User)
        except Exception as e:
            print(f"Model conversion error: {type(e).__name__}: {str(e)}")

    finally:
        await client.close()


async def retries_and_caching():
    """Demonstrate retry and caching information in responses."""
    print("\n=== Retry and Caching Example ===\n")

    client = (
        ReqxClientBuilder()
        .with_max_retries(2)
        .with_retry_backoff(0.1)
        .with_cache(enabled=True, ttl=10)
        .build()
    )

    try:
        # First request - should be a cache miss
        print("Making first request...")
        response1 = await client.get("https://jsonplaceholder.typicode.com/users/1")
        print(f"Cache hit: {response1.cache_hit}")

        # Second request to same URL - should be a cache hit
        print("\nMaking second request to same URL...")
        response2 = await client.get("https://jsonplaceholder.typicode.com/users/1")
        print(f"Cache hit: {response2.cache_hit}")

        # Try a request that will fail and retry
        print("\nMaking request to an endpoint that will cause retries...")
        bad_url = "https://httpbin.org/status/500"  # This will return a 500 error

        try:
            # This should retry but eventually fail
            client.transport.raise_for_status = False
            response3 = await client.get(bad_url)
            print(f"Final status code: {response3.status_code}")
            print(f"Request attempt: {response3.request_attempt}")  # This should be > 1
            print(f"Retries: {response3.request_retries}")
        except Exception as e:
            print(f"Request failed after retries: {e}")

    finally:
        await client.close()


async def batch_requests():
    """Demonstrate batch requests with EnhancedResponse."""
    print("\n=== Batch Request Example ===\n")

    client = ReqxClientBuilder().build()

    try:
        # Prepare batch requests
        requests = [
            {"method": "GET", "url": "https://jsonplaceholder.typicode.com/users/1"},
            {"method": "GET", "url": "https://jsonplaceholder.typicode.com/users/2"},
            {"method": "GET", "url": "https://jsonplaceholder.typicode.com/posts/1"},
            {
                "method": "POST",
                "url": "https://jsonplaceholder.typicode.com/posts",
                "json": {"title": "Test", "body": "Test post", "userId": 1},
            },
        ]

        # Execute batch requests
        responses = await client.batch_request(requests)

        # Process responses
        for i, response in enumerate(responses):
            if isinstance(response, EnhancedResponse):
                print(f"\nRequest {i+1}:")
                print(f"  URL: {requests[i]['url']}")
                print(f"  Method: {requests[i]['method']}")
                print(f"  Status: {response.status_code}")
                print(f"  Time: {response.request_time:.4f}s")

                # For POST request, show the response body
                if requests[i]["method"] == "POST":
                    print(f"  Response: {json.dumps(response.json(), indent=2)}")
            else:
                print(f"\nRequest {i+1} failed: {response}")

        # Batch with model conversion
        user_responses = await client.batch_request(
            requests=[
                {"method": "GET", "url": "https://jsonplaceholder.typicode.com/users/1"},
                {"method": "GET", "url": "https://jsonplaceholder.typicode.com/users/2"},
            ],
            response_model=User,
        )

        print("\nBatch with model conversion:")
        for i, user in enumerate(user_responses):
            if isinstance(user, User):
                print(f"  User {i+1}: {user.name} ({user.email})")
            else:
                print(f"  User {i+1} conversion failed: {user}")

    finally:
        await client.close()


async def main():
    """Run all demonstration functions."""
    await basic_enhanced_response()
    await pydantic_model_conversion()
    await error_handling()
    await retries_and_caching()
    await batch_requests()


if __name__ == "__main__":
    asyncio.run(main())
