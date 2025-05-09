#!/usr/bin/env python
"""
Example demonstrating various async request capabilities of the enhanced-httpx library.
This example shows handling multiple concurrent requests, using response models,
request retries, and JSON path extraction.
"""

import asyncio
import os
import sys
import time
from typing import Optional

from pydantic import BaseModel

# Add parent directory to path for importing enhanced_httpx
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import EnhancedClient, select_json_path


# Define Pydantic models for API responses
class User(BaseModel):
    id: int
    name: str
    email: str
    username: Optional[str] = None
    phone: Optional[str] = None


class Post(BaseModel):
    id: int
    userId: int
    title: str
    body: str


class TodoItem(BaseModel):
    id: int
    userId: int
    title: str
    completed: bool


class Comment(BaseModel):
    id: int
    postId: int
    name: str
    email: str
    body: str


async def fetch_user_with_posts(client: EnhancedClient, user_id: int):
    """Fetch a user and their posts."""
    print(f"\n--- Fetching user {user_id} with posts ---")

    # Fetch user and posts concurrently
    user_task = client.get(
        f"https://jsonplaceholder.typicode.com/users/{user_id}", response_model=User
    )
    posts_task = client.get(
        "https://jsonplaceholder.typicode.com/posts", params={"userId": str(user_id)}
    )

    # Wait for both requests to complete
    user, posts_response = await asyncio.gather(user_task, posts_task)

    # Parse posts response
    posts_data = posts_response.json()
    posts = [Post(**post) for post in posts_data]

    # Print results
    print(f"User: {user.name} ({user.email})")
    print(f"Posts: {len(posts)}")
    for i, post in enumerate(posts[:3], 1):
        print(f"  {i}. {post.title[:50]}...")
    if len(posts) > 3:
        print(f"  ... and {len(posts) - 3} more posts")

    return user, posts


async def fetch_with_json_path(client: EnhancedClient):
    """Demonstrate using JSONPath to extract specific data from responses."""
    print("\n--- Using JSONPath to extract data ---")

    # Make a request
    response = await client.get("https://jsonplaceholder.typicode.com/users")
    users_data = response.json()

    # Extract specific fields using JSON path
    emails = select_json_path(users_data, "$[*].email")
    companies = select_json_path(users_data, "$[*].company.name")
    geo_coords = select_json_path(users_data, "$[*].address.geo")

    # Print results
    print("User emails:")
    for email in emails[:5]:
        print(f"  - {email}")

    print("\nCompany names:")
    for company in companies[:5]:
        print(f"  - {company}")

    print("\nGeo coordinates:")
    for coords in geo_coords[:3]:
        print(f"  - Lat: {coords['lat']}, Lng: {coords['lng']}")


async def demonstrate_retry_mechanism(client: EnhancedClient):
    """Demonstrate the retry mechanism."""
    print("\n--- Demonstrating retry mechanism ---")

    # Configure client with custom retry settings
    retry_client = EnhancedClient(
        timeout=2.0,  # Short timeout to trigger faster
        max_retries=3,
        retry_backoff=0.5,
        debug=True,  # Enable debug logging
    )

    print("Making request to a slow endpoint (should retry)...")
    try:
        # This will likely time out and retry
        await retry_client.get("https://httpbin.org/delay/3")
    except Exception as e:
        print(f"Request failed after retries: {str(e)}")

    # Close the client
    await retry_client.close()


async def batch_requests_with_rate_limiting(client: EnhancedClient):
    """Demonstrate batch requests with rate limiting."""
    print("\n--- Batch requests with rate limiting ---")

    # Number of requests to make
    num_requests = 10
    print(f"Making {num_requests} requests with rate limiting...")

    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(3)  # Maximum 3 concurrent requests

    async def fetch_todo(todo_id):
        async with semaphore:
            print(f"Fetching todo {todo_id}...")
            response = await client.get(
                f"https://jsonplaceholder.typicode.com/todos/{todo_id}", response_model=TodoItem
            )
            await asyncio.sleep(0.2)  # Simulate rate limiting
            return response

    # Create tasks for all requests
    tasks = [fetch_todo(i) for i in range(1, num_requests + 1)]

    # Execute all tasks concurrently (but limited by the semaphore)
    start_time = time.time()
    todos = await asyncio.gather(*tasks)
    elapsed = time.time() - start_time

    # Print results
    print(f"Completed {num_requests} requests in {elapsed:.2f} seconds")
    print("Sample results:")
    for todo in todos[:3]:
        status = "✅ Completed" if todo.completed else "❌ Not completed"
        print(f"  - Todo #{todo.id}: {todo.title[:30]}... ({status})")


async def main():
    """Main function to run all examples."""
    print("Enhanced HTTPX Library Examples")
    print("==============================\n")

    # Create a shared client
    async with EnhancedClient() as client:
        # Example 1: Fetch a user with their posts
        await fetch_user_with_posts(client, 1)

        # Example 2: Using JSONPath to extract specific data
        await fetch_with_json_path(client)

        # Example 3: Batch requests with rate limiting
        await batch_requests_with_rate_limiting(client)

    # Example 4: Demonstrate retry mechanism with a separate client
    await demonstrate_retry_mechanism(EnhancedClient())

    print("\nAll examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
