#!/usr/bin/env python
"""
Example demonstrating the use of persistence in Enhanced-HTTPX.

This example shows how to:
1. Configure persistence to preserve learned settings
2. Observe the performance improvements from adaptive optimizations
3. View statistics about learned timeouts
"""

import asyncio
import os
import pathlib
import time
from typing import Dict, List

import sys

# Add the parent directory to the path so we can import the library
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.builder import ReqxClientBuilder


async def run_example_requests(client, urls: List[str], verbose: bool = True):
    """Run a series of requests and display timing information."""
    start_time = time.time()
    responses = []

    # Make requests sequentially for demonstration purposes
    for url in urls:
        if verbose:
            req_start = time.time()
            print(f"Requesting {url}...")

        try:
            response = await client.get(url, timeout=5.0)
            responses.append(response)

            if verbose:
                req_time = time.time() - req_start
                print(f"  -> Completed in {req_time:.4f}s")
        except Exception as e:
            print(f"  -> Error: {str(e)}")

    total_time = time.time() - start_time
    print(f"\nCompleted {len(urls)} requests in {total_time:.4f} seconds")
    print(f"Average request time: {total_time / len(urls):.4f} seconds")

    return responses


async def demonstrate_persistence():
    """
    Demonstrate the benefits of persistence for optimizing performance.

    This example simulates multiple sessions of an application,
    showing how persistence allows the client to maintain optimizations
    across restarts.
    """
    print("\n=== PERSISTENCE DEMONSTRATION ===\n")

    # Create a temporary directory for persistence data
    persistence_dir = pathlib.Path.home() / ".enhanced_httpx_demo"
    persistence_dir.mkdir(parents=True, exist_ok=True)
    print(f"Storing persistence data in {persistence_dir}")

    # List of URLs to request in each session
    test_urls = [
        "https://httpbin.org/get",
        "https://httpbin.org/headers",
        "https://api.github.com/zen",
        "https://api.github.com/octocat",
        "https://jsonplaceholder.typicode.com/posts/1",
        "https://jsonplaceholder.typicode.com/users/1",
        "https://httpbin.org/delay/1",  # This one is slower and will benefit from adaptive timeout
    ]

    # First session: without persistence enabled
    print("\n--- Session 1: Without Persistence ---")
    client1 = (
        ReqxClientBuilder()
        .with_adaptive_timeout(True)  # Enable adaptive timeouts
        .with_persistence(False)  # But disable persistence
        .build()
    )

    await run_example_requests(client1, test_urls)

    # Show timeout statistics at the end of the first session
    stats = client1.get_timeout_statistics()
    print("\nAdaptive timeout statistics after first session:")
    for host, host_stats in stats.items():
        print(f"  {host}:")
        print(f"    Current timeout: {host_stats['current_timeout']:.2f}s")
        print(f"    Average duration: {host_stats['avg_duration']:.4f}s")
        print(f"    Success rate: {host_stats['success_rate']:.2%}")

    await client1.close()

    # Second session: with persistence enabled
    print("\n--- Session 2: With Persistence ---")
    client2 = (
        ReqxClientBuilder()
        .with_adaptive_timeout(True)
        .with_persistence(True, persistence_dir)  # Enable persistence and specify path
        .build()
    )

    await run_example_requests(client2, test_urls)

    # Show timeout statistics at the end of the second session
    stats = client2.get_timeout_statistics()
    print("\nAdaptive timeout statistics after second session:")
    for host, host_stats in stats.items():
        print(f"  {host}:")
        print(f"    Current timeout: {host_stats['current_timeout']:.2f}s")
        print(f"    Average duration: {host_stats['avg_duration']:.4f}s")
        print(f"    Success rate: {host_stats['success_rate']:.2%}")

    await client2.close()

    # Third session: with persistence enabled again, showing the benefits
    print("\n--- Session 3: With Persistence (showing benefits) ---")
    client3 = (
        ReqxClientBuilder()
        .with_adaptive_timeout(True)
        .with_persistence(True, persistence_dir)  # Use the same persistence path
        .build()
    )

    await run_example_requests(client3, test_urls)

    # Show timeout statistics at the end of the third session
    stats = client3.get_timeout_statistics()
    print("\nAdaptive timeout statistics after third session:")
    for host, host_stats in stats.items():
        print(f"  {host}:")
        print(f"    Current timeout: {host_stats['current_timeout']:.2f}s")
        print(f"    Average duration: {host_stats['avg_duration']:.4f}s")
        print(f"    Success rate: {host_stats['success_rate']:.2%}")
        print(f"    Samples: {host_stats['samples']}")  # Show accumulated samples

    await client3.close()

    # Final comparison: run a benchmark to compare with and without persistence
    print("\n--- Final Benchmark: With vs. Without Persistence ---")
    print("\nWithout Persistence:")
    client_without = ReqxClientBuilder().with_adaptive_timeout(True).build()
    start = time.time()
    await run_example_requests(client_without, test_urls, verbose=False)
    time_without = time.time() - start
    await client_without.close()

    print("\nWith Persistence:")
    client_with = (
        ReqxClientBuilder()
        .with_adaptive_timeout(True)
        .with_persistence(True, persistence_dir)
        .build()
    )
    start = time.time()
    await run_example_requests(client_with, test_urls, verbose=False)
    time_with = time.time() - start
    await client_with.close()

    # Calculate improvement
    improvement = (time_without - time_with) / time_without * 100

    print("\n=== RESULTS ===")
    print(f"Without persistence: {time_without:.4f}s")
    print(f"With persistence: {time_with:.4f}s")
    print(f"Improvement: {improvement:.2f}%")

    print("\nPersistence allows enhanced-httpx to preserve learned timeout settings")
    print("and other optimizations across application restarts, providing better")
    print("performance over time as the client learns about the behavior of hosts.")


async def main():
    """Run all demonstration functions."""
    await demonstrate_persistence()


if __name__ == "__main__":
    asyncio.run(main())
