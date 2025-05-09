#!/usr/bin/env python
"""
Comprehensive example demonstrating the enhanced features of Reqx.

This example showcases:
1. Adaptive protocol selection (aiohttp vs httpx)
2. Performance metrics and analysis
3. Adaptive timeout management
4. Auto-optimized connection pooling
5. Advanced builder pattern usage
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

from src.builder import ReqxClientBuilder
from src.client import ReqxClient


async def demo_adaptive_protocol_selection():
    """Demonstrate the automatic protocol selection for different hosts."""
    print("\n=== ADAPTIVE PROTOCOL SELECTION ===")

    # Create a hybrid client with debug enabled
    client = ReqxClientBuilder().with_debug(True).with_timeout(10.0).build()

    # List of URLs with different protocol support
    urls = [
        "https://www.google.com/",  # Likely HTTP/2
        "https://httpbin.org/get",  # HTTP/1.1
        "https://www.cloudflare.com/",  # HTTP/2 + possibly HTTP/3
        "https://example.com/",  # Basic HTTP/1.1
        "https://github.com/",  # HTTP/2
    ]

    print("Making requests to different hosts to demonstrate protocol selection...")
    for url in urls:
        start_time = time.time()
        response = await client.get(url)
        duration = time.time() - start_time

        # Extract protocol information if available
        protocol = response.http_version if hasattr(response, "http_version") else "HTTP/1.1"

        print(f"URL: {url}")
        print(f"  Status: {response.status_code}")
        print(f"  Protocol: {protocol}")
        print(f"  Duration: {duration:.4f}s")

    # If metrics are enabled, show the protocol preferences
    if hasattr(client.transport, "get_metrics_summary"):
        metrics = client.transport.get_metrics_summary()
        print("\nTransport Metrics Summary:")
        print(json.dumps(metrics, indent=2))

    await client.close()


async def demo_adaptive_timeout():
    """Demonstrate the adaptive timeout feature."""
    print("\n=== ADAPTIVE TIMEOUT MANAGEMENT ===")

    # Create a client with adaptive timeouts enabled
    client = (
        ReqxClientBuilder()
        .with_adaptive_timeout(True)  # Enable adaptive timeouts
        .with_timeout(5.0)  # Start with a 5 second timeout
        .build()
    )

    # Test URLs with varying response times
    fast_url = "https://httpbin.org/get"  # Usually fast
    slow_url = "https://httpbin.org/delay/2"  # 2 second delay

    print("Making requests to gather timeout statistics...")

    # Make multiple requests to gather statistics
    for i in range(10):
        print(f"Request batch {i+1}/10...")

        # Request to fast URL
        start = time.time()
        await client.get(fast_url)
        print(f"  Fast URL response time: {time.time() - start:.4f}s")

        # Request to slow URL
        start = time.time()
        await client.get(slow_url)
        print(f"  Slow URL response time: {time.time() - start:.4f}s")

    # Get the timeout statistics
    stats = client.get_timeout_statistics()

    print("\nAdaptive Timeout Statistics:")
    print(json.dumps(stats, indent=2, default=str))

    await client.close()


async def demo_optimized_connection_pooling():
    """Demonstrate the auto-optimized connection pooling."""
    print("\n=== AUTO-OPTIMIZED CONNECTION POOLING ===")

    # Create a client with auto optimization
    client = ReqxClientBuilder().auto_optimize(True).build()  # Enable auto-optimization

    # Get system information
    from src.utils import get_optimal_connection_pool_settings

    optimal_settings = get_optimal_connection_pool_settings()

    print("Optimized connection pool settings for this system:")
    print(json.dumps(optimal_settings, indent=2))

    # Create a batch of URLs to test connection pooling
    urls = ["https://httpbin.org/get?id=" + str(i) for i in range(20)]

    print("\nMaking 20 parallel requests to test connection pool...")
    start = time.time()
    responses = await client.parallel_get(urls, max_concurrency=optimal_settings["max_connections"])
    duration = time.time() - start

    print(f"Completed 20 requests in {duration:.4f}s")
    print(f"Average request time: {duration/20:.4f}s")

    await client.close()


async def demo_performance_profiles():
    """Demonstrate the preset performance profiles."""
    print("\n=== PERFORMANCE PROFILES ===")

    # Create a high-performance client
    high_perf_client = (
        ReqxClientBuilder().for_high_performance().build()  # Use high performance preset
    )

    # Create a high-reliability client
    high_reliability_client = (
        ReqxClientBuilder().for_reliability().build()  # Use high reliability preset
    )

    # Compare the configurations
    print("High Performance Profile Settings:")
    print(f"  HTTP/2 Enabled: {high_perf_client.transport.supports_http2()}")
    print(f"  Max Retries: {high_perf_client.max_retries}")
    print(f"  Cache Enabled: {high_perf_client.enable_cache}")

    print("\nHigh Reliability Profile Settings:")
    print(f"  HTTP/2 Enabled: {high_reliability_client.transport.supports_http2()}")
    print(f"  Max Retries: {high_reliability_client.max_retries}")
    print(f"  Timeout: {high_reliability_client.timeout}s")

    # Test both clients with a sample URL
    test_url = "https://httpbin.org/get"

    print("\nTesting high performance client...")
    start = time.time()
    await high_perf_client.get(test_url)
    high_perf_time = time.time() - start

    print("\nTesting high reliability client...")
    start = time.time()
    await high_reliability_client.get(test_url)
    high_rel_time = time.time() - start

    print(f"\nHigh Performance Time: {high_perf_time:.4f}s")
    print(f"High Reliability Time: {high_rel_time:.4f}s")

    await asyncio.gather(high_perf_client.close(), high_reliability_client.close())


async def main():
    """Run all demonstration functions."""
    print(f"=== REQX ENHANCED FEATURES DEMONSTRATION ===")
    print(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("This demo showcases the enhanced features of Reqx.")

    # Run all demos
    await demo_adaptive_protocol_selection()
    await demo_adaptive_timeout()
    await demo_optimized_connection_pooling()
    await demo_performance_profiles()

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
