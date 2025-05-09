#!/usr/bin/env python
"""
Example demonstrating the hybrid transport functionality of Enhanced-HTTPX.

This example shows how the client can intelligently switch between
httpx and aiohttp transports based on the protocol and performance
characteristics of each endpoint.
"""

import asyncio
import time
from typing import Dict, List

from src.builder import ReqxClientBuilder


async def measure_performance(client, url: str, requests: int = 10) -> Dict:
    """Measure performance of requests to a URL."""
    start_time = time.time()
    results = []

    for _ in range(requests):
        try:
            start = time.time()
            response = await client.get(url)
            duration = time.time() - start
            results.append({"status": response.status_code, "duration": duration})
        except Exception as e:
            results.append({"status": "error", "error": str(e)})

    total_time = time.time() - start_time
    successful = [r for r in results if isinstance(r["status"], int) and r["status"] < 400]

    return {
        "url": url,
        "total_time": total_time,
        "avg_time": (
            sum(r["duration"] for r in successful) / len(successful) if successful else None
        ),
        "success_rate": len(successful) / requests if requests else 0,
        "results": results,
    }


async def compare_transports(urls: List[str]):
    """Compare performance between different transport configurations."""
    print("=== Enhanced-HTTPX Transport Comparison ===\n")

    # Create clients with different transport configurations
    httpx_client = ReqxClientBuilder().use_aiohttp(False).with_http2(True).build()

    aiohttp_client = ReqxClientBuilder().use_aiohttp(True).with_http2(False).build()

    hybrid_client = (
        ReqxClientBuilder()
        .use_aiohttp(True)
        .with_http2(True)
        .with_transport_learning(True)  # Enable intelligent switching
        .build()
    )

    clients = {"HTTPX": httpx_client, "aiohttp": aiohttp_client, "Hybrid": hybrid_client}

    try:
        for name, client in clients.items():
            print(f"\n== Testing {name} Client ==")

            for url in urls:
                print(f"\nTesting URL: {url}")
                results = await measure_performance(client, url)

                print(f"  Success Rate: {results['success_rate'] * 100:.1f}%")
                if results["avg_time"]:
                    print(f"  Average Time: {results["avg_time"] * 1000:.2f}ms")
                print(f"  Total Time: {results['total_time']:.2f}s")

                # For hybrid client, show the transport used
                if name == "Hybrid":
                    metrics = client.get_transport_metrics()
                    host = url.split("/")[2]  # Extract hostname
                    if host in metrics.get("host_metrics", {}):
                        host_metrics = metrics["host_metrics"][host]
                        preferred = host_metrics.get("preferred", "unknown")
                        print(f"  Preferred Transport: {preferred}")

                        # Show counts per transport
                        for transport, data in host_metrics.items():
                            if transport != "preferred":
                                print(
                                    f"    - {transport}: {data.get('count', 0)} requests, "
                                    f"{data.get('avg_duration', 0) * 1000:.2f}ms avg"
                                )

    finally:
        # Close clients
        for client in clients.values():
            await client.close()


async def advanced_hybrid_example():
    """More complex example showing hybrid client capabilities."""
    print("\n=== Advanced Hybrid Client Example ===\n")

    # Create an optimized hybrid client
    client = (
        ReqxClientBuilder()
        .use_aiohttp(True)  # Enable aiohttp for HTTP/1.1
        .with_http2(True)  # Enable HTTP/2 with httpx
        .with_adaptive_timeout(True)  # Enable adaptive timeouts
        .auto_optimize()  # Auto-optimize based on system resources
        .build()
    )

    try:
        # Make requests to different endpoints
        urls = [
            "https://httpbin.org/get",  # HTTP/1.1
            "https://www.google.com/",  # HTTP/2 capable
            "https://api.github.com/zen",  # HTTP/2 capable API
        ]

        print("Making initial requests to train the hybrid client...")
        for url in urls:
            # Make a few requests to let the client learn
            for _ in range(5):
                await client.get(url)

        print("\nTransport metrics after training:")
        metrics = client.get_transport_metrics()

        # Print high-level metrics
        print(f"Total requests: {metrics.get('total_requests', 0)}")
        print("Transport usage:")
        for transport, data in metrics.get("transports", {}).items():
            print(f"  - {transport}: {data.get('requests', 0)} requests")

        print("\nHost-specific metrics:")
        for host, host_data in metrics.get("host_metrics", {}).items():
            preferred = host_data.get("preferred", "unknown")
            print(f"  {host}: Preferred transport = {preferred}")

            # Show detailed metrics for this host
            for transport, data in host_data.items():
                if transport != "preferred" and isinstance(data, dict):
                    print(
                        f"    - {transport}: {data.get('count', 0)} requests, "
                        f"{data.get('avg_duration', 0) * 1000:.2f}ms avg"
                    )

        # Make parallel requests to demonstrate throughput improvement
        print("\nMaking parallel requests to demonstrate throughput...")
        start_time = time.time()
        tasks = []
        for url in urls * 5:  # 5 requests to each URL
            tasks.append(client.get(url))

        responses = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time

        successful = [r for r in responses if hasattr(r, "status_code")]
        print(f"Made {len(tasks)} parallel requests in {total_time:.2f}s")
        print(f"Success rate: {len(successful) / len(tasks) * 100:.1f}%")

    finally:
        # Close client
        await client.close()


async def main():
    """Run the main example."""
    # Test URLs - mix of HTTP/1.1 and HTTP/2 capable endpoints
    urls = [
        "https://httpbin.org/get",  # HTTP/1.1
        "https://www.google.com/",  # HTTP/2 capable
        "https://api.github.com/zen",  # HTTP/2 capable API
    ]

    # Compare different transport configurations
    await compare_transports(urls)

    # Show advanced hybrid client capabilities
    await advanced_hybrid_example()


if __name__ == "__main__":
    asyncio.run(main())
