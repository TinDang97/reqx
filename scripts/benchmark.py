#!/usr/bin/env python
"""
Benchmark script to compare performance of enhanced_httpx with standard httpx
and other HTTP client libraries.
"""

import asyncio
import time
import httpx
import aiohttp
import argparse
from tabulate import tabulate
import statistics
import sys
import os
import uvloop
from concurrent.futures import ProcessPoolExecutor

# Add parent directory to path for importing enhanced_httpx
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.enhanced_httpx import EnhancedClient

# Enable uvloop for enhanced performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Default settings
DEFAULT_URL = "https://httpbin.org/get"
DEFAULT_CONCURRENT_REQUESTS = 100
DEFAULT_RUNS = 3


class BenchmarkResult:
    def __init__(self, name, requests, duration, req_per_sec, avg_req_time):
        self.name = name
        self.requests = requests
        self.duration = duration
        self.req_per_sec = req_per_sec
        self.avg_req_time = avg_req_time


async def benchmark_enhanced_httpx(url, total_requests, headers=None, params=None):
    """Benchmark the enhanced_httpx client."""
    async with EnhancedClient() as client:
        start = time.time()
        tasks = []
        for _ in range(total_requests):
            tasks.append(client.get(url, headers=headers, params=params))
        await asyncio.gather(*tasks)
        end = time.time()
        duration = end - start

    return BenchmarkResult(
        name="enhanced_httpx",
        requests=total_requests,
        duration=duration,
        req_per_sec=total_requests / duration,
        avg_req_time=duration / total_requests,
    )


async def benchmark_httpx(url, total_requests, headers=None, params=None):
    """Benchmark the standard httpx client."""
    async with httpx.AsyncClient() as client:
        start = time.time()
        tasks = []
        for _ in range(total_requests):
            tasks.append(client.get(url, headers=headers, params=params))
        await asyncio.gather(*tasks)
        end = time.time()
        duration = end - start

    return BenchmarkResult(
        name="httpx",
        requests=total_requests,
        duration=duration,
        req_per_sec=total_requests / duration,
        avg_req_time=duration / total_requests,
    )


async def benchmark_aiohttp(url, total_requests, headers=None, params=None):
    """Benchmark the aiohttp client."""
    async with aiohttp.ClientSession() as session:
        start = time.time()
        tasks = []
        for _ in range(total_requests):
            tasks.append(session.get(url, headers=headers, params=params))
        responses = await asyncio.gather(*tasks)
        # Ensure bodies are read to match behavior of other clients
        for resp in responses:
            await resp.text()
        end = time.time()
        duration = end - start

    return BenchmarkResult(
        name="aiohttp",
        requests=total_requests,
        duration=duration,
        req_per_sec=total_requests / duration,
        avg_req_time=duration / total_requests,
    )


async def run_single_benchmark(client_type, url, requests, headers, params):
    """Run a single benchmark for a specific client."""
    if client_type == "enhanced_httpx":
        return await benchmark_enhanced_httpx(url, requests, headers, params)
    elif client_type == "httpx":
        return await benchmark_httpx(url, requests, headers, params)
    elif client_type == "aiohttp":
        return await benchmark_aiohttp(url, requests, headers, params)
    else:
        raise ValueError(f"Unknown client type: {client_type}")


async def run_benchmarks(url, concurrent_requests, runs, clients):
    """Run all benchmarks."""
    results = []

    print(f"Running benchmarks against {url}")
    print(f"Concurrent requests: {concurrent_requests}")
    print(f"Runs per client: {runs}")
    print(f"Clients: {', '.join(clients)}")
    print()

    for client in clients:
        client_results = []
        print(f"Benchmarking {client}...")

        for i in range(1, runs + 1):
            print(f"  Run {i}/{runs}...")
            result = await run_single_benchmark(client, url, concurrent_requests, None, None)
            client_results.append(result)

        # Calculate average metrics
        avg_duration = statistics.mean(r.duration for r in client_results)
        avg_req_per_sec = statistics.mean(r.req_per_sec for r in client_results)
        avg_req_time = statistics.mean(r.avg_req_time for r in client_results)

        results.append(
            BenchmarkResult(
                name=client,
                requests=concurrent_requests,
                duration=avg_duration,
                req_per_sec=avg_req_per_sec,
                avg_req_time=avg_req_time,
            )
        )

    # Sort results by requests per second (descending)
    results.sort(key=lambda r: r.req_per_sec, reverse=True)
    return results


def print_results(results):
    """Print benchmark results in a table."""
    table_data = []
    headers = ["Client", "Requests", "Duration (s)", "Req/sec", "Avg Request (ms)"]

    for result in results:
        table_data.append(
            [
                result.name,
                result.requests,
                f"{result.duration:.3f}",
                f"{result.req_per_sec:.2f}",
                f"{result.avg_req_time * 1000:.2f}",
            ]
        )

    print()
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Calculate relative performance
    fastest = results[0]
    print("\nRelative Performance:")
    for result in results[1:]:
        diff = (fastest.req_per_sec - result.req_per_sec) / fastest.req_per_sec * 100
        print(f"  {fastest.name} is {diff:.1f}% faster than {result.name}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark HTTP client performance")
    parser.add_argument(
        "--url", default=DEFAULT_URL, help=f"URL to benchmark against (default: {DEFAULT_URL})"
    )
    parser.add_argument(
        "-n",
        "--requests",
        type=int,
        default=DEFAULT_CONCURRENT_REQUESTS,
        help=f"Number of concurrent requests (default: {DEFAULT_CONCURRENT_REQUESTS})",
    )
    parser.add_argument(
        "-r",
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Number of benchmark runs per client (default: {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--clients",
        nargs="+",
        choices=["enhanced_httpx", "httpx", "aiohttp"],
        default=["enhanced_httpx", "httpx", "aiohttp"],
        help="HTTP clients to benchmark",
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    try:
        results = await run_benchmarks(args.url, args.requests, args.runs, args.clients)
        print_results(results)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")
        return 1
    except Exception as e:
        print(f"\nError: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
