#!/usr/bin/env python
"""
Benchmark script to compare performance of reqx with standard httpx
and other HTTP client libraries.
"""

import argparse
import asyncio
import csv
import gc
import json
import os
import statistics
import sys
import time
import traceback
import tracemalloc
from datetime import datetime

import aiohttp
import httpx
import uvloop
from tabulate import tabulate

# Add parent directory to path for importing reqx
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import ReqxClient

# Enable uvloop for enhanced performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Default settings
DEFAULT_URL = "http://0.0.0.0:80"
DEFAULT_CONCURRENT_REQUESTS = 100
DEFAULT_RUNS = 3


class BenchmarkResult:
    def __init__(self, name, requests, duration, req_per_sec, avg_req_time, memory_usage=None):
        self.name = name
        self.requests = requests
        self.duration = duration
        self.req_per_sec = req_per_sec
        self.avg_req_time = avg_req_time
        self.memory_usage = memory_usage
        self.timestamp = datetime.now().isoformat()

    def to_dict(self):
        """Convert the result to a dictionary for export."""
        return {
            "client": self.name,
            "requests": self.requests,
            "duration_seconds": round(self.duration, 3),
            "requests_per_second": round(self.req_per_sec, 2),
            "avg_request_time_ms": round(self.avg_req_time * 1000, 2),
            "memory_usage_kb": self.memory_usage,
            "timestamp": self.timestamp,
        }


async def benchmark_reqx(
    url, total_requests, method="get", headers=None, params=None, json_data=None
):
    """Benchmark the reqx client."""
    # Start memory tracking
    gc.collect()
    tracemalloc.start()

    async with ReqxClient() as client:
        start = time.time()
        tasks = []
        for _ in range(total_requests):
            if method == "get":
                tasks.append(client.get(url, headers=headers, params=params))
            elif method == "post":
                tasks.append(client.post(url, headers=headers, params=params, json=json_data))
            elif method == "put":
                tasks.append(client.put(url, headers=headers, params=params, json=json_data))
            elif method == "delete":
                tasks.append(client.delete(url, headers=headers, params=params))

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            print(f"Error during reqx benchmark: {str(e)}")
            raise e

        end = time.time()
        duration = end - start

    # Get memory usage and stop tracking
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return BenchmarkResult(
        name="reqx",
        requests=total_requests,
        duration=duration,
        req_per_sec=total_requests / duration,
        avg_req_time=duration / total_requests,
        memory_usage=peak // 1024,  # Convert to KB
    )


async def benchmark_httpx(
    url, total_requests, method="get", headers=None, params=None, json_data=None
):
    """Benchmark the standard httpx client."""
    # Start memory tracking
    gc.collect()
    tracemalloc.start()

    async with httpx.AsyncClient() as client:
        start = time.time()
        tasks = []
        for _ in range(total_requests):
            if method == "get":
                tasks.append(client.get(url, headers=headers, params=params))
            elif method == "post":
                tasks.append(client.post(url, headers=headers, params=params, json=json_data))
            elif method == "put":
                tasks.append(client.put(url, headers=headers, params=params, json=json_data))
            elif method == "delete":
                tasks.append(client.delete(url, headers=headers, params=params))

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            print(f"Error during httpx benchmark: {str(e)}")

        end = time.time()
        duration = end - start

    # Get memory usage and stop tracking
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return BenchmarkResult(
        name="httpx",
        requests=total_requests,
        duration=duration,
        req_per_sec=total_requests / duration,
        avg_req_time=duration / total_requests,
        memory_usage=peak // 1024,  # Convert to KB
    )


async def benchmark_aiohttp(
    url, total_requests, method="get", headers=None, params=None, json_data=None
):
    """Benchmark the aiohttp client."""
    # Start memory tracking
    gc.collect()
    tracemalloc.start()

    async with aiohttp.ClientSession() as session:
        start = time.time()
        tasks = []
        for _ in range(total_requests):
            if method == "get":
                tasks.append(session.get(url, headers=headers, params=params))
            elif method == "post":
                tasks.append(session.post(url, headers=headers, params=params, json=json_data))
            elif method == "put":
                tasks.append(session.put(url, headers=headers, params=params, json=json_data))
            elif method == "delete":
                tasks.append(session.delete(url, headers=headers, params=params))

        try:
            responses = await asyncio.gather(*tasks)
            # Ensure bodies are read to match behavior of other clients
            for resp in responses:
                await resp.text()
        except Exception as e:
            print(f"Error during aiohttp benchmark: {str(e)}")

        end = time.time()
        duration = end - start

    # Get memory usage and stop tracking
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return BenchmarkResult(
        name="aiohttp",
        requests=total_requests,
        duration=duration,
        req_per_sec=total_requests / duration,
        avg_req_time=duration / total_requests,
        memory_usage=peak // 1024,  # Convert to KB
    )


async def run_single_benchmark(client_type, url, requests, method, headers, params, json_data):
    """Run a single benchmark for a specific client."""
    if client_type == "reqx":
        return await benchmark_reqx(url, requests, method, headers, params, json_data)
    elif client_type == "httpx":
        return await benchmark_httpx(url, requests, method, headers, params, json_data)
    elif client_type == "aiohttp":
        return await benchmark_aiohttp(url, requests, method, headers, params, json_data)
    else:
        raise ValueError(f"Unknown client type: {client_type}")


async def run_benchmarks(
    url, concurrent_requests, runs, clients, method, headers, params, json_data
):
    """Run all benchmarks."""
    results = []

    print(f"Running benchmarks against {url}")
    print(f"HTTP method: {method.upper()}")
    print(f"Concurrent requests: {concurrent_requests}")
    print(f"Runs per client: {runs}")
    print(f"Clients: {', '.join(clients)}")
    print()

    for client in clients:
        client_results = []
        print(f"Benchmarking {client}...")

        for i in range(1, runs + 1):
            print(f"  Run {i}/{runs}...")
            result = await run_single_benchmark(
                client, url, concurrent_requests, method, headers, params, json_data
            )
            client_results.append(result)

        # Calculate average metrics
        avg_duration = statistics.mean(r.duration for r in client_results)
        avg_req_per_sec = statistics.mean(r.req_per_sec for r in client_results)
        avg_req_time = statistics.mean(r.avg_req_time for r in client_results)
        avg_memory = (
            statistics.mean(r.memory_usage for r in client_results)
            if all(r.memory_usage is not None for r in client_results)
            else None
        )

        results.append(
            BenchmarkResult(
                name=client,
                requests=concurrent_requests,
                duration=avg_duration,
                req_per_sec=avg_req_per_sec,
                avg_req_time=avg_req_time,
                memory_usage=avg_memory,
            )
        )

    # Sort results by requests per second (descending)
    results.sort(key=lambda r: r.req_per_sec, reverse=True)
    return results


def print_results(results):
    """Print benchmark results in a table."""
    table_data = []
    headers = ["Client", "Requests", "Duration (s)", "Req/sec", "Avg Request (ms)", "Memory (KB)"]

    for result in results:
        memory_str = f"{result.memory_usage:.2f}" if result.memory_usage is not None else "N/A"
        table_data.append(
            [
                result.name,
                result.requests,
                f"{result.duration:.3f}",
                f"{result.req_per_sec:.2f}",
                f"{result.avg_req_time * 1000:.2f}",
                memory_str,
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

    # Memory comparison if available
    if all(r.memory_usage is not None for r in results):
        most_efficient = min(results, key=lambda r: r.memory_usage)
        print("\nMemory Efficiency:")
        for result in [r for r in results if r.name != most_efficient.name]:
            diff = (
                (result.memory_usage - most_efficient.memory_usage)
                / most_efficient.memory_usage
                * 100
            )
            print(f"  {most_efficient.name} uses {diff:.1f}% less memory than {result.name}")


def export_results(results, output_format, filename=None):
    """Export benchmark results to a file."""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.{output_format}"

    result_dicts = [r.to_dict() for r in results]

    if output_format == "json":
        with open(filename, "w") as f:
            json.dump(result_dicts, f, indent=2)
    elif output_format == "csv":
        with open(filename, "w", newline="") as f:
            if result_dicts:
                writer = csv.DictWriter(f, fieldnames=result_dicts[0].keys())
                writer.writeheader()
                writer.writerows(result_dicts)

    print(f"\nResults exported to {filename}")


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
        choices=["reqx", "httpx", "aiohttp"],
        default=["reqx", "httpx", "aiohttp"],
        help="HTTP clients to benchmark",
    )
    parser.add_argument(
        "--method",
        choices=["get", "post", "put", "delete"],
        default="get",
        help="HTTP method to benchmark (default: get)",
    )
    parser.add_argument(
        "--json-data",
        default="{}",
        help="JSON data to send with POST/PUT requests (as a JSON string)",
    )
    parser.add_argument(
        "--export",
        choices=["json", "csv"],
        help="Export results to a file (specify format: json or csv)",
    )
    parser.add_argument(
        "--output-file", help="Output filename for exported results (default: auto-generated)"
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    try:
        # Parse JSON data if provided
        json_data = None
        if args.json_data and args.method in ["post", "put"]:
            try:
                json_data = json.loads(args.json_data)
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON data: {args.json_data}")
                return 1

        results = await run_benchmarks(
            args.url,
            args.requests,
            args.runs,
            args.clients,
            args.method,
            None,  # headers
            None,  # params
            json_data,
        )

        print_results(results)

        # Export results if requested
        if args.export:
            export_results(results, args.export, args.output_file)

    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")
        return 1
    except Exception as e:
        print(f"\nError: {str(e)}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
