#!/usr/bin/env python
"""
Benchmark script to measure the performance benefits of persistence.

This script compares the performance of clients with and without persistence
to demonstrate the benefits of persisting optimized settings across runs.
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import tempfile
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.builder import ReqxClientBuilder


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    iteration: int
    requests: int
    duration: float
    req_per_sec: float
    avg_req_time: float
    min_req_time: float
    max_req_time: float
    memory_usage: int  # KB
    timeouts: int = 0
    errors: int = 0


@dataclass
class BenchmarkSummary:
    """Summary statistics for multiple benchmark runs."""

    name: str
    avg_req_per_sec: float
    avg_duration: float
    avg_memory: int
    percentile_95: float
    percentile_99: float
    timeline: List[float]  # Request per second over time
    memory_timeline: List[int]  # Memory usage over time


async def warmup_requests(url: str, count: int = 3):
    """Perform a few warmup requests to initialize connections."""
    print("Performing warmup requests...")
    client = ReqxClientBuilder().with_timeout(5.0).build()
    for _ in range(count):
        try:
            await client.get(url)
        except Exception:
            pass
    await client.close()
    print("Warmup complete")


async def run_benchmark_with_persistence(
    url: str,
    requests_per_iteration: int,
    iterations: int,
    persistence_path: Optional[str] = None,
    concurrency: int = 10,
    params: Optional[Dict] = None,
):
    """
    Run benchmark with persistence enabled.

    This simulates multiple sessions using the same persistent storage,
    allowing the client to learn and optimize over time.

    Args:
        url: URL to benchmark against
        requests_per_iteration: Number of requests per iteration
        iterations: Number of benchmark iterations
        persistence_path: Path to store persistence files, or None for temp dir
        concurrency: Maximum concurrent requests
        params: Optional query parameters for requests

    Returns:
        List of BenchmarkResult objects
    """
    # Create a temporary directory if no persistence path provided
    if persistence_path is None:
        temp_dir = tempfile.TemporaryDirectory()
        persistence_path = temp_dir.name

    results = []

    # Run multiple iterations, simulating multiple application sessions
    # but using the same persistent storage
    for i in range(iterations):
        print(f"Running iteration {i+1}/{iterations} with persistence...")

        # Create a client with persistence enabled
        client = (
            ReqxClientBuilder()
            .for_high_performance()
            .with_adaptive_timeout(True)
            .with_persistence(enabled=True, path=persistence_path)
            .build()
        )

        # Start memory tracking
        tracemalloc.start()

        # Record request times for calculation
        request_times = []
        timeouts = 0
        errors = 0

        # Start timing
        start = time.time()

        # Create batch of requests
        urls = [url for _ in range(requests_per_iteration)]

        # Execute requests with metrics tracking
        try:
            tasks = []
            semaphore = asyncio.Semaphore(concurrency)

            async def execute_request(url):
                nonlocal timeouts, errors
                req_start = time.time()
                try:
                    async with semaphore:
                        response = await client.get(url, params=params, timeout=5.0)
                    req_time = time.time() - req_start
                    request_times.append(req_time)
                except asyncio.TimeoutError:
                    timeouts += 1
                except Exception:
                    errors += 1

            for url in urls:
                tasks.append(asyncio.create_task(execute_request(url)))

            await asyncio.gather(*tasks)

        finally:
            await client.close()

        # Calculate benchmark metrics
        duration = time.time() - start
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Store the result
        result = BenchmarkResult(
            name="with_persistence",
            iteration=i + 1,
            requests=requests_per_iteration,
            duration=duration,
            req_per_sec=requests_per_iteration / duration if duration > 0 else 0,
            avg_req_time=statistics.mean(request_times) if request_times else 0,
            min_req_time=min(request_times) if request_times else 0,
            max_req_time=max(request_times) if request_times else 0,
            memory_usage=peak // 1024,  # Convert to KB
            timeouts=timeouts,
            errors=errors,
        )
        results.append(result)

    # If we created a temp dir, clean it up
    if not persistence_path and "temp_dir" in locals():
        temp_dir.cleanup()

    return results


async def run_benchmark_without_persistence(
    url: str,
    requests_per_iteration: int,
    iterations: int,
    concurrency: int = 10,
    params: Optional[Dict] = None,
):
    """
    Run benchmark without persistence.

    This simulates multiple sessions without any persistence,
    so the client has to relearn optimal settings each time.

    Args:
        url: URL to benchmark against
        requests_per_iteration: Number of requests per iteration
        iterations: Number of benchmark iterations
        concurrency: Maximum concurrent requests
        params: Optional query parameters for requests

    Returns:
        List of BenchmarkResult objects
    """
    results = []

    # Run multiple iterations, each time with a fresh client with no persistence
    for i in range(iterations):
        print(f"Running iteration {i+1}/{iterations} without persistence...")

        # Create a client with adaptive timeout but no persistence
        client = (
            ReqxClientBuilder()
            .for_high_performance()
            .with_adaptive_timeout(True)
            .with_persistence(enabled=False)
            .build()
        )

        # Start memory tracking
        tracemalloc.start()

        # Record request times
        request_times = []
        timeouts = 0
        errors = 0

        # Start timing
        start = time.time()

        # Create batch of requests
        urls = [url for _ in range(requests_per_iteration)]

        # Execute requests with metrics tracking
        try:
            tasks = []
            semaphore = asyncio.Semaphore(concurrency)

            async def execute_request(url):
                nonlocal timeouts, errors
                req_start = time.time()
                try:
                    async with semaphore:
                        response = await client.get(url, params=params, timeout=5.0)
                    req_time = time.time() - req_start
                    request_times.append(req_time)
                except asyncio.TimeoutError:
                    timeouts += 1
                except Exception:
                    errors += 1

            for url in urls:
                tasks.append(asyncio.create_task(execute_request(url)))

            await asyncio.gather(*tasks)

        finally:
            await client.close()

        # Calculate benchmark metrics
        duration = time.time() - start
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Store the result
        result = BenchmarkResult(
            name="without_persistence",
            iteration=i + 1,
            requests=requests_per_iteration,
            duration=duration,
            req_per_sec=requests_per_iteration / duration if duration > 0 else 0,
            avg_req_time=statistics.mean(request_times) if request_times else 0,
            min_req_time=min(request_times) if request_times else 0,
            max_req_time=max(request_times) if request_times else 0,
            memory_usage=peak // 1024,  # Convert to KB
            timeouts=timeouts,
            errors=errors,
        )
        results.append(result)

    return results


def summarize_results(results: List[BenchmarkResult]) -> BenchmarkSummary:
    """
    Calculate summary statistics for benchmark results.

    Args:
        results: List of benchmark results

    Returns:
        Summary statistics
    """
    if not results:
        return BenchmarkSummary(
            name="empty",
            avg_req_per_sec=0,
            avg_duration=0,
            avg_memory=0,
            percentile_95=0,
            percentile_99=0,
            timeline=[],
            memory_timeline=[],
        )

    # Extract the name from the first result
    name = results[0].name

    # Calculate statistics
    req_per_sec = [r.req_per_sec for r in results]
    durations = [r.duration for r in results]
    memory = [r.memory_usage for r in results]
    req_times = []

    # Flatten request times to calculate percentiles
    for r in results:
        avg_times = r.avg_req_time
        req_times.extend([avg_times] * r.requests)

    # Sort for percentile calculation
    req_times.sort()

    # Calculate percentiles if we have enough data
    p95 = p99 = 0
    if req_times:
        p95_idx = int(len(req_times) * 0.95)
        p99_idx = int(len(req_times) * 0.99)
        p95 = req_times[min(p95_idx, len(req_times) - 1)]
        p99 = req_times[min(p99_idx, len(req_times) - 1)]

    return BenchmarkSummary(
        name=name,
        avg_req_per_sec=statistics.mean(req_per_sec),
        avg_duration=statistics.mean(durations),
        avg_memory=int(statistics.mean(memory)),
        percentile_95=p95,
        percentile_99=p99,
        timeline=req_per_sec,
        memory_timeline=memory,
    )


def print_results(
    with_persistence_results: List[BenchmarkResult],
    without_persistence_results: List[BenchmarkResult],
):
    """
    Print benchmark results in a human-readable format.

    Args:
        with_persistence_results: Results with persistence enabled
        without_persistence_results: Results without persistence
    """
    # Create summary statistics
    with_summary = summarize_results(with_persistence_results)
    without_summary = summarize_results(without_persistence_results)

    # Print results
    print("\n=== BENCHMARK RESULTS ===\n")

    # Print summary comparison
    print("Performance Comparison:")
    print(
        f"{'Metric':<25} {'With Persistence':<18} {'Without Persistence':<20} {'Improvement':<12}"
    )
    print("-" * 80)

    # Calculate improvement percentage
    if without_summary.avg_req_per_sec > 0:
        req_imp = (
            (with_summary.avg_req_per_sec - without_summary.avg_req_per_sec)
            / without_summary.avg_req_per_sec
            * 100
        )
    else:
        req_imp = 0

    if without_summary.avg_duration > 0:
        dur_imp = (
            (without_summary.avg_duration - with_summary.avg_duration)
            / without_summary.avg_duration
            * 100
        )
    else:
        dur_imp = 0

    print(
        f"{'Requests/sec':<25} {with_summary.avg_req_per_sec:<18.2f} {without_summary.avg_req_per_sec:<20.2f} {req_imp:>+.2f}%"
    )
    print(
        f"{'Avg. Duration (s)':<25} {with_summary.avg_duration:<18.2f} {without_summary.avg_duration:<20.2f} {dur_imp:>+.2f}%"
    )

    # Compare 95th and 99th percentile response times
    if without_summary.percentile_95 > 0:
        p95_imp = (
            (without_summary.percentile_95 - with_summary.percentile_95)
            / without_summary.percentile_95
            * 100
        )
    else:
        p95_imp = 0

    if without_summary.percentile_99 > 0:
        p99_imp = (
            (without_summary.percentile_99 - with_summary.percentile_99)
            / without_summary.percentile_99
            * 100
        )
    else:
        p99_imp = 0

    print(
        f"{'95th Percentile (s)':<25} {with_summary.percentile_95:<18.4f} {without_summary.percentile_95:<20.4f} {p95_imp:>+.2f}%"
    )
    print(
        f"{'99th Percentile (s)':<25} {with_summary.percentile_99:<18.4f} {without_summary.percentile_99:<20.4f} {p99_imp:>+.2f}%"
    )

    # Memory usage
    if without_summary.avg_memory > 0:
        mem_imp = (
            (without_summary.avg_memory - with_summary.avg_memory)
            / without_summary.avg_memory
            * 100
        )
    else:
        mem_imp = 0

    print(
        f"{'Memory Usage (KB)':<25} {with_summary.avg_memory:<18d} {without_summary.avg_memory:<20d} {mem_imp:>+.2f}%"
    )

    # Print iteration details
    print("\nDetailed Results by Iteration:")
    print("\nWith Persistence:")
    for i, result in enumerate(with_persistence_results):
        print(
            f"  Iteration {i+1}: {result.req_per_sec:.2f} req/s, {result.avg_req_time*1000:.2f}ms avg, {result.timeouts} timeouts"
        )

    print("\nWithout Persistence:")
    for i, result in enumerate(without_persistence_results):
        print(
            f"  Iteration {i+1}: {result.req_per_sec:.2f} req/s, {result.avg_req_time*1000:.2f}ms avg, {result.timeouts} timeouts"
        )


def plot_results(
    with_persistence_results: List[BenchmarkResult],
    without_persistence_results: List[BenchmarkResult],
    output_file: Optional[str] = None,
):
    """
    Plot benchmark results.

    Args:
        with_persistence_results: Results with persistence enabled
        without_persistence_results: Results without persistence
        output_file: Optional file to save the plot
    """
    # Create summary objects
    with_summary = summarize_results(with_persistence_results)
    without_summary = summarize_results(without_persistence_results)

    # Set up the figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Persistence Benchmark Results", fontsize=16)

    # Plot 1: Requests per second over iterations
    iterations = list(range(1, len(with_persistence_results) + 1))
    axs[0, 0].plot(
        iterations,
        [r.req_per_sec for r in with_persistence_results],
        "b-o",
        label="With Persistence",
    )
    axs[0, 0].plot(
        iterations,
        [r.req_per_sec for r in without_persistence_results],
        "r-o",
        label="Without Persistence",
    )
    axs[0, 0].set_title("Performance Over Iterations")
    axs[0, 0].set_xlabel("Iteration")
    axs[0, 0].set_ylabel("Requests/second")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot 2: Average request times over iterations
    axs[0, 1].plot(
        iterations,
        [r.avg_req_time * 1000 for r in with_persistence_results],
        "b-o",
        label="With Persistence",
    )
    axs[0, 1].plot(
        iterations,
        [r.avg_req_time * 1000 for r in without_persistence_results],
        "r-o",
        label="Without Persistence",
    )
    axs[0, 1].set_title("Response Time Over Iterations")
    axs[0, 1].set_xlabel("Iteration")
    axs[0, 1].set_ylabel("Average Response Time (ms)")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot 3: Memory usage
    axs[1, 0].plot(
        iterations,
        [r.memory_usage for r in with_persistence_results],
        "b-o",
        label="With Persistence",
    )
    axs[1, 0].plot(
        iterations,
        [r.memory_usage for r in without_persistence_results],
        "r-o",
        label="Without Persistence",
    )
    axs[1, 0].set_title("Memory Usage Over Iterations")
    axs[1, 0].set_xlabel("Iteration")
    axs[1, 0].set_ylabel("Memory Usage (KB)")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot 4: Bar chart comparing average metrics
    labels = ["Requests/sec", "Avg. Duration (s)", "Memory (MB)"]
    with_values = [
        with_summary.avg_req_per_sec,
        with_summary.avg_duration,
        with_summary.avg_memory / 1024,
    ]
    without_values = [
        without_summary.avg_req_per_sec,
        without_summary.avg_duration,
        without_summary.avg_memory / 1024,
    ]

    x = np.arange(len(labels))
    width = 0.35

    axs[1, 1].bar(x - width / 2, with_values, width, label="With Persistence")
    axs[1, 1].bar(x + width / 2, without_values, width, label="Without Persistence")
    axs[1, 1].set_title("Average Performance Comparison")
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(labels)
    axs[1, 1].legend()

    # Adjust layout and save if requested
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def export_results(
    with_persistence_results: List[BenchmarkResult],
    without_persistence_results: List[BenchmarkResult],
    output_file: str,
):
    """
    Export benchmark results to a JSON file.

    Args:
        with_persistence_results: Results with persistence enabled
        without_persistence_results: Results without persistence
        output_file: File to write results to
    """
    # Convert to dictionaries
    with_dicts = [asdict(r) for r in with_persistence_results]
    without_dicts = [asdict(r) for r in without_persistence_results]

    # Create result object
    result = {
        "timestamp": time.time(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "with_persistence": with_dicts,
        "without_persistence": without_dicts,
        "with_persistence_summary": asdict(summarize_results(with_persistence_results)),
        "without_persistence_summary": asdict(summarize_results(without_persistence_results)),
    }

    # Save to file
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Results exported to {output_file}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark persistence performance")

    parser.add_argument(
        "--url", type=str, default="https://httpbin.org/get", help="URL to benchmark against"
    )
    parser.add_argument("--requests", type=int, default=50, help="Number of requests per iteration")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations to run")
    parser.add_argument("--concurrency", type=int, default=10, help="Maximum concurrent requests")
    parser.add_argument("--persistence-path", type=str, help="Directory to store persistence data")
    parser.add_argument("--output-json", type=str, help="File to export results as JSON")
    parser.add_argument("--plot", action="store_true", help="Generate plots of results")
    parser.add_argument(
        "--plot-file", type=str, help="File to save plot to (if --plot is specified)"
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    # Ensure Python 3.7+
    if sys.version_info < (3, 7):
        print("This script requires Python 3.7 or higher")
        return 1

    print(f"Benchmarking {args.url}")
    print(f"Requests per iteration: {args.requests}")
    print(f"Iterations: {args.iterations}")
    print(f"Maximum concurrency: {args.concurrency}")
    print(f"Persistence path: {args.persistence_path or 'temporary directory'}")

    # Perform warmup requests to initialize connections
    await warmup_requests(args.url)

    # Run benchmarks
    with_persistence_results = await run_benchmark_with_persistence(
        args.url, args.requests, args.iterations, args.persistence_path, args.concurrency
    )

    without_persistence_results = await run_benchmark_without_persistence(
        args.url, args.requests, args.iterations, args.concurrency
    )

    # Print results
    print_results(with_persistence_results, without_persistence_results)

    # Export results if requested
    if args.output_json:
        export_results(with_persistence_results, without_persistence_results, args.output_json)

    # Generate plot if requested
    if args.plot:
        try:
            plot_results(with_persistence_results, without_persistence_results, args.plot_file)
        except Exception as e:
            print(f"Error generating plot: {str(e)}")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
