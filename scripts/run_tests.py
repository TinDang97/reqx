#!/usr/bin/env python
"""
Script to run tests with coverage and optional features.
"""

import argparse
import os
import platform
import subprocess
import sys

# Configure paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
TESTS_DIR = os.path.join(ROOT_DIR, "tests")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run enhanced-httpx tests")
    parser.add_argument(
        "--coverage", "-c", action="store_true", help="Run tests with coverage reporting"
    )
    parser.add_argument("--html", action="store_true", help="Generate HTML coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Run tests in verbose mode")
    parser.add_argument("--test-path", "-p", help="Specific test file or directory to run")
    parser.add_argument("--junit", action="store_true", help="Generate JUnit XML reports")
    parser.add_argument(
        "--quick", action="store_true", help="Run tests without slow integration tests"
    )
    return parser.parse_args()


def run_tests(args):
    """Run the tests with the specified options."""
    # Base command
    cmd = ["pytest"]

    # Add verbosity
    if args.verbose:
        cmd.append("-v")

    # Add coverage if requested
    if args.coverage:
        cmd.extend(["--cov=enhanced_httpx", "--cov-report=term", "--cov-config=.coveragerc"])

        if args.html:
            cmd.append("--cov-report=html")

    # Add JUnit XML reporting if requested
    if args.junit:
        cmd.append("--junitxml=test-results.xml")

    # Add quick mode (skip slow tests)
    if args.quick:
        cmd.append("-k 'not slow'")

    # Add specific test path if provided
    if args.test_path:
        cmd.append(args.test_path)
    else:
        cmd.append(TESTS_DIR)

    # Print command
    print(f"Running: {' '.join(cmd)}")

    # Run the command
    process = subprocess.run(" ".join(cmd), shell=True, cwd=ROOT_DIR)
    return process.returncode


def ensure_test_requirements():
    """Ensure test dependencies are installed."""
    print("Checking test dependencies...")

    try:
        import httpx
        import pytest
        import pytest_cov
        import respx

        print("All required test dependencies are installed.")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Installing required test dependencies...")

        cmd = [sys.executable, "-m", "pip", "install", "-e", f"{ROOT_DIR}[dev]"]
        subprocess.run(cmd)


def main():
    """Main entry point."""
    # Ensure we're in the right environment
    ensure_test_requirements()

    # Parse arguments
    args = parse_args()

    # Run the tests
    return run_tests(args)


if __name__ == "__main__":
    sys.exit(main())
