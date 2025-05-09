"""
Command-line interface for enhanced-httpx.

This module provides a CLI for making HTTP requests directly from the terminal,
similar to tools like curl or httpie but with the enhanced features of the library.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import orjson

from .client import EnhancedClient
from .middleware import CompressionMiddleware, TracingMiddleware
from .utils import deserialize_json, serialize_json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("enhanced_httpx.cli")


def parse_key_value(s: str) -> tuple:
    """Parse a key-value string in the format key=value or key:value."""
    if "=" in s:
        k, v = s.split("=", 1)
    elif ":" in s:
        k, v = s.split(":", 1)
    else:
        k, v = s, ""
    return k.strip(), v.strip()


def parse_headers(header_strings: List[str]) -> Dict[str, str]:
    """Parse header strings in the format 'Name: Value'."""
    headers = {}
    for header in header_strings:
        name, value = parse_key_value(header)
        headers[name] = value
    return headers


def parse_data(data_strings: List[str]) -> Dict[str, str]:
    """Parse data strings in the format 'key=value'."""
    data = {}
    for item in data_strings:
        key, value = parse_key_value(item)
        data[key] = value
    return data


def parse_json_file(file_path: str) -> Dict[str, Any]:
    """Parse JSON from a file."""
    with open(os.path.expanduser(file_path), "r") as f:
        return json.load(f)


def format_response(response, output_format="auto", include_headers=False, colorize=True) -> str:
    """Format response for display."""
    # Prepare the output
    result = []

    # Add status line
    status_line = f"HTTP/{response.http_version} {response.status_code} {response.reason_phrase}"
    if colorize:
        if 200 <= response.status_code < 300:
            status_line = f"\033[32m{status_line}\033[0m"  # Green
        elif 400 <= response.status_code < 500:
            status_line = f"\033[33m{status_line}\033[0m"  # Yellow
        elif 500 <= response.status_code < 600:
            status_line = f"\033[31m{status_line}\033[0m"  # Red

    result.append(status_line)

    # Add headers if requested
    if include_headers:
        for name, value in response.headers.items():
            result.append(f"{name}: {value}")
        result.append("")  # Empty line after headers

    # Format body based on content type and requested format
    content_type = response.headers.get("content-type", "").lower()

    try:
        if output_format == "raw" or output_format == "text":
            # Raw output
            result.append(response.text)
        elif "json" in content_type or output_format == "json":
            # JSON output
            try:
                json_data = response.json()
                result.append(json.dumps(json_data, indent=2, sort_keys=True, ensure_ascii=False))
            except Exception:
                # Fall back to raw text if JSON parsing fails
                result.append(response.text)
        elif output_format == "headers":
            # Only show headers
            pass
        elif "html" in content_type or "xml" in content_type:
            # For HTML/XML, show raw text
            result.append(response.text)
        else:
            # Default to showing raw text
            result.append(response.text)
    except Exception as e:
        result.append(f"Error formatting response: {str(e)}")
        result.append(response.text)

    return "\n".join(result)


async def make_request(args):
    """Make an HTTP request based on CLI arguments."""
    url = args.url
    method = args.method

    # Prepare common client settings
    client_kwargs = {
        "timeout": args.timeout,
        "follow_redirects": not args.no_follow_redirects,
        "verify_ssl": not args.no_verify,
        "max_retries": args.retry,
        "enable_cache": args.cache,
        "debug": args.debug,
    }

    # Set up headers
    headers = {}
    if args.headers:
        headers.update(parse_headers(args.headers))

    # Auto-add content type header if needed
    if not any(k.lower() == "content-type" for k in headers):
        if args.json or args.json_file:
            headers["Content-Type"] = "application/json"
        elif args.form:
            headers["Content-Type"] = "application/x-www-form-urlencoded"

    # Prepare request configuration
    request_kwargs = {
        "headers": headers,
    }

    # Set up data or JSON body
    if args.form:
        request_kwargs["data"] = parse_data(args.form)
    elif args.data:
        request_kwargs["content"] = args.data
    elif args.json:
        request_kwargs["json"] = args.json
    elif args.json_file:
        request_kwargs["json"] = parse_json_file(args.json_file)

    # Set up query parameters
    if args.params:
        request_kwargs["params"] = parse_data(args.params)

    # Create client
    async with EnhancedClient(**client_kwargs) as client:
        # Add middleware as needed
        if args.compress:
            compression = CompressionMiddleware(
                compress_requests=True,
                min_size_to_compress=500,  # Lower threshold for CLI
                compress_type=args.compress_type,
            )
            client._middleware_chain.add(compression)

        if args.trace:
            tracing = TracingMiddleware(
                trace_header_name="X-Enhanced-HTTPX-Trace",
                trace_all_headers=args.debug,
            )
            client._middleware_chain.add(tracing)

        try:
            # Start timer
            import time

            start_time = time.time()

            # Make the request
            response = await client.request(method, url, **request_kwargs)

            # Calculate timing
            duration = time.time() - start_time

            # Print timing info if requested
            if args.timing:
                print(f"Request completed in {duration:.3f} seconds", file=sys.stderr)

            # Format and print response
            print(
                format_response(
                    response,
                    output_format=args.output,
                    include_headers=args.include_headers,
                    colorize=not args.no_color,
                )
            )

            return response.status_code
        except Exception as e:
            print(f"Error: {str(e)}", file=sys.stderr)
            if args.debug:
                import traceback

                traceback.print_exc()
            return 1


def create_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(description="Enhanced HTTPX - Command line HTTP client")
    parser.add_argument(
        "url",
        help="URL to request",
    )
    parser.add_argument(
        "-X",
        "--method",
        default="GET",
        help="HTTP method (default: GET)",
    )
    parser.add_argument(
        "-H",
        "--headers",
        action="append",
        help="Headers in the format 'Name: Value'",
    )
    parser.add_argument(
        "-d",
        "--data",
        help="Request body data (raw)",
    )
    parser.add_argument(
        "-f",
        "--form",
        action="append",
        help="Form data in the format 'key=value'",
    )
    parser.add_argument(
        "-j",
        "--json",
        type=json.loads,
        help="JSON data (as string)",
    )
    parser.add_argument(
        "--json-file",
        help="Path to JSON file to use as request body",
    )
    parser.add_argument(
        "-p",
        "--params",
        action="append",
        help="Query parameters in the format 'key=value'",
    )
    parser.add_argument(
        "-o",
        "--output",
        choices=["auto", "json", "text", "raw", "headers"],
        default="auto",
        help="Response output format (default: auto)",
    )
    parser.add_argument(
        "-i",
        "--include-headers",
        action="store_true",
        help="Include response headers in output",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--no-follow-redirects",
        action="store_true",
        help="Don't follow redirects",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Disable SSL certificate verification",
    )
    parser.add_argument(
        "-r",
        "--retry",
        type=int,
        default=3,
        help="Maximum number of retries (default: 3)",
    )
    parser.add_argument(
        "-c",
        "--cache",
        action="store_true",
        help="Enable response caching",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Enable request compression",
    )
    parser.add_argument(
        "--compress-type",
        choices=["gzip", "deflate", "br"],
        default="gzip",
        help="Compression algorithm (default: gzip)",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Enable request tracing",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Show request timing information",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colorized output",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # If input is a pipe and no data is specified, read from stdin
    if (
        args.method != "GET"
        and not sys.stdin.isatty()
        and not (args.data or args.form or args.json or args.json_file)
    ):
        args.data = sys.stdin.read().strip()

    # Run the request
    try:
        status_code = asyncio.run(make_request(args))
        sys.exit(0 if (200 <= status_code < 300) else status_code)
    except KeyboardInterrupt:
        print("Request canceled.", file=sys.stderr)
        sys.exit(130)  # Standard exit code for SIGINT


if __name__ == "__main__":
    main()
