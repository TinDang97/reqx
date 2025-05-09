#!/usr/bin/env python
"""
Example script that demonstrates how to use the reqx CLI.
This script shows various commands you can run with the CLI.
"""

import os
import subprocess
import sys
import textwrap

from colorama import Fore, Style, init

# Initialize colorama for cross-platform colored terminal text
init()

# Add parent directory to path for importing reqx
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def print_title(title):
    """Print a formatted title."""
    print(f"\n{Fore.CYAN}{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}{Style.RESET_ALL}")


def print_command(command):
    """Print a formatted command."""
    print(f"\n{Fore.GREEN}$ {command}{Style.RESET_ALL}")


def run_command(command):
    """Run a command and print its output."""
    print_command(command)
    print()

    # Run the command
    process = subprocess.run(
        command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )

    # Print command output
    print(process.stdout)
    return process.returncode


def print_explanation(text):
    """Print a formatted explanation."""
    wrapped_text = textwrap.fill(text, width=60)
    print(f"{Fore.YELLOW}{wrapped_text}{Style.RESET_ALL}\n")


def main():
    """Main function."""
    print_title("Enhanced HTTPX CLI Examples")

    # Example 1: Basic GET request
    print_explanation(
        "Example 1: Basic GET request to httpbin.org/get. "
        "This shows a simple GET request with the reqx CLI."
    )
    run_command("python -m reqx.cli get https://httpbin.org/get")

    # Example 2: GET with headers and parameters
    print_explanation(
        "Example 2: GET request with custom headers and query parameters. "
        "This demonstrates how to add custom headers and query parameters to a request."
    )
    run_command(
        "python -m reqx.cli get https://httpbin.org/get "
        '-H "User-Agent=EnhancedHTTPX-Demo" '
        '-H "Accept=application/json" '
        '-p "param1=value1" -p "param2=value2"'
    )

    # Example 3: POST with JSON data
    print_explanation(
        "Example 3: POST request with JSON data. "
        "This shows how to send JSON data in a POST request."
    )
    run_command(
        "python -m reqx.cli post https://httpbin.org/post "
        '-j \'{"name": "Test User", "email": "test@example.com", "age": 30}\''
    )

    # Example 4: Using JSON path to extract data
    print_explanation(
        "Example 4: Using JSONPath to extract specific data from the response. "
        "This shows how to extract just the fields you need from a JSON response."
    )
    run_command(
        "python -m reqx.cli get https://httpbin.org/json "
        '--json-path "$.slideshow.slides[*].title"'
    )

    # Example 5: Saving response to a file
    print_explanation(
        "Example 5: Saving the response to a file. "
        "This demonstrates how to save the response body to a file."
    )
    response_file = os.path.join(os.path.dirname(__file__), "response.json")
    run_command(f'python -m reqx.cli get https://httpbin.org/get --pretty --save "{response_file}"')
    print(f"Saved response to: {response_file}")

    # Example 6: Generate equivalent curl command
    print_explanation(
        "Example 6: Generate an equivalent curl command. "
        "This shows how to get the curl equivalent of a request, which can be useful for sharing or debugging."
    )
    run_command(
        "python -m reqx.cli post https://httpbin.org/post "
        '-H "Content-Type=application/json" '
        '-H "Authorization=Bearer token123" '
        '-j \'{"data": "test"}\' '
        "--curl"
    )

    # Example 7: Debug mode
    print_explanation(
        "Example 7: Running in debug mode. "
        "This shows how to enable debug logging for detailed request and response information."
    )
    run_command("python -m reqx.cli --debug get https://httpbin.org/headers " '-H "X-Debug=true"')

    print(f"\n{Fore.GREEN}All examples completed!{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
