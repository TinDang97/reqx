"""
Utility functions for Reqx client.
"""

import json
import logging
import os
import platform
import shlex
import socket
import time
from typing import Any, Dict, Optional, Tuple, Union
from urllib.parse import urlencode

import httpx
import jsonpath_ng
import orjson
import psutil

logger = logging.getLogger("reqx")


def serialize_json(obj: Any) -> str:
    """
    Serialize an object to a JSON string using orjson for better performance.
    """
    return orjson.dumps(obj).decode("utf-8")


def deserialize_json(json_str: str) -> Any:
    """
    Deserialize a JSON string to an object using orjson for better performance.
    """
    return orjson.loads(json_str)


def log_request(url: str, method: str, headers: Dict[str, str], body: Any) -> None:
    """
    Log details of an outgoing HTTP request.
    """
    logger.debug(f"> {method} {url}")
    for name, value in headers.items():
        logger.debug(f"> {name}: {value}")
    if body:
        if isinstance(body, (dict, list)):
            body_str = serialize_json(body)
        else:
            body_str = str(body)
        if len(body_str) > 1000:
            logger.debug(f"> Body: {body_str[:1000]}... (truncated)")
        else:
            logger.debug(f"> Body: {body_str}")


def log_response(response: httpx.Response) -> None:
    """
    Log details of an HTTP response.
    """
    logger.debug(f"< HTTP {response.status_code} {response.reason_phrase}")
    for name, value in response.headers.items():
        logger.debug(f"< {name}: {value}")
    try:
        if "application/json" in response.headers.get("content-type", ""):
            body = response.json()
            body_str = serialize_json(body)
        else:
            body_str = response.text

        if len(body_str) > 1000:
            logger.debug(f"< Body: {body_str[:1000]}... (truncated)")
        else:
            logger.debug(f"< Body: {body_str}")
    except Exception as e:
        logger.debug(f"< Body (failed to parse): {e}")


def format_curl_command(
    method: str,
    url: str,
    headers: Dict[str, str],
    body: Any,
    params: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Format a curl command equivalent to the HTTP request.

    Args:
        method: HTTP method
        url: Request URL
        headers: Request headers
        body: Request body
        params: Query parameters

    Returns:
        Formatted curl command
    """
    cmd_parts = ["curl", "-X", method]

    # Add URL with query parameters
    if params:
        query_string = urlencode(params)
        if "?" in url:
            url += f"&{query_string}"
        else:
            url += f"?{query_string}"

    cmd_parts.append(shlex.quote(url))

    # Add headers
    for name, value in headers.items():
        cmd_parts.extend(["-H", shlex.quote(f"{name}: {value}")])

    # Add request body
    if body:
        if isinstance(body, (dict, list)):
            body_str = json.dumps(body)
            cmd_parts.extend(["-d", shlex.quote(body_str)])
        elif isinstance(body, str):
            cmd_parts.extend(["-d", shlex.quote(body)])

    # Join all parts with spaces
    return " ".join(cmd_parts)


def _build_replace_dict(source: dict, target: dict, prefix: str = "") -> Dict[str, Tuple[Any, Any]]:
    """Build a dictionary of changes between two dictionaries."""
    result = {}

    # Find added/changed keys
    for key, value in target.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in source:
            result[path] = (None, value)
        elif isinstance(value, dict) and isinstance(source[key], dict):
            result.update(_build_replace_dict(source[key], value, path))
        elif value != source[key]:
            result[path] = (source[key], value)

    # Find removed keys
    for key in source:
        path = f"{prefix}.{key}" if prefix else key
        if key not in target and path not in result:
            result[path] = (source[key], None)

    return result


def dict_diff(source: dict, target: dict) -> Dict[str, Tuple[Any, Any]]:
    """
    Generate a diff between two dictionaries.

    Args:
        source: Source dictionary
        target: Target dictionary

    Returns:
        Dictionary where keys are paths and values are (old_value, new_value) tuples
    """
    return _build_replace_dict(source, target)


def select_json_path(data: Union[dict, list], json_path: str) -> Any:
    """
    Select data using JSONPath expressions.

    Args:
        data: Input data (dict or list)
        json_path: JSONPath expression

    Returns:
        Selected data

    Raises:
        ValueError: If JSONPath is invalid
    """
    try:
        # Parse the JSONPath expression
        jsonpath_expr = jsonpath_ng.parse(json_path)

        # Find all matches
        matches = [match.value for match in jsonpath_expr.find(data)]

        # Return the matches
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            return matches
        else:
            return None
    except Exception as e:
        raise ValueError(f"Invalid JSONPath expression: {str(e)}") from e


def is_rate_limited(response: httpx.Response) -> bool:
    """
    Check if a response indicates rate limiting.

    Args:
        response: HTTP response to check

    Returns:
        True if rate limited, False otherwise
    """
    status_code = response.status_code

    # Common rate limit status codes
    if status_code in (429, 418):  # Too Many Requests, I'm a teapot (used by some APIs)
        return True

    # Check for rate limit headers
    headers = response.headers
    rate_limit_headers = [
        "x-rate-limit-remaining",
        "x-ratelimit-remaining",
        "ratelimit-remaining",
        "retry-after",
        "x-retry-after",
    ]

    for header in rate_limit_headers:
        if header in headers:
            value = headers[header]
            try:
                # If it's a numeric value and zero or negative, it's rate limited
                if float(value) <= 0:
                    return True
            except (ValueError, TypeError):
                # Not a numeric value
                pass

    return False


def get_retry_after(response: httpx.Response) -> Optional[float]:
    """
    Get the retry-after value from a response.

    Args:
        response: HTTP response to check

    Returns:
        Number of seconds to wait before retrying, or None if not available
    """
    headers = response.headers

    # Check Retry-After header (could be seconds or HTTP date)
    retry_after = headers.get("retry-after") or headers.get("x-retry-after")
    if retry_after:
        try:
            # Try to parse as number of seconds
            return float(retry_after)
        except (ValueError, TypeError):
            # Try to parse as HTTP date
            try:
                from email.utils import parsedate_to_datetime

                retry_date = parsedate_to_datetime(retry_after)
                now = time.time()
                return max(0, (retry_date.timestamp() - now))
            except (TypeError, ValueError):
                pass

    # Check rate limit reset headers
    reset_headers = ["x-rate-limit-reset", "x-ratelimit-reset", "ratelimit-reset"]

    for header in reset_headers:
        if header in headers:
            try:
                reset_time = float(headers[header])
                now = time.time()

                # Some APIs return seconds since epoch
                if reset_time > now:
                    return reset_time - now
                # Some APIs return seconds from now
                else:
                    return reset_time
            except (ValueError, TypeError):
                pass

    return None


def sanitize_sensitive_data(data: Any) -> Any:
    """
    Sanitize sensitive data in a dictionary or list.

    Args:
        data: Input data (dict or list)
    Returns:
        Sanitized data
    """
    if isinstance(data, dict):
        return {k: sanitize_sensitive_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_sensitive_data(item) for item in data]
    elif isinstance(data, str):
        # Replace sensitive information with a placeholder
        return data.replace("sensitive", "[REDACTED]")
    else:
        return data


def get_system_resource_metrics() -> Dict[str, Union[int, float]]:
    """
    Get metrics about system resources.

    Returns:
        Dictionary containing system metrics
    """
    metrics = {
        "cpu_count": psutil.cpu_count(logical=False) or 1,  # Physical cores
        "cpu_count_logical": psutil.cpu_count(logical=True) or 1,  # Logical cores
        "memory_gb": psutil.virtual_memory().total / (1024 * 1024 * 1024),  # Memory in GB
        "memory_available_gb": psutil.virtual_memory().available / (1024 * 1024 * 1024),
        "os": platform.system(),
        "python_implementation": platform.python_implementation(),
    }

    if hasattr(socket, "SOMAXCONN"):
        metrics["socket_max_connections"] = socket.SOMAXCONN

    return metrics


def get_optimal_connection_pool_settings() -> Dict[str, int]:
    """
    Calculate optimal connection pool settings based on system resources.

    This function analyzes the available system resources and returns
    recommended connection pool settings for optimal performance.

    Returns:
        Dictionary with recommended connection pool settings
    """
    metrics = get_system_resource_metrics()

    # Base calculations on available CPU cores and memory
    cpu_cores = metrics["cpu_count_logical"]
    memory_gb = metrics["memory_available_gb"]

    # Calculate pool size based on available resources
    # Formula: min(max(8, cores * 4), 100)
    max_connections = min(max(8, cpu_cores * 4), 100)

    # Adjust based on available memory - reduce connections if memory is limited
    if memory_gb < 1.0:  # Less than 1GB available
        max_connections = min(max_connections, 20)
    elif memory_gb < 2.0:  # Less than 2GB available
        max_connections = min(max_connections, 40)
    elif memory_gb > 8.0:  # More than 8GB available
        max_connections = min(max_connections * 2, 200)  # Double up to 200

    # Calculate keepalive connections as a percentage of max
    max_keepalive = int(max_connections * 0.3)

    # Calculate keepalive expiry
    # Use shorter expiry when resources are constrained
    if memory_gb < 1.0:
        keepalive_expiry = 30  # 30 seconds for low memory
    else:
        keepalive_expiry = 60  # 60 seconds default

    return {
        "max_connections": int(max_connections),
        "max_keepalive_connections": int(max_keepalive),
        "keepalive_expiry": int(keepalive_expiry),
    }


def detect_proxy_settings() -> Dict[str, Optional[str]]:
    """
    Detect system proxy settings.

    Returns:
        Dictionary with proxy settings
    """
    proxies = {
        "http": os.environ.get("HTTP_PROXY"),
        "https": os.environ.get("HTTPS_PROXY"),
        "no_proxy": os.environ.get("NO_PROXY"),
    }

    return proxies


def is_known_http2_host(hostname: str) -> bool:
    """
    Check if a hostname is known to support HTTP/2.

    Args:
        hostname: Hostname to check

    Returns:
        True if the host is known to support HTTP/2, False otherwise
    """
    # List of known hosts that support HTTP/2
    http2_hosts = [
        "www.google.com",
        "github.com",
        "www.cloudflare.com",
        "www.facebook.com",
        "www.youtube.com",
        "twitter.com",
        "www.amazon.com",
        "www.reddit.com",
        "www.wikipedia.org",
    ]

    # Check if hostname matches or is a subdomain of a known host
    for known_host in http2_hosts:
        if hostname == known_host or hostname.endswith("." + known_host):
            return True

    return False


def calculate_backoff(retry_count: int, base_delay: float = 0.5, max_delay: float = 60.0) -> float:
    """
    Calculate exponential backoff delay for retries.

    Args:
        retry_count: Current retry attempt number (1-based)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds

    Returns:
        Delay in seconds for the current retry
    """
    # Full jitter exponential backoff
    # delay = min(cap, base * 2 ** attempt) * random(0, 1)
    import random

    # Calculate exponential backoff with full jitter
    delay = min(max_delay, base_delay * (2 ** (retry_count - 1)))
    delay = delay * random.random()

    return delay


def adaptive_retry_strategy(retry_count: int, status_code: Optional[int] = None) -> float:
    """
    Calculate the adaptive retry delay based on retry count and status code.

    This is more advanced than simple exponential backoff because it
    considers the type of error (via status code) to adjust the backoff strategy.

    Args:
        retry_count: Current retry attempt (0-based)
        status_code: HTTP status code from the failed request, if available

    Returns:
        Delay in seconds before the next retry attempt
    """
    # Base delay with exponential backoff and small random jitter
    import random

    base_delay = 0.1 * (2**retry_count)
    jitter = random.uniform(0, 0.1 * base_delay)

    # Adjust based on status code if provided
    status_factor = 1.0
    if status_code is not None:
        if status_code == 429:  # Too Many Requests
            # Increase backoff for rate limiting
            status_factor = 2.0
        elif 500 <= status_code < 600:
            # Server errors might need longer backoffs
            status_factor = 1.5

    # Calculate final delay with limits
    delay = (base_delay * status_factor) + jitter

    # Cap at reasonable maximum (30 seconds)
    return min(delay, 30.0)
