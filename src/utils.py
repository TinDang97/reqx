"""
Utility functions for enhanced-httpx.
"""

import json
import logging
import shlex
import time
from typing import Any, Dict, Optional, Tuple, Union
from urllib.parse import urlencode

import httpx
import jsonpath_ng
import orjson

logger = logging.getLogger("enhanced_httpx")


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
