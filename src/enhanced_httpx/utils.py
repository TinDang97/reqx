import orjson
import logging
import time
from typing import Any, Dict, List, Optional, Union, Callable
from jsonpath_ng import parse as jsonpath_parse
from jsonpath_ng.exceptions import JsonPathParserError
from datetime import datetime

logger = logging.getLogger("enhanced_httpx")


def serialize_json(data, default=None):
    """Serialize data to JSON using orjson for maximum performance."""
    return orjson.dumps(
        data,
        default=default,
        option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SERIALIZE_DATACLASS | orjson.OPT_UTC_Z,
    ).decode("utf-8")


def deserialize_json(data):
    """Deserialize JSON data using orjson for maximum performance."""
    if not data:
        return None
    return orjson.loads(data)


def log_request(url, method, headers=None, body=None, params=None):
    """
    Log HTTP request details with proper sanitization of sensitive information.

    Args:
        url: Request URL
        method: HTTP method
        headers: Request headers
        body: Request body
        params: Request query parameters
    """
    sanitized_headers = sanitize_sensitive_data(headers) if headers else {}
    sanitized_body = sanitize_sensitive_data(body) if body else None

    logger.debug(f"Request: {method} {url}")
    if params:
        logger.debug(f"Params: {sanitize_sensitive_data(params)}")
    if sanitized_headers:
        logger.debug(f"Headers: {sanitized_headers}")
    if sanitized_body:
        logger.debug(f"Body: {sanitized_body}")


def log_response(response):
    """
    Log HTTP response details.

    Args:
        response: HTTP response object
    """
    logger.debug(f"Response: {response.status_code} {response.reason_phrase} from {response.url}")
    logger.debug(f"Headers: {response.headers}")

    # Try to log the body, but limit it to avoid excessive logging
    try:
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type or "application/javascript" in content_type:
            body = response.json()
            # Truncate large responses
            if isinstance(body, dict) and len(body) > 5:
                truncated_body = {k: body[k] for k in list(body.keys())[:5]}
                truncated_body["..."] = f"(truncated, {len(body)} keys total)"
                logger.debug(f"Body (truncated): {truncated_body}")
            else:
                logger.debug(f"Body: {body}")
        else:
            text = response.text
            if len(text) > 500:
                logger.debug(f"Body: {text[:500]}... (truncated, {len(text)} characters total)")
            else:
                logger.debug(f"Body: {text}")
    except Exception as e:
        logger.debug(f"Could not parse response body: {str(e)}")


def sanitize_sensitive_data(data: Any) -> Any:
    """
    Sanitize sensitive information in data like passwords, tokens, etc.

    Args:
        data: The data to sanitize

    Returns:
        Sanitized data with sensitive information masked
    """
    sensitive_keys = [
        "password",
        "passwd",
        "secret",
        "token",
        "api_key",
        "apikey",
        "access_token",
        "auth",
        "credentials",
        "private",
    ]

    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            key_lower = key.lower() if isinstance(key, str) else ""
            if any(sensitive in key_lower for sensitive in sensitive_keys) and value:
                sanitized[key] = "********"
            else:
                sanitized[key] = sanitize_sensitive_data(value)
        return sanitized
    elif isinstance(data, list):
        return [sanitize_sensitive_data(item) for item in data]
    else:
        return data


def select_json_path(data: Any, path: str) -> Any:
    """
    Extract data from JSON using JSONPath syntax.

    Args:
        data: JSON data (parsed, not string)
        path: JSONPath expression (e.g., '$.store.book[0].title')

    Returns:
        Extracted data or None if the path doesn't match

    Raises:
        ValueError: If the JSONPath expression is invalid
    """
    try:
        jsonpath_expr = jsonpath_parse(path)
        matches = [match.value for match in jsonpath_expr.find(data)]
        if not matches:
            return None
        elif len(matches) == 1:
            return matches[0]  # Return single value directly
        else:
            return matches  # Return list of values
    except JsonPathParserError as e:
        raise ValueError(f"Invalid JSONPath expression: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error extracting data with JSONPath: {str(e)}")


def time_request(func: Callable) -> Callable:
    """
    A decorator to time HTTP requests for performance monitoring.

    Args:
        func: The function to decorate

    Returns:
        Decorated function that logs timing information
    """

    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.debug(f"Request completed in {elapsed_time:.4f} seconds")
        return result

    return wrapper


def format_curl_command(
    method: str,
    url: str,
    headers: Dict[str, str] = None,
    data: Any = None,
    params: Dict[str, str] = None,
) -> str:
    """
    Format a curl command from request parameters for debugging and sharing.

    Args:
        method: HTTP method
        url: Request URL
        headers: Request headers
        data: Request body
        params: Request query parameters

    Returns:
        curl command as a string
    """
    curl_command = f"curl -X {method}"

    if headers:
        for key, value in headers.items():
            curl_command += f" -H '{key}: {value}'"

    if data:
        if isinstance(data, dict):
            data_str = serialize_json(data)
            curl_command += f" -d '{data_str}'"
        elif isinstance(data, str):
            curl_command += f" -d '{data}'"

    # Add parameters to URL if any
    if params:
        import urllib.parse

        query_string = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
        if "?" in url:
            url = f"{url}&{query_string}"
        else:
            url = f"{url}?{query_string}"

    curl_command += f" '{url}'"
    return curl_command
