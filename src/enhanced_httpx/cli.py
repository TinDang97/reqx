import asyncio
import json
import sys
import os
import click
from typing import Optional, Dict, Any, List
import orjson
import uvloop
import logging
from pathlib import Path
from .client import EnhancedClient
from .models import HttpMethod
from .utils import serialize_json, deserialize_json, format_curl_command, select_json_path

# Set uvloop as the event loop policy
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("enhanced_httpx.cli")


def parse_key_value(ctx, param, value):
    """Parse key-value pairs from command line arguments."""
    if not value:
        return {}

    result = {}
    for item in value:
        if "=" not in item:
            raise click.BadParameter(f"Invalid key-value pair: {item}. Expected format: key=value")
        key, val = item.split("=", 1)
        result[key] = val
    return result


def load_json_file(file_path):
    """Load JSON data from a file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise click.BadParameter(f"Failed to load JSON file '{file_path}': {str(e)}")


def get_content_type(headers):
    """Get content type from headers."""
    headers_lower = {k.lower(): v for k, v in headers.items()}
    return headers_lower.get("content-type", "")


def parse_response(response, json_path=None, pretty=False):
    """Parse response for display."""
    try:
        if "application/json" in get_content_type(response.headers):
            data = response.json()

            # Apply JSON path if provided
            if json_path:
                try:
                    data = select_json_path(data, json_path)
                except Exception as e:
                    click.echo(f"Error applying JSON path: {str(e)}", err=True)

            if pretty:
                return json.dumps(data, indent=2)
            else:
                return serialize_json(data)
        else:
            return response.text
    except Exception as e:
        return f"Failed to parse response: {str(e)}\n{response.text}"


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx, debug):
    """Enhanced HTTPX CLI - A command-line tool for making HTTP requests."""
    if debug:
        logging.getLogger("enhanced_httpx").setLevel(logging.DEBUG)
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug


@cli.command()
@click.argument("url", required=True)
@click.option(
    "-H",
    "--header",
    "headers",
    multiple=True,
    callback=parse_key_value,
    help="Custom headers in KEY=VALUE format",
)
@click.option(
    "-c",
    "--cookie",
    "cookies",
    multiple=True,
    callback=parse_key_value,
    help="Cookies in KEY=VALUE format",
)
@click.option(
    "-p",
    "--param",
    "params",
    multiple=True,
    callback=parse_key_value,
    help="Query parameters in KEY=VALUE format",
)
@click.option("-t", "--timeout", type=float, default=30.0, help="Request timeout in seconds")
@click.option("--no-verify", is_flag=True, help="Disable SSL verification")
@click.option("--no-redirect", is_flag=True, help="Disable following redirects")
@click.option("-j", "--json-path", help="JSONPath expression to extract specific data")
@click.option("--pretty", is_flag=True, help="Pretty-print JSON output")
@click.option("--save", type=click.Path(), help="Save response to file")
@click.option("--curl", is_flag=True, help="Print equivalent curl command")
@click.pass_context
async def get(
    ctx,
    url,
    headers,
    cookies,
    params,
    timeout,
    no_verify,
    no_redirect,
    json_path,
    pretty,
    save,
    curl,
):
    """Send a GET request."""
    await make_request(
        ctx,
        "GET",
        url,
        headers,
        cookies,
        params,
        None,
        timeout,
        not no_verify,
        not no_redirect,
        json_path,
        pretty,
        save,
        curl,
    )


@cli.command()
@click.argument("url", required=True)
@click.option(
    "-H",
    "--header",
    "headers",
    multiple=True,
    callback=parse_key_value,
    help="Custom headers in KEY=VALUE format",
)
@click.option(
    "-c",
    "--cookie",
    "cookies",
    multiple=True,
    callback=parse_key_value,
    help="Cookies in KEY=VALUE format",
)
@click.option(
    "-p",
    "--param",
    "params",
    multiple=True,
    callback=parse_key_value,
    help="Query parameters in KEY=VALUE format",
)
@click.option("-d", "--data", help="Request body as a string")
@click.option("-j", "--json", "json_data", help="Request body as a JSON string")
@click.option(
    "-f",
    "--file",
    "json_file",
    type=click.Path(exists=True),
    help="JSON file to use as request body",
)
@click.option("-t", "--timeout", type=float, default=30.0, help="Request timeout in seconds")
@click.option("--no-verify", is_flag=True, help="Disable SSL verification")
@click.option("--no-redirect", is_flag=True, help="Disable following redirects")
@click.option("--json-path", help="JSONPath expression to extract specific data")
@click.option("--pretty", is_flag=True, help="Pretty-print JSON output")
@click.option("--save", type=click.Path(), help="Save response to file")
@click.option("--curl", is_flag=True, help="Print equivalent curl command")
@click.pass_context
async def post(
    ctx,
    url,
    headers,
    cookies,
    params,
    data,
    json_data,
    json_file,
    timeout,
    no_verify,
    no_redirect,
    json_path,
    pretty,
    save,
    curl,
):
    """Send a POST request."""
    body = None
    if json_file:
        body = load_json_file(json_file)
    elif json_data:
        try:
            body = json.loads(json_data)
        except json.JSONDecodeError:
            raise click.BadParameter(f"Invalid JSON: {json_data}")
    elif data:
        body = data

    await make_request(
        ctx,
        "POST",
        url,
        headers,
        cookies,
        params,
        body,
        timeout,
        not no_verify,
        not no_redirect,
        json_path,
        pretty,
        save,
        curl,
    )


@cli.command()
@click.argument("url", required=True)
@click.option(
    "-H",
    "--header",
    "headers",
    multiple=True,
    callback=parse_key_value,
    help="Custom headers in KEY=VALUE format",
)
@click.option(
    "-c",
    "--cookie",
    "cookies",
    multiple=True,
    callback=parse_key_value,
    help="Cookies in KEY=VALUE format",
)
@click.option(
    "-p",
    "--param",
    "params",
    multiple=True,
    callback=parse_key_value,
    help="Query parameters in KEY=VALUE format",
)
@click.option("-d", "--data", help="Request body as a string")
@click.option("-j", "--json", "json_data", help="Request body as a JSON string")
@click.option(
    "-f",
    "--file",
    "json_file",
    type=click.Path(exists=True),
    help="JSON file to use as request body",
)
@click.option("-t", "--timeout", type=float, default=30.0, help="Request timeout in seconds")
@click.option("--no-verify", is_flag=True, help="Disable SSL verification")
@click.option("--no-redirect", is_flag=True, help="Disable following redirects")
@click.option("--json-path", help="JSONPath expression to extract specific data")
@click.option("--pretty", is_flag=True, help="Pretty-print JSON output")
@click.option("--save", type=click.Path(), help="Save response to file")
@click.option("--curl", is_flag=True, help="Print equivalent curl command")
@click.pass_context
async def put(
    ctx,
    url,
    headers,
    cookies,
    params,
    data,
    json_data,
    json_file,
    timeout,
    no_verify,
    no_redirect,
    json_path,
    pretty,
    save,
    curl,
):
    """Send a PUT request."""
    body = None
    if json_file:
        body = load_json_file(json_file)
    elif json_data:
        try:
            body = json.loads(json_data)
        except json.JSONDecodeError:
            raise click.BadParameter(f"Invalid JSON: {json_data}")
    elif data:
        body = data

    await make_request(
        ctx,
        "PUT",
        url,
        headers,
        cookies,
        params,
        body,
        timeout,
        not no_verify,
        not no_redirect,
        json_path,
        pretty,
        save,
        curl,
    )


@cli.command()
@click.argument("url", required=True)
@click.option(
    "-H",
    "--header",
    "headers",
    multiple=True,
    callback=parse_key_value,
    help="Custom headers in KEY=VALUE format",
)
@click.option(
    "-c",
    "--cookie",
    "cookies",
    multiple=True,
    callback=parse_key_value,
    help="Cookies in KEY=VALUE format",
)
@click.option(
    "-p",
    "--param",
    "params",
    multiple=True,
    callback=parse_key_value,
    help="Query parameters in KEY=VALUE format",
)
@click.option("-d", "--data", help="Request body as a string")
@click.option("-j", "--json", "json_data", help="Request body as a JSON string")
@click.option(
    "-f",
    "--file",
    "json_file",
    type=click.Path(exists=True),
    help="JSON file to use as request body",
)
@click.option("-t", "--timeout", type=float, default=30.0, help="Request timeout in seconds")
@click.option("--no-verify", is_flag=True, help="Disable SSL verification")
@click.option("--no-redirect", is_flag=True, help="Disable following redirects")
@click.option("--json-path", help="JSONPath expression to extract specific data")
@click.option("--pretty", is_flag=True, help="Pretty-print JSON output")
@click.option("--save", type=click.Path(), help="Save response to file")
@click.option("--curl", is_flag=True, help="Print equivalent curl command")
@click.pass_context
async def patch(
    ctx,
    url,
    headers,
    cookies,
    params,
    data,
    json_data,
    json_file,
    timeout,
    no_verify,
    no_redirect,
    json_path,
    pretty,
    save,
    curl,
):
    """Send a PATCH request."""
    body = None
    if json_file:
        body = load_json_file(json_file)
    elif json_data:
        try:
            body = json.loads(json_data)
        except json.JSONDecodeError:
            raise click.BadParameter(f"Invalid JSON: {json_data}")
    elif data:
        body = data

    await make_request(
        ctx,
        "PATCH",
        url,
        headers,
        cookies,
        params,
        body,
        timeout,
        not no_verify,
        not no_redirect,
        json_path,
        pretty,
        save,
        curl,
    )


@cli.command()
@click.argument("url", required=True)
@click.option(
    "-H",
    "--header",
    "headers",
    multiple=True,
    callback=parse_key_value,
    help="Custom headers in KEY=VALUE format",
)
@click.option(
    "-c",
    "--cookie",
    "cookies",
    multiple=True,
    callback=parse_key_value,
    help="Cookies in KEY=VALUE format",
)
@click.option(
    "-p",
    "--param",
    "params",
    multiple=True,
    callback=parse_key_value,
    help="Query parameters in KEY=VALUE format",
)
@click.option("-t", "--timeout", type=float, default=30.0, help="Request timeout in seconds")
@click.option("--no-verify", is_flag=True, help="Disable SSL verification")
@click.option("--no-redirect", is_flag=True, help="Disable following redirects")
@click.option("--pretty", is_flag=True, help="Pretty-print JSON output")
@click.option("--save", type=click.Path(), help="Save response to file")
@click.option("--curl", is_flag=True, help="Print equivalent curl command")
@click.pass_context
async def delete(
    ctx, url, headers, cookies, params, timeout, no_verify, no_redirect, pretty, save, curl
):
    """Send a DELETE request."""
    await make_request(
        ctx,
        "DELETE",
        url,
        headers,
        cookies,
        params,
        None,
        timeout,
        not no_verify,
        not no_redirect,
        None,
        pretty,
        save,
        curl,
    )


async def make_request(
    ctx,
    method,
    url,
    headers,
    cookies,
    params,
    body,
    timeout,
    verify_ssl,
    follow_redirects,
    json_path,
    pretty,
    save,
    curl,
):
    """Make an HTTP request and display the response."""
    debug = ctx.obj.get("debug", False)

    # Add Content-Type header if sending JSON and not already set
    if isinstance(body, (dict, list)) and "content-type" not in {
        k.lower(): v for k, v in headers.items()
    }:
        headers = dict(headers)
        headers["Content-Type"] = "application/json"

    if curl:
        # Print equivalent curl command
        cmd = format_curl_command(method, url, headers, body, params)
        click.echo(cmd)
        return

    try:
        async with EnhancedClient(debug=debug) as client:
            start_time = asyncio.get_event_loop().time()

            # Make the request
            response = await client.request(
                method=method,
                url=url,
                headers=headers,
                cookies=cookies,
                params=params,
                json=body if isinstance(body, (dict, list)) else None,
                data=body if isinstance(body, str) else None,
                timeout=timeout,
                follow_redirects=follow_redirects,
                verify_ssl=verify_ssl,
            )

            elapsed = asyncio.get_event_loop().time() - start_time

            # Display response info
            click.echo(
                f"HTTP/{response.http_version} {response.status_code} {response.reason_phrase}"
            )
            click.echo(f"Request completed in {elapsed:.4f}s")

            # Display headers if debug
            if debug:
                click.echo("\nResponse headers:")
                for key, value in response.headers.items():
                    click.echo(f"{key}: {value}")

            # Parse and display response body
            response_content = parse_response(response, json_path, pretty)

            # Save to file if requested
            if save:
                try:
                    with open(save, "w") as f:
                        f.write(response_content)
                    click.echo(f"\nResponse saved to {save}")
                except Exception as e:
                    click.echo(f"Error saving response: {str(e)}", err=True)

            # Display response body
            click.echo("\nResponse body:")
            click.echo(response_content)

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli(obj={})
