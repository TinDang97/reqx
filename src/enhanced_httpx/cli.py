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
from tabulate import tabulate
import datetime
from .client import EnhancedClient
from .config import (
    load_config,
    save_config,
    load_request_template,
    save_request_template,
    list_templates,
    RequestTemplate,
    EnhancedHttpxConfig,
    DEFAULT_CONFIG_FILE,
)
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


def save_request_history(history_file, method, url, status_code, elapsed):
    """Save request to history file."""
    if not history_file:
        return

    try:
        # Create history entry
        history_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "method": method,
            "url": url,
            "status_code": status_code,
            "elapsed": elapsed,
        }

        # Load existing history or create new
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            history = []

        # Add new entry and trim if needed
        history.append(history_entry)
        if len(history) > 100:  # Keep last 100 entries
            history = history[-100:]

        # Save history file
        with open(history_file, "w") as f:
            json.dump(history, f)
    except Exception as e:
        logger.error(f"Failed to save request history: {e}")


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.option("--config", type=click.Path(), help="Path to config file")
@click.pass_context
def cli(ctx, debug, config):
    """Enhanced HTTPX CLI - A command-line tool for making HTTP requests."""
    if debug:
        logging.getLogger("enhanced_httpx").setLevel(logging.DEBUG)

    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug

    # Load configuration
    try:
        ctx.obj["config"] = load_config(config)
        if debug:
            logger.debug(f"Loaded configuration from {config or DEFAULT_CONFIG_FILE}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        ctx.obj["config"] = None


@cli.command()
@click.argument("template_name")
@click.option(
    "--method",
    type=click.Choice(["GET", "POST", "PUT", "PATCH", "DELETE"]),
    default="GET",
    help="HTTP method",
)
@click.option("--url", required=True, help="URL for the template")
@click.option(
    "-H",
    "--header",
    "headers",
    multiple=True,
    callback=parse_key_value,
    help="Headers in KEY=VALUE format",
)
@click.option(
    "-p",
    "--param",
    "params",
    multiple=True,
    callback=parse_key_value,
    help="Query parameters in KEY=VALUE format",
)
@click.option("-b", "--body", help="Request body as JSON string")
@click.option("-d", "--description", help="Template description")
@click.pass_context
def save_template(ctx, template_name, method, url, headers, params, body, description):
    """Save a request as a named template."""
    config = ctx.obj.get("config")
    if not config:
        click.echo("Error: Configuration not loaded", err=True)
        return

    # Parse body if provided
    body_data = None
    if body:
        try:
            body_data = json.loads(body)
        except json.JSONDecodeError:
            click.echo("Error: Body must be valid JSON", err=True)
            return

    # Create template
    template = RequestTemplate(
        name=template_name,
        method=method,
        url=url,
        headers=headers,
        params=params,
        body=body_data,
        description=description,
    )

    # Save template
    if save_request_template(template, config):
        click.echo(f"Template '{template_name}' saved successfully")
    else:
        click.echo("Failed to save template", err=True)


@cli.command(name="list-templates")
@click.pass_context
def list_all_templates(ctx):
    """List all available request templates."""
    config = ctx.obj.get("config")
    if not config:
        click.echo("Error: Configuration not loaded", err=True)
        return

    templates = list_templates(config)

    if not templates:
        click.echo("No templates found")
        return

    # Display templates in a table
    table_data = [(name, description or "") for name, description in templates]
    headers = ["Template Name", "Description"]
    click.echo(tabulate(table_data, headers=headers, tablefmt="grid"))


@cli.command()
@click.argument("template_name")
@click.option(
    "-H",
    "--header",
    "headers",
    multiple=True,
    callback=parse_key_value,
    help="Additional headers in KEY=VALUE format",
)
@click.option(
    "-p",
    "--param",
    "params",
    multiple=True,
    callback=parse_key_value,
    help="Additional query parameters in KEY=VALUE format",
)
@click.option("-j", "--json", "json_data", help="Override request body as a JSON string")
@click.option(
    "-f",
    "--file",
    "json_file",
    type=click.Path(exists=True),
    help="JSON file to override request body",
)
@click.option("-t", "--timeout", type=float, help="Request timeout in seconds")
@click.option("--json-path", help="JSONPath expression to extract specific data")
@click.option("--pretty", is_flag=True, help="Pretty-print JSON output")
@click.option("--save", type=click.Path(), help="Save response to file")
@click.option("--curl", is_flag=True, help="Print equivalent curl command")
@click.pass_context
async def template(
    ctx,
    template_name,
    headers,
    params,
    json_data,
    json_file,
    timeout,
    json_path,
    pretty,
    save,
    curl,
):
    """Execute a saved request template."""
    config = ctx.obj.get("config")
    if not config:
        click.echo("Error: Configuration not loaded", err=True)
        return

    # Load the template
    template = load_request_template(template_name, config)
    if not template:
        click.echo(f"Error: Template '{template_name}' not found", err=True)
        return

    # Merge command line parameters with template
    merged_headers = {**template.headers, **headers}
    merged_params = {**template.params, **params}

    # Determine body
    body = template.body
    if json_file:
        body = load_json_file(json_file)
    elif json_data:
        try:
            body = json.loads(json_data)
        except json.JSONDecodeError:
            raise click.BadParameter(f"Invalid JSON: {json_data}")

    # Use timeout from CLI or default from config
    actual_timeout = timeout or config.default_timeout

    # Execute the request
    await make_request(
        ctx,
        template.method,
        template.url,
        merged_headers,
        {},  # cookies not stored in templates
        merged_params,
        body,
        actual_timeout,
        config.verify_ssl,
        config.follow_redirects,
        json_path,
        pretty,
        save,
        curl,
    )


@cli.command()
@click.option("-n", "--count", type=int, default=10, help="Number of entries to show")
@click.pass_context
def history(ctx, count):
    """Show request history."""
    config = ctx.obj.get("config")
    if not config or not config.history_file:
        click.echo("Error: History tracking not configured", err=True)
        return

    try:
        # Load history file
        try:
            with open(config.history_file, "r") as f:
                history = json.load(f)
        except FileNotFoundError:
            click.echo("No request history found")
            return

        # Sort by timestamp (newest first) and take the requested number
        history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        history = history[:count]

        # Format for display
        table_data = []
        for entry in history:
            timestamp = datetime.datetime.fromisoformat(entry.get("timestamp", ""))
            formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            elapsed = entry.get("elapsed", 0)
            table_data.append(
                [
                    formatted_time,
                    entry.get("method", ""),
                    entry.get("status_code", ""),
                    f"{elapsed:.2f}s" if isinstance(elapsed, (int, float)) else elapsed,
                    entry.get("url", ""),
                ]
            )

        headers = ["Time", "Method", "Status", "Elapsed", "URL"]
        click.echo(tabulate(table_data, headers=headers, tablefmt="grid"))

    except Exception as e:
        click.echo(f"Error reading history: {str(e)}", err=True)


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
@click.option("-t", "--timeout", type=float, help="Request timeout in seconds")
@click.option("--no-verify", is_flag=True, help="Disable SSL verification")
@click.option("--no-redirect", is_flag=True, help="Disable following redirects")
@click.option("-j", "--json-path", help="JSONPath expression to extract specific data")
@click.option("--pretty", is_flag=True, help="Pretty-print JSON output")
@click.option("--save", type=click.Path(), help="Save response to file")
@click.option("--curl", is_flag=True, help="Print equivalent curl command")
@click.option("--save-as-template", help="Save this request as a template with the given name")
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
    save_as_template,
):
    """Send a GET request."""
    # Apply config defaults
    config = ctx.obj.get("config")
    if config:
        timeout = timeout or config.default_timeout
        verify_ssl = not no_verify and config.verify_ssl
        follow_redirects = not no_redirect and config.follow_redirects
        # Merge with default headers from config
        if config.default_headers:
            merged_headers = {**config.default_headers, **headers}
        else:
            merged_headers = headers
    else:
        timeout = timeout or 30.0
        verify_ssl = not no_verify
        follow_redirects = not no_redirect
        merged_headers = headers

    # Save as template if requested
    if save_as_template and config:
        template = RequestTemplate(
            name=save_as_template,
            method="GET",
            url=url,
            headers=merged_headers,
            params=params,
        )
        save_request_template(template, config)
        click.echo(f"Saved as template: {save_as_template}")

    await make_request(
        ctx,
        "GET",
        url,
        merged_headers,
        cookies,
        params,
        None,
        timeout,
        verify_ssl,
        follow_redirects,
        json_path,
        pretty,
        save,
        curl,
    )


# Similar updates for POST, PUT, PATCH, DELETE methods...
# For brevity, I'm not including all of them in this update


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
    config = ctx.obj.get("config")

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

            # Save to history if enabled
            if config and config.history_file:
                save_request_history(
                    config.history_file, method, url, response.status_code, elapsed
                )

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
