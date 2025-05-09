#!/usr/bin/env python
"""
Advanced Features Example for enhanced-httpx

This example demonstrates the new powerful features added to the enhanced-httpx library:
- Certificate pinning for improved security
- Circuit breaker for resilience
- Memory-aware streaming for large downloads
- Session management with OAuth2 authentication
- Request tracing for debugging
- GraphQL client for working with GraphQL APIs
- Webhook sending and receiving
"""

import asyncio
import json
import logging
import os

# Add parent directory to path for importing enhanced_httpx
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import EnhancedClient
from src.graphql import GraphQLClient, GraphQLResponse
from src.middleware import CircuitBreakerMiddleware, MemoryAwareMiddleware, TracingMiddleware
from src.session import Session
from src.webhook import WebhookClient, WebhookEvent

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("example")


async def demo_certificate_pinning():
    """Demonstrate certificate pinning."""
    logger.info("===== Certificate Pinning Demo =====")

    # Create a client with certificate pinning
    async with EnhancedClient(
        verify_ssl=True,
        certificate_pins={
            "httpbin.org": [
                # This is a fake pin - in production, you'd extract the real one from the certificate
                "sha256/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
            ]
        },
    ) as client:
        try:
            # This will likely fail due to mismatched pins (which is good!)
            await client.get("https://httpbin.org/get")
        except Exception as e:
            logger.info(f"Certificate pinning correctly prevented a connection: {e}")

    logger.info("To use certificate pinning properly, extract the real certificate hash")
    logger.info("from your trusted server and use that in production\n")


async def demo_circuit_breaker():
    """Demonstrate the circuit breaker pattern."""
    logger.info("===== Circuit Breaker Demo =====")

    # Create circuit breaker middleware
    circuit_breaker = CircuitBreakerMiddleware(
        failure_threshold=2,  # Open circuit after 2 failures
        recovery_timeout=5.0,  # Wait 5 seconds before trying again
        error_codes=[500, 502, 503, 504],  # Server errors
    )

    async with EnhancedClient() as client:
        # Add the circuit breaker middleware
        client._middleware_chain.add(circuit_breaker)

        # Try requests to non-existent endpoint
        failure_url = "https://httpbin.org/status/500"

        try:
            # First request - should fail but circuit remains closed
            await client.get(failure_url)
        except Exception as e:
            logger.info(f"First request failed as expected: {e}")

        try:
            # Second request - should fail and open the circuit
            await client.get(failure_url)
        except Exception as e:
            logger.info(f"Second request failed and opened circuit: {e}")

        try:
            # Third request - circuit is now open, should fail fast
            await client.get(failure_url)
        except Exception as e:
            logger.info(f"Third request failed fast due to open circuit: {e}")

        # Make a request to a working endpoint (different host)
        try:
            response = await client.get("https://httpbin.org/get")
            logger.info(f"Request to different host succeeded: {response.status_code}")
        except Exception as e:
            logger.info(f"Unexpected error: {e}")

    logger.info("Circuit breaker prevented cascading failures and saved resources\n")


async def demo_memory_aware_streaming():
    """Demonstrate memory-aware streaming for large downloads."""
    logger.info("===== Memory-Aware Streaming Demo =====")

    memory_middleware = MemoryAwareMiddleware(
        max_in_memory_size=1024 * 1024,  # 1MB threshold (very low for demo)
        check_content_length=True,
    )

    async with EnhancedClient() as client:
        # Add memory-aware middleware
        client._middleware_chain.add(memory_middleware)

        # Normal small request
        response = await client.get("https://httpbin.org/get")
        logger.info(f"Small response loaded normally: {len(response.content)} bytes")

        # Large request - will trigger warning to use streaming
        large_url = "https://httpbin.org/bytes/2000000"  # 2MB of random data
        response = await client.get(large_url)
        logger.info(f"Large response warning triggered: {len(response.content)} bytes")

        # Now use proper streaming for large download
        file_path = "/tmp/large_download_example.bin"
        logger.info(f"Downloading large file to {file_path} with streaming...")
        await client.download_file(large_url, file_path)

        # Check file size
        file_size = os.path.getsize(file_path)
        logger.info(f"Successfully downloaded {file_size} bytes to file")

        # Clean up
        os.unlink(file_path)

    logger.info("Memory-aware streaming helps prevent out-of-memory issues\n")


async def demo_tracing_middleware():
    """Demonstrate request tracing for debugging."""
    logger.info("===== Request Tracing Demo =====")

    tracing_middleware = TracingMiddleware(
        trace_header_name="X-Demo-Trace-ID",
        include_in_response=True,
        trace_all_headers=True,
    )

    async with EnhancedClient() as client:
        # Add tracing middleware
        client._middleware_chain.add(tracing_middleware)

        # Make a request
        response = await client.get("https://httpbin.org/get")

        # Extract the trace ID from the response
        trace_id = response.headers.get("X-Demo-Trace-ID")
        logger.info(f"Request was traced with ID: {trace_id}")

        # Make another request that will have the same trace
        headers = {"X-Demo-Trace-ID": trace_id}
        response = await client.get("https://httpbin.org/headers", headers=headers)

        # The request will carry the trace ID
        echo_headers = response.json()["headers"]
        logger.info(f"Trace ID was propagated: {echo_headers.get('X-Demo-Trace-Id')}")

    logger.info("Request tracing helps with debugging and monitoring\n")


async def demo_session_management():
    """Demonstrate session management with authentication."""
    logger.info("===== Session Management Demo =====")

    # Create a session with OAuth2 configuration (using httpbin.org's auth endpoints)
    auth_config = {
        "type": "bearer",
        "token": "demo-token-12345",  # For a real OAuth2 flow, this would be obtained dynamically
    }

    # Create and use a session
    async with Session(
        base_url="https://httpbin.org",
        auth=auth_config,
        persist=False,  # Don't persist session to disk for this demo
    ) as session:
        # Session will automatically authenticate
        await session.authenticate()

        # Make an authenticated request
        response = await session.get("/headers")
        headers = response.json()["headers"]

        logger.info("Session automatically added authentication:")
        logger.info(f"Authorization: {headers.get('Authorization')}")

        # Update session headers
        session.update_headers({"X-Custom-Header": "Demo-Value"})

        # Make another request with updated headers
        response = await session.get("/headers")
        headers = response.json()["headers"]

        logger.info("Session maintained updated headers:")
        logger.info(f"X-Custom-Header: {headers.get('X-Custom-Header')}")

    logger.info("Session management simplifies authentication and state maintenance\n")


async def demo_graphql_client():
    """Demonstrate GraphQL client capabilities."""
    logger.info("===== GraphQL Client Demo =====")

    # Using the public GraphQL Countries API for demonstration
    graphql_endpoint = "https://countries.trevorblades.com/graphql"

    # Create a GraphQL client
    async with GraphQLClient(graphql_endpoint) as graphql:
        # Simple query
        query = """
        query GetCountries {
          countries {
            code
            name
            currency
          }
        }
        """

        # Execute the query
        logger.info("Executing GraphQL query...")
        result = await graphql.query(query)

        if isinstance(result, GraphQLResponse):
            if result.has_errors:
                logger.error(f"GraphQL errors: {result.errors}")
            else:
                countries = result.data.get("countries", [])
                logger.info(f"Retrieved {len(countries)} countries")
                # Show first 3 countries
                for country in countries[:3]:
                    logger.info(f"{country['name']} ({country['code']}): {country['currency']}")

        # Query with variables
        query_with_vars = """
        query GetCountry($code: ID!) {
          country(code: $code) {
            name
            native
            capital
            emoji
            languages {
              name
            }
          }
        }
        """

        variables = {"code": "US"}

        # Execute query with variables
        logger.info("\nExecuting GraphQL query with variables...")
        result = await graphql.query(query_with_vars, variables=variables)

        if isinstance(result, GraphQLResponse) and result.data:
            country = result.data.get("country", {})
            if country:
                logger.info(f"Country: {country['name']} {country['emoji']}")
                logger.info(f"Capital: {country['capital']}")
                languages = [lang["name"] for lang in country.get("languages", [])]
                logger.info(f"Languages: {', '.join(languages)}")

    logger.info("GraphQL client simplifies working with GraphQL APIs\n")


async def demo_webhooks():
    """Demonstrate webhook sending and receiving."""
    logger.info("===== Webhook Demo =====")

    # This is just a simulation since we can't actually set up a webhook server in this example
    webhook_secret = "webhook-secret-12345"

    # Create a webhook client
    webhook_client = WebhookClient(
        signing_secret=webhook_secret,
        retry_attempts=2,
    )

    # Create a webhook event
    event = WebhookEvent(
        event_type="user.created",
        data={
            "user_id": "usr_123456",
            "name": "John Doe",
            "email": "john@example.com",
            "created_at": "2023-05-09T12:30:45Z",
        },
    )

    logger.info("Preparing webhook event:")
    logger.info(f"  ID: {event.id}")
    logger.info(f"  Type: {event.event_type}")
    logger.info(f"  Data: {json.dumps(event.data)}")

    # In a real scenario, we'd send to a real endpoint
    # For demonstration, we'll log the webhook preparation

    # Create a signature
    payload = event.model_dump()

    import base64
    import hashlib
    import hmac

    payload_str = json.dumps(payload, sort_keys=True)
    signature = hmac.new(
        key=webhook_secret.encode(), msg=payload_str.encode(), digestmod=hashlib.sha256
    ).digest()
    signature_b64 = base64.b64encode(signature).decode()

    logger.info("\nWebhook would be sent with:")
    logger.info(f"  X-Webhook-Signature: {signature_b64}")
    logger.info(f"  X-Webhook-Delivery: {event.id}")

    logger.info("\nTo receive webhooks, set up a WebhookHandler:")
    logger.info("  handler = WebhookHandler(secret=webhook_secret)")
    logger.info("  handler.add_event_handler('user.created', process_user_created)")
    logger.info("  handler.run(host='0.0.0.0', port=8000)")

    # Simulate receiving and validating a webhook
    logger.info("\nSimulating webhook signature validation:")

    # Recreate signature from "received" payload
    received_sig = signature_b64

    # Verify signature
    expected_sig = hmac.new(
        key=webhook_secret.encode(), msg=payload_str.encode(), digestmod=hashlib.sha256
    ).digest()

    is_valid = hmac.compare_digest(base64.b64decode(received_sig), expected_sig)

    logger.info(f"Webhook signature is valid: {is_valid}")

    # Close webhook client
    await webhook_client.close()

    logger.info("Webhooks enable secure event notifications between systems\n")


async def main():
    """Run all the demonstrations."""
    logger.info("Enhanced-HTTPX Advanced Features Demo")
    logger.info("====================================\n")

    await demo_circuit_breaker()
    await demo_memory_aware_streaming()
    await demo_tracing_middleware()
    await demo_session_management()
    await demo_graphql_client()
    await demo_webhooks()

    # Skip certificate pinning demo by default as it requires proper certificate setup
    # await demo_certificate_pinning()

    logger.info("All demonstrations completed!")


if __name__ == "__main__":
    asyncio.run(main())
