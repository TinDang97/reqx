"""
Webhook client and server for enhanced-httpx.

This module provides functionality for both sending and receiving webhooks.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from .client import EnhancedClient

logger = logging.getLogger("enhanced_httpx.webhook")


class WebhookEvent(BaseModel):
    """Base class for webhook events."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str
    timestamp: float = Field(default_factory=time.time)
    data: Dict[str, Any]


class WebhookDeliveryResult(BaseModel):
    """Result of a webhook delivery attempt."""

    success: bool
    status_code: Optional[int] = None
    response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    delivery_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)


class WebhookClient:
    """
    Client for sending webhook events to endpoints.

    Features:
    - Signing of webhook payloads
    - Delivery retries
    - Payload validation
    - Asynchronous batch delivery
    """

    def __init__(
        self,
        client: Optional[EnhancedClient] = None,
        default_headers: Dict[str, str] = None,
        signing_secret: Optional[str] = None,
        retry_attempts: int = 3,
        retry_backoff: float = 2.0,
    ):
        """
        Initialize a webhook client.

        Args:
            client: Optional EnhancedClient instance
            default_headers: Default headers to include with webhook requests
            signing_secret: Secret used to sign webhook payloads
            retry_attempts: Number of delivery retry attempts
            retry_backoff: Backoff factor for retries
        """
        self.client = client or EnhancedClient(
            timeout=10.0,  # Shorter timeout for webhooks
            max_retries=retry_attempts,
            retry_backoff=retry_backoff,
        )
        self.default_headers = default_headers or {
            "Content-Type": "application/json",
            "User-Agent": "Enhanced-HTTPX-Webhook-Client/1.0",
        }
        self.signing_secret = signing_secret
        self.retry_attempts = retry_attempts

    async def __aenter__(self):
        """Support for async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting context."""
        await self.close()

    async def close(self):
        """Close the underlying client."""
        await self.client.close()

    async def send_webhook(
        self,
        url: str,
        event: Union[WebhookEvent, Dict[str, Any]],
        headers: Dict[str, str] = None,
        timeout: float = None,
        sign_payload: bool = True,
    ) -> WebhookDeliveryResult:
        """
        Send a webhook event to the specified URL.

        Args:
            url: The webhook URL to send the event to
            event: The webhook event to send
            headers: Optional additional headers
            timeout: Optional request timeout
            sign_payload: Whether to sign the payload (if signing_secret is set)

        Returns:
            Result of the webhook delivery attempt
        """
        # Convert event to dict if it's a model
        if isinstance(event, WebhookEvent):
            payload = event.model_dump()
        else:
            # Ensure it has basic webhook fields
            payload = dict(event)
            if "id" not in payload:
                payload["id"] = str(uuid.uuid4())
            if "timestamp" not in payload:
                payload["timestamp"] = time.time()

        # Prepare headers
        request_headers = dict(self.default_headers)
        if headers:
            request_headers.update(headers)

        # Sign payload if requested and secret is available
        if sign_payload and self.signing_secret:
            signature = self._generate_signature(payload)
            request_headers["X-Webhook-Signature"] = signature

        # Generate a delivery ID for tracking
        delivery_id = str(uuid.uuid4())
        request_headers["X-Webhook-Delivery"] = delivery_id

        try:
            # Send the webhook
            response = await self.client.post(
                url=url,
                json=payload,
                headers=request_headers,
                timeout=timeout,
            )

            # Parse response if possible
            try:
                response_data = response.json()
            except Exception:
                response_data = {"text": response.text}

            return WebhookDeliveryResult(
                success=response.status_code >= 200 and response.status_code < 300,
                status_code=response.status_code,
                response=response_data,
                delivery_id=delivery_id,
            )

        except Exception as e:
            return WebhookDeliveryResult(
                success=False,
                error=str(e),
                delivery_id=delivery_id,
            )

    async def send_batch(
        self,
        urls: List[str],
        event: Union[WebhookEvent, Dict[str, Any]],
        concurrency: int = 5,
    ) -> List[WebhookDeliveryResult]:
        """
        Send a webhook event to multiple URLs in parallel.

        Args:
            urls: List of webhook URLs to send the event to
            event: The webhook event to send
            concurrency: Maximum number of concurrent deliveries

        Returns:
            List of delivery results in the same order as URLs
        """
        tasks = []
        semaphore = asyncio.Semaphore(concurrency)

        async def _deliver(url):
            async with semaphore:
                return await self.send_webhook(url, event)

        for url in urls:
            tasks.append(_deliver(url))

        return await asyncio.gather(*tasks)

    def _generate_signature(self, payload: Dict[str, Any]) -> str:
        """
        Generate a signature for the webhook payload.

        Args:
            payload: The webhook payload to sign

        Returns:
            Base64-encoded signature
        """
        if not self.signing_secret:
            return ""

        # Convert payload to canonical string
        payload_str = json.dumps(payload, sort_keys=True)

        # Generate HMAC signature
        signature = hmac.new(
            key=self.signing_secret.encode(), msg=payload_str.encode(), digestmod=hashlib.sha256
        ).digest()

        # Return base64 encoded signature
        return base64.b64encode(signature).decode()


class WebhookHandler:
    """
    Handler for receiving and processing incoming webhooks.

    This class provides a FastAPI application that can be used
    to receive and process webhooks.
    """

    def __init__(
        self,
        path: str = "/webhook",
        secret: Optional[str] = None,
        verify_signature: bool = True,
        allowed_ips: List[str] = None,
        allowed_origins: List[str] = None,
        event_handlers: Dict[str, Callable[[WebhookEvent], Awaitable[None]]] = None,
    ):
        """
        Initialize a webhook handler.

        Args:
            path: URL path to mount the webhook handler on
            secret: Secret used to verify webhook signatures
            verify_signature: Whether to verify webhook signatures
            allowed_ips: List of allowed client IPs
            allowed_origins: List of allowed origins for CORS
            event_handlers: Dict mapping event types to handler functions
        """
        # Initialize FastAPI app
        self.app = FastAPI(
            title="Enhanced HTTPX Webhook Receiver",
            description="Webhook endpoint for receiving events",
        )

        # Configuration
        self.path = path
        self.secret = secret
        self.verify_signature = verify_signature and secret is not None
        self.allowed_ips = allowed_ips or []

        # Set up CORS
        if allowed_origins:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=allowed_origins,
                allow_methods=["POST"],
                allow_headers=["Content-Type", "X-Webhook-Signature", "X-Webhook-Delivery"],
                max_age=3600,
            )

        # Event handlers
        self.event_handlers = event_handlers or {}

        # Set up webhook endpoint
        self.app.post(
            path,
            response_model=Dict[str, str],
        )(self._handle_webhook)

        # Add health check endpoint
        self.app.get(
            "/webhook/health",
            response_model=Dict[str, str],
        )(self._health_check)

    async def _health_check(self):
        """Health check endpoint."""
        return {"status": "ok"}

    async def _verify_signature(self, payload: Dict[str, Any], signature: str) -> bool:
        """
        Verify the signature of a webhook payload.

        Args:
            payload: The webhook payload
            signature: The provided signature

        Returns:
            True if the signature is valid
        """
        if not self.secret or not signature:
            return False

        # Convert payload to canonical string
        payload_str = json.dumps(payload, sort_keys=True)

        # Generate HMAC signature
        expected_sig = hmac.new(
            key=self.secret.encode(), msg=payload_str.encode(), digestmod=hashlib.sha256
        ).digest()

        # Decode provided signature
        try:
            provided_sig = base64.b64decode(signature)
            return hmac.compare_digest(expected_sig, provided_sig)
        except:
            return False

    async def _handle_webhook(
        self,
        request: Request,
        x_webhook_signature: Optional[str] = Header(None),
        x_webhook_delivery: Optional[str] = Header(None),
    ):
        """
        Handle incoming webhook requests.

        Args:
            request: The FastAPI request object
            x_webhook_signature: The webhook signature header
            x_webhook_delivery: The webhook delivery ID header
        """
        # Check IP if restrictions are in place
        if self.allowed_ips:
            client_ip = request.client.host
            if client_ip not in self.allowed_ips:
                logger.warning(f"Webhook received from unauthorized IP: {client_ip}")
                raise HTTPException(status_code=403, detail="IP not allowed")

        # Parse request body
        try:
            payload = await request.json()
        except:
            logger.warning("Webhook received with invalid JSON payload")
            raise HTTPException(status_code=400, detail="Invalid JSON payload")

        # Verify signature if enabled
        if self.verify_signature:
            if not x_webhook_signature:
                logger.warning("Webhook received without signature")
                raise HTTPException(status_code=401, detail="Missing signature")

            signature_valid = await self._verify_signature(payload, x_webhook_signature)
            if not signature_valid:
                logger.warning("Webhook received with invalid signature")
                raise HTTPException(status_code=401, detail="Invalid signature")

        # Convert to event model
        try:
            event = WebhookEvent.model_validate(payload)
        except Exception as e:
            logger.warning(f"Webhook received with invalid event format: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid event format")

        # Process event
        event_type = event.event_type
        handler = self.event_handlers.get(event_type)

        if not handler:
            # Default handler if specific one not found
            handler = self.event_handlers.get("*")

        if handler:
            # Process event asynchronously
            try:
                asyncio.create_task(handler(event))
            except Exception as e:
                logger.error(f"Error processing webhook: {str(e)}")

        # Return success even if no handler found, to confirm receipt
        return {"status": "received", "event_id": event.id}

    def add_event_handler(
        self,
        event_type: str,
        handler: Callable[[WebhookEvent], Awaitable[None]],
    ):
        """
        Add a handler for a specific event type.

        Args:
            event_type: The event type to handle, or "*" for all events
            handler: Async function that handles the event
        """
        self.event_handlers[event_type] = handler

    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        log_level: str = "info",
        reload: bool = False,
    ):
        """
        Run the webhook server.

        Args:
            host: Host to listen on
            port: Port to listen on
            log_level: Logging level
            reload: Whether to enable auto-reload
        """
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level=log_level,
            reload=reload,
        )
