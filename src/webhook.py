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
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

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
        default_headers: Dict[str, str] | None = None,
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
        headers: Dict[str, str] | None = None,
        timeout: float | None = None,
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
