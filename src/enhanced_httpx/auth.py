"""Authentication providers for EnhancedClient.

This module provides a structured approach to API authentication with various
authentication schemes including OAuth, JWT, API keys, and more.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, Awaitable, Union, ClassVar
import time
import asyncio
import logging
from datetime import datetime, timedelta
from urllib.parse import urljoin

import httpx
from pydantic import BaseModel, Field

from .exceptions import AuthenticationError

logger = logging.getLogger("enhanced_httpx.auth")


class AuthProvider(ABC):
    """Base class for all authentication providers."""

    @abstractmethod
    async def authenticate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate a request by modifying it (e.g. adding headers).

        Args:
            request: The request dictionary containing method, url, headers, etc.

        Returns:
            Modified request dictionary with authentication details.
        """
        pass

    @abstractmethod
    async def refresh(self) -> None:
        """Refresh authentication credentials if needed."""
        pass


class ApiKeyAuth(AuthProvider):
    """API Key authentication provider."""

    def __init__(
        self, api_key: str, header_name: str = "X-API-Key", query_param: Optional[str] = None
    ):
        """Initialize API Key authentication.

        Args:
            api_key: The API key value
            header_name: Name of the header to use for the API key
            query_param: If set, adds the API key as a query parameter with this name
        """
        self.api_key = api_key
        self.header_name = header_name
        self.query_param = query_param

    async def authenticate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Add API key to the request."""
        # Deep copy the request to avoid modifying the original
        request = {**request}

        # Add headers if they don't exist
        if "headers" not in request:
            request["headers"] = {}

        # Add API key as header
        request["headers"][self.header_name] = self.api_key

        # Add as query param if configured
        if self.query_param:
            if "params" not in request:
                request["params"] = {}
            request["params"][self.query_param] = self.api_key

        return request

    async def refresh(self) -> None:
        """No refresh needed for API key auth."""
        pass


class BearerTokenAuth(AuthProvider):
    """Bearer token authentication provider."""

    def __init__(self, token: str, prefix: str = "Bearer"):
        """Initialize Bearer token authentication.

        Args:
            token: The bearer token
            prefix: The prefix to use before the token (usually "Bearer")
        """
        self.token = token
        self.prefix = prefix

    async def authenticate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Add bearer token to the request."""
        request = {**request}

        if "headers" not in request:
            request["headers"] = {}

        request["headers"]["Authorization"] = f"{self.prefix} {self.token}"

        return request

    async def refresh(self) -> None:
        """No refresh needed for simple bearer token auth."""
        pass


class TokenModel(BaseModel):
    """Model for OAuth token responses."""

    access_token: str
    token_type: str = "Bearer"
    expires_in: Optional[int] = None
    refresh_token: Optional[str] = None
    scope: Optional[str] = None
    id_token: Optional[str] = None


class OAuth2Auth(AuthProvider):
    """OAuth 2.0 authentication provider with automatic token refresh."""

    def __init__(
        self,
        token_url: str,
        client_id: str,
        client_secret: str,
        scope: str = "",
        grant_type: str = "client_credentials",
        refresh_before_expiry: int = 60,  # seconds
        auth_request_hook: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        token_model: type = TokenModel,
    ):
        """Initialize OAuth 2.0 authentication.

        Args:
            token_url: URL to request access tokens
            client_id: OAuth client ID
            client_secret: OAuth client secret
            scope: OAuth scope(s)
            grant_type: OAuth grant type
            refresh_before_expiry: Seconds before expiry to refresh token
            auth_request_hook: Function to modify the token request
            token_model: Pydantic model for token response parsing
        """
        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.scope = scope
        self.grant_type = grant_type
        self.refresh_before_expiry = refresh_before_expiry
        self.auth_request_hook = auth_request_hook
        self.token_model = token_model

        self._token: Optional[token_model] = None
        self._token_expiry: Optional[datetime] = None
        self._lock = asyncio.Lock()

    async def authenticate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Add OAuth access token to the request."""
        await self._ensure_valid_token()

        request = {**request}

        if "headers" not in request:
            request["headers"] = {}

        if self._token:
            request["headers"][
                "Authorization"
            ] = f"{self._token.token_type} {self._token.access_token}"

        return request

    async def refresh(self) -> None:
        """Force a token refresh."""
        async with self._lock:
            await self._fetch_token()

    async def _ensure_valid_token(self) -> None:
        """Ensure we have a valid token, refreshing if necessary."""
        # If we don't have a token or it's close to expiry, refresh
        if (
            self._token is None
            or self._token_expiry is None
            or datetime.now() + timedelta(seconds=self.refresh_before_expiry) >= self._token_expiry
        ):
            await self.refresh()

    async def _fetch_token(self) -> None:
        """Fetch a new access token."""
        token_request = {
            "grant_type": self.grant_type,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        if self.scope:
            token_request["scope"] = self.scope

        # For refresh token grant
        if self.grant_type == "refresh_token" and self._token and self._token.refresh_token:
            token_request["refresh_token"] = self._token.refresh_token

        # Apply custom hook if provided
        if self.auth_request_hook:
            token_request = self.auth_request_hook(token_request)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.token_url,
                    data=token_request,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                response.raise_for_status()

                # Parse the token response
                token_data = response.json()
                self._token = self.token_model.model_validate(token_data)

                # Set token expiry
                if self._token.expires_in:
                    self._token_expiry = datetime.now() + timedelta(seconds=self._token.expires_in)
                else:
                    # Default to 1 hour if no expiry provided
                    self._token_expiry = datetime.now() + timedelta(hours=1)

                logger.debug(f"OAuth token refreshed, expires at {self._token_expiry}")

        except Exception as e:
            logger.error(f"Failed to fetch OAuth token: {str(e)}")
            raise AuthenticationError(f"OAuth authentication failed: {str(e)}")


class JWTAuth(AuthProvider):
    """JWT authentication provider with automatic signing."""

    def __init__(
        self,
        private_key: str,
        algorithm: str = "HS256",
        key_id: Optional[str] = None,
        header_name: str = "Authorization",
        header_prefix: str = "Bearer",
        claims: Optional[Dict[str, Any]] = None,
        expiry: int = 3600,  # 1 hour
        audience: Optional[str] = None,
        issuer: Optional[str] = None,
    ):
        """Initialize JWT authentication.

        Args:
            private_key: Private key for JWT signing
            algorithm: JWT signing algorithm
            key_id: Key ID for JWK identification
            header_name: Header name for the JWT
            header_prefix: Prefix for the JWT in the header
            claims: Additional JWT claims
            expiry: JWT expiry time in seconds
            audience: JWT audience claim
            issuer: JWT issuer claim
        """
        try:
            import jwt
        except ImportError:
            raise ImportError("JWT authentication requires PyJWT. Install with 'pip install pyjwt'")

        self.private_key = private_key
        self.algorithm = algorithm
        self.key_id = key_id
        self.header_name = header_name
        self.header_prefix = header_prefix
        self.claims = claims or {}
        self.expiry = expiry
        self.audience = audience
        self.issuer = issuer

        # Keep a cached token
        self._cached_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

    async def authenticate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Add JWT token to the request."""
        # Ensure we have a valid token
        if not self._cached_token or (self._token_expiry and datetime.now() >= self._token_expiry):
            await self.refresh()

        request = {**request}

        if "headers" not in request:
            request["headers"] = {}

        request["headers"][self.header_name] = f"{self.header_prefix} {self._cached_token}"

        return request

    async def refresh(self) -> None:
        """Generate a new JWT token."""
        try:
            import jwt

            # Set standard claims
            now = int(time.time())
            payload = {"iat": now, "exp": now + self.expiry, **self.claims}

            if self.audience:
                payload["aud"] = self.audience

            if self.issuer:
                payload["iss"] = self.issuer

            # Set headers if key_id is provided
            headers = {}
            if self.key_id:
                headers["kid"] = self.key_id

            # Generate the token
            self._cached_token = jwt.encode(
                payload,
                self.private_key,
                algorithm=self.algorithm,
                headers=headers if headers else None,
            )

            # Set expiry time
            self._token_expiry = datetime.now() + timedelta(
                seconds=self.expiry - 60
            )  # Refresh 1 min before expiry

        except Exception as e:
            logger.error(f"Failed to generate JWT: {str(e)}")
            raise AuthenticationError(f"JWT authentication failed: {str(e)}")
