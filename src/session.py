"""
Enhanced Session module for maintaining state across HTTP requests.

This module provides a Session class that can be used to maintain cookies,
authentication, and other state across multiple requests. It builds upon the
ReqxClient for making the actual requests.
"""

import json
import logging
import os
import time
from typing import Any, Dict, TypeVar

import httpx

from .client import ReqxClient
from .exceptions import AuthenticationError

logger = logging.getLogger("reqx.session")

T = TypeVar("T")


class Session:
    """
    A session object for maintaining state across requests.

    The Session maintains cookies, default headers, and can handle authentication
    flows automatically.
    """

    def __init__(
        self,
        base_url: str = "",
        headers: Dict[str, str] | None = None,
        cookies: Dict[str, str] | None = None,
        auth: Dict[str, Any] | None = None,
        persist: bool = False,
        session_file: str | None = None,
        **client_kwargs,
    ):
        """
        Initialize a new session.

        Args:
            base_url: Base URL for all requests in this session
            headers: Default headers for all requests
            cookies: Default cookies for all requests
            auth: Authentication configuration (see below for options)
            persist: Whether to persist session data to disk
            session_file: Path to file for storing session data
            **client_kwargs: Additional arguments to pass to ReqxClient constructor

        Authentication options:
            - Basic auth: {"type": "basic", "username": "user", "password": "pass"}
            - Bearer token: {"type": "bearer", "token": "your-token"}
            - API key: {"type": "apikey", "key": "X-API-Key", "value": "your-key"}
            - OAuth2:
                {"type": "oauth2", "token_url": "url", "client_id": "id", "client_secret": "secret"}
        """
        self.base_url = base_url
        self.headers = headers or {}
        self.cookies = cookies or {}
        self.auth_config = auth or {}
        self.persist = persist

        # Session state
        self.authenticated = False
        self.auth_token = None
        self.auth_expiry = None

        # File for persisting session data if enabled
        if persist:
            self.session_file = session_file or os.path.expanduser("~/.enhanced-httpx-session.json")
            self._load_session()
        else:
            self.session_file = None

        # Create client with session defaults
        self.client = ReqxClient(
            base_url=self.base_url,
            headers=self.headers,
            cookies=self.cookies,
            **client_kwargs,
        )

        # Apply authentication if configured
        if self.auth_config:
            # Setup authentication but don't await it yet
            self._setup_auth_flag = True

    def _setup_authentication(self):
        """Set up authentication based on the current configuration."""
        import asyncio

        # Create a new event loop for running the authentication
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.authenticate())
        finally:
            loop.close()

    async def __aenter__(self):
        """Support for async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting context."""
        if self.persist:
            self._save_session()
        await self.close()

    async def close(self):
        """Close the underlying client and save session if persistent."""
        if self.persist:
            self._save_session()
        await self.client.close()

    async def authenticate(self, force: bool = False):
        """
        Perform authentication based on the auth configuration.

        Args:
            force: Force re-authentication even if already authenticated

        Returns:
            Success status of authentication
        """
        # Skip if already authenticated and not forced
        if self.authenticated and not force:
            # Check if token is expired
            if self.auth_expiry is not None and time.time() > self.auth_expiry:
                logger.info("Authentication token has expired, re-authenticating")
            else:
                return True

        if not self.auth_config:
            logger.warning("No authentication configuration provided")
            return False

        auth_type = self.auth_config.get("type", "").lower()

        if auth_type == "basic":
            # Basic auth is handled directly by the client
            username = self.auth_config.get("username")
            password = self.auth_config.get("password")
            if not username or not password:
                raise AuthenticationError("Username and password are required for basic auth")

            self.client.default_headers["Authorization"] = httpx.BasicAuth(
                username, password
            )._auth_header
            self.authenticated = True

        elif auth_type == "bearer":
            # Bearer token auth
            token = self.auth_config.get("token")
            if not token:
                raise AuthenticationError("Token is required for bearer auth")

            self.client.default_headers["Authorization"] = f"Bearer {token}"
            self.authenticated = True

        elif auth_type == "apikey":
            # API key auth (in header or query param)
            key = self.auth_config.get("key")
            value = self.auth_config.get("value")
            in_header = self.auth_config.get("in", "header") == "header"

            if not key or not value:
                raise AuthenticationError("Key and value are required for API key auth")

            if in_header:
                self.client.default_headers[key] = value
            else:
                # For query params, we'll add it to all requests
                self.client.add_request_middleware(self._create_api_key_middleware(key, value))
            self.authenticated = True

        elif auth_type == "oauth2":
            # OAuth2 flow
            await self._perform_oauth2_flow(force)

        else:
            raise AuthenticationError(f"Unsupported authentication type: {auth_type}")

        # Save session after authentication if persistence is enabled
        if self.persist:
            self._save_session()

        return self.authenticated

    async def _perform_oauth2_flow(self, force: bool = False):
        """
        Perform OAuth2 authentication flow.

        Args:
            force: Force re-authentication even if token is still valid

        Raises:
            AuthenticationError: If authentication fails
        """
        # Check if we have a valid token already
        if self.auth_token and self.auth_expiry and time.time() < self.auth_expiry and not force:
            logger.debug("Using existing valid OAuth2 token")
            return True

        token_url = self.auth_config.get("token_url")
        client_id = self.auth_config.get("client_id")
        client_secret = self.auth_config.get("client_secret")
        scope = self.auth_config.get("scope", "")

        if not token_url or not client_id:
            raise AuthenticationError(
                "token_url and client_id are required for OAuth2 authentication"
            )

        # Grant type determines the flow (client_credentials is most common for API access)
        grant_type = self.auth_config.get("grant_type", "client_credentials")

        # Prepare token request data
        data = {
            "grant_type": grant_type,
            "client_id": client_id,
            "scope": scope,
        }

        # Add client secret if provided (some APIs require it)
        if client_secret:
            data["client_secret"] = client_secret

        # For authorization_code flow
        if grant_type == "authorization_code":
            # This would typically happen after a user authorization step
            code = self.auth_config.get("code")
            redirect_uri = self.auth_config.get("redirect_uri")

            if not code:
                raise AuthenticationError(
                    "Authorization code is required for authorization_code flow"
                )

            data["code"] = code
            if redirect_uri:
                data["redirect_uri"] = redirect_uri

        # For password flow
        elif grant_type == "password":
            username = self.auth_config.get("username")
            password = self.auth_config.get("password")

            if not username or not password:
                raise AuthenticationError("Username and password are required for password flow")

            data["username"] = username
            data["password"] = password

        # For refresh_token flow
        elif grant_type == "refresh_token":
            refresh_token = self.auth_config.get("refresh_token") or self.auth_config.get("refresh")

            if not refresh_token:
                raise AuthenticationError("Refresh token is required for refresh_token flow")

            data["refresh_token"] = refresh_token

        # Make token request
        try:
            # Use a temporary client to avoid auth loop
            headers = {"Content-Type": "application/x-www-form-urlencoded"}

            # Some APIs require basic auth for client credentials
            if client_id and client_secret and self.auth_config.get("use_basic_for_token", False):
                auth = httpx.BasicAuth(client_id, client_secret)
                # Remove from form data
                if "client_id" in data:
                    del data["client_id"]
                if "client_secret" in data:
                    del data["client_secret"]
            else:
                auth = None

            # Make the token request
            response = await self.client.post(
                token_url,
                data=data,
                headers=headers,
                auth=auth,
            )

            token_data = response.json()

            # Extract token and expiry
            self.auth_token = token_data.get("access_token")
            expires_in = token_data.get("expires_in")

            if not self.auth_token:
                raise AuthenticationError(f"No access token in response: {token_data}")

            # Calculate expiry time (with a small buffer to be safe)
            if expires_in:
                self.auth_expiry = time.time() + float(expires_in) - 60  # 60s buffer
            else:
                # Default to 1 hour if no expiry specified
                self.auth_expiry = time.time() + 3600

            # Store refresh token if present
            if "refresh_token" in token_data:
                self.auth_config["refresh_token"] = token_data["refresh_token"]

            # Set authorization header
            self.client.default_headers["Authorization"] = f"Bearer {self.auth_token}"
            self.authenticated = True

            logger.info("OAuth2 authentication successful")

        except Exception as e:
            self.authenticated = False
            raise AuthenticationError(f"OAuth2 authentication failed: {str(e)}") from e

    def _create_api_key_middleware(self, key: str, value: str):
        """Create a middleware that adds API key to query params."""

        async def _api_key_middleware(method: str, url: str, request_kwargs: Dict[str, Any]):
            if "params" not in request_kwargs:
                request_kwargs["params"] = {}

            if isinstance(request_kwargs["params"], dict):
                request_kwargs["params"][key] = value

            return request_kwargs

        return _api_key_middleware

    def _load_session(self):
        """Load session data from disk if available."""
        if self.session_file is None or not os.path.exists(self.session_file):
            return

        try:
            with open(self.session_file, "r") as f:
                data = json.load(f)

            # Restore session state
            self.cookies = data.get("cookies", {})
            self.headers = data.get("headers", {})
            self.auth_token = data.get("auth_token")
            self.auth_expiry = data.get("auth_expiry")
            self.authenticated = data.get("authenticated", False)

            # If there was a refresh token, restore it
            if "auth_config" in data and "refresh_token" in data["auth_config"]:
                if not self.auth_config:
                    self.auth_config = {}
                self.auth_config["refresh_token"] = data["auth_config"]["refresh_token"]

            logger.debug(f"Session loaded from {self.session_file}")
        except Exception as e:
            logger.warning(f"Failed to load session: {e}")

    def _save_session(self):
        """Save session data to disk."""
        if not self.session_file:
            return

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(self.session_file)), exist_ok=True)

        data = {
            "cookies": self.cookies,
            "headers": self.headers,
            "auth_token": self.auth_token,
            "auth_expiry": self.auth_expiry,
            "authenticated": self.authenticated,
        }

        # Save auth config with sensitive items filtered out
        if self.auth_config:
            auth_config = {k: v for k, v in self.auth_config.items()}
            # Don't save secrets except for refresh token which is needed
            for key in ["client_secret", "password"]:
                if key in auth_config:
                    del auth_config[key]
            data["auth_config"] = auth_config

        try:
            with open(self.session_file, "w") as f:
                json.dump(data, f)
            logger.debug(f"Session saved to {self.session_file}")
        except Exception as e:
            logger.warning(f"Failed to save session: {e}")

    # Forward request methods to client
    async def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Send a request using the session's client."""
        # Check if we need to refresh authentication
        if self.authenticated and self.auth_expiry and time.time() > self.auth_expiry:
            logger.debug("Authentication token expired, refreshing")
            await self.authenticate()

        response = await self.client.request(method, url, **kwargs)
        if response is None:
            raise ValueError(f"Request to {url} failed to return a response")
        return response

    async def get(self, url: str, **kwargs) -> httpx.Response:
        """Send a GET request using the session's client."""
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> httpx.Response:
        """Send a POST request using the session's client."""
        return await self.request("POST", url, **kwargs)

    async def put(self, url: str, **kwargs) -> httpx.Response:
        """Send a PUT request using the session's client."""
        return await self.request("PUT", url, **kwargs)

    async def patch(self, url: str, **kwargs) -> httpx.Response:
        """Send a PATCH request using the session's client."""
        return await self.request("PATCH", url, **kwargs)

    async def delete(self, url: str, **kwargs) -> httpx.Response:
        """Send a DELETE request using the session's client."""
        return await self.request("DELETE", url, **kwargs)

    async def head(self, url: str, **kwargs) -> httpx.Response:
        """Send a HEAD request using the session's client."""
        return await self.request("HEAD", url, **kwargs)

    async def options(self, url: str, **kwargs) -> httpx.Response:
        """Send an OPTIONS request using the session's client."""
        return await self.request("OPTIONS", url, **kwargs)

    def update_headers(self, headers: Dict[str, str]):
        """Update session headers."""
        self.headers.update(headers)
        self.client.default_headers.update(headers)

    def update_cookies(self, cookies: Dict[str, str]):
        """Update session cookies."""
        self.cookies.update(cookies)
        # Note: Client doesn't have a direct way to update cookies
        # They'll be sent automatically on next request

    def update_auth_config(self, auth_config: Dict[str, Any]):
        """Update authentication configuration."""
        self.auth_config.update(auth_config)
        self.authenticated = False  # Force re-authentication with new config
