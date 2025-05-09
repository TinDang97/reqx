from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest
from cli import ReqxClient
from exceptions import NotFoundError, RequestError, ResponseError, ServerError
from pydantic import BaseModel


class UserModel(BaseModel):
    id: int
    name: str
    email: Optional[str] = None


class ListUsersResponse(BaseModel):
    data: List[UserModel]


class UserResponse(BaseModel):
    data: UserModel


@pytest.fixture
async def client():
    async with ReqxClient(base_url="https://api.example.com") as client:
        yield client


@pytest.mark.asyncio
async def test_client_initialization():
    """Test that client initializes with correct defaults."""
    client = ReqxClient()

    assert client.base_url == ""
    assert client.timeout == 30.0
    assert client.follow_redirects is True
    assert client.verify_ssl is True
    assert client.max_retries == 3

    await client.close()


@pytest.mark.asyncio
async def test_client_custom_parameters():
    """Test that client initializes with custom parameters."""
    client = ReqxClient(
        base_url="https://example.com",
        timeout=10.0,
        max_connections=50,
        follow_redirects=False,
        verify_ssl=False,
        max_retries=5,
        http2=True,
    )

    assert client.base_url == "https://example.com"
    assert client.timeout == 10.0
    assert client.follow_redirects is False
    assert client.verify_ssl is False
    assert client.max_retries == 5

    await client.close()


@pytest.mark.asyncio
async def test_client_context_manager():
    """Test client as a context manager."""
    async with ReqxClient() as client:
        assert client.client is not None

    # Client should be closed after the context manager exits


@pytest.mark.asyncio
async def test_get_request_success(mock_api, client):
    """Test a successful GET request."""
    response = await client.get("/api/users/1")

    assert response.status_code == 200
    assert response.json() == {"data": {"id": 1, "name": "Test User", "email": "test@example.com"}}


@pytest.mark.asyncio
async def test_get_with_response_model(mock_api, client):
    """Test parsing a GET response into a model."""
    user = await client.get("/api/users/1", response_model=UserResponse)

    assert isinstance(user, UserResponse)
    assert user.data.id == 1
    assert user.data.name == "Test User"
    assert user.data.email == "test@example.com"


@pytest.mark.asyncio
async def test_get_list_with_response_model(mock_api, client):
    """Test parsing a list response into a model."""
    users = await client.get("/api/users", response_model=ListUsersResponse)

    assert isinstance(users, ListUsersResponse)
    assert len(users.data) == 2
    assert users.data[0].id == 1
    assert users.data[0].name == "User 1"
    assert users.data[1].id == 2
    assert users.data[1].name == "User 2"


@pytest.mark.asyncio
async def test_post_request_success(mock_api, client):
    """Test a successful POST request."""
    new_user = {"name": "New User", "email": "new@example.com"}
    response = await client.post("/api/users", json=new_user)

    assert response.status_code == 201
    assert response.json() == {"data": {"id": 3, "name": "New User", "email": "new@example.com"}}


@pytest.mark.asyncio
async def test_put_request_success(mock_api, client):
    """Test a successful PUT request."""
    updated_user = {"name": "Updated User", "email": "test@example.com"}
    response = await client.put("/api/users/1", json=updated_user)

    assert response.status_code == 200
    assert response.json() == {
        "data": {"id": 1, "name": "Updated User", "email": "test@example.com"}
    }


@pytest.mark.asyncio
async def test_delete_request_success(mock_api, client):
    """Test a successful DELETE request."""
    response = await client.delete("/api/users/1")

    assert response.status_code == 204


@pytest.mark.asyncio
async def test_404_error_handling(mock_api, client):
    """Test handling of 404 errors."""
    with pytest.raises(NotFoundError):
        await client.get("/api/not-found")


@pytest.mark.asyncio
async def test_500_error_handling(mock_api, client):
    """Test handling of 500 errors."""
    with pytest.raises(ServerError):
        await client.get("/api/server-error")


@pytest.mark.asyncio
async def test_retry_on_network_error():
    """Test retry behavior on network error."""
    with patch("httpx.AsyncClient.request") as mock_request:
        # First call raises an error, second call succeeds
        mock_request.side_effect = [
            Exception("Connection error"),
            MagicMock(
                status_code=200, raise_for_status=lambda: None, json=lambda: {"success": True}
            ),
        ]

        client = ReqxClient(max_retries=1, retry_backoff=0.01)
        response = await client.get("https://example.com/api")

        assert mock_request.call_count == 2
        assert response.json() == {"success": True}

        await client.close()


@pytest.mark.asyncio
async def test_max_retries_exceeded():
    """Test that RequestError is raised when max retries are exceeded."""
    with patch("httpx.AsyncClient.request") as mock_request:
        # All calls raise network errors
        mock_request.side_effect = Exception("Connection error")

        client = ReqxClient(max_retries=2, retry_backoff=0.01)

        with pytest.raises(RequestError):
            await client.get("https://example.com/api")

        # Should have tried 3 times (initial + 2 retries)
        assert mock_request.call_count == 3

        await client.close()


@pytest.mark.asyncio
async def test_url_preparation(client):
    """Test URL preparation with base URL."""
    with patch("httpx.AsyncClient.request") as mock_request:
        mock_response = MagicMock(status_code=200, raise_for_status=lambda: None)
        mock_request.return_value = mock_response

        # Should add base URL
        await client.get("/relative-path")
        args, kwargs = mock_request.call_args
        assert kwargs["url"] == "https://api.example.com/relative-path"

        # Should not modify absolute URL
        await client.get("https://other-domain.com/path")
        args, kwargs = mock_request.call_args
        assert kwargs["url"] == "https://other-domain.com/path"


@pytest.mark.asyncio
async def test_headers_and_cookies_merging(client):
    """Test merging of headers and cookies."""
    client.default_headers = {"User-Agent": "Test", "Accept": "application/json"}
    client.default_cookies = {"default": "value"}

    with patch("httpx.AsyncClient.request") as mock_request:
        mock_response = MagicMock(status_code=200, raise_for_status=lambda: None)
        mock_request.return_value = mock_response

        # Custom headers and cookies should be merged with defaults
        await client.get(
            "/test",
            headers={"Authorization": "Bearer token", "Accept": "text/html"},
            cookies={"session": "123"},
        )

        args, kwargs = mock_request.call_args

        # Accept from default is overridden
        assert kwargs["headers"] == {
            "User-Agent": "Test",
            "Accept": "text/html",
            "Authorization": "Bearer token",
        }

        # Both cookies are included
        assert kwargs["cookies"] == {"default": "value", "session": "123"}


@pytest.mark.asyncio
async def test_response_model_validation_error(mock_api, client):
    """Test error handling with invalid response models."""

    # Define a model that won't match the response
    class InvalidModel(BaseModel):
        wrong_field: str

    with pytest.raises(ResponseError):
        await client.get("/api/users/1", response_model=InvalidModel)
