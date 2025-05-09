# tests/conftest.py

import os
import uuid
from typing import Dict, Optional

import orjson
import pytest
import respx
from enhanced_httpx import EnhancedClient
from httpx import Response

# Set timezone for consistent test results
os.environ["TZ"] = "UTC"

# Sample test data
MOCK_JSON_RESPONSE = {
    "data": {
        "id": 1,
        "name": "Test User",
        "email": "test@example.com",
        "nested": {"key": "value", "list": [1, 2, 3, 4, 5]},
    },
    "meta": {"total": 10, "page": 1},
}

MOCK_TEXT_RESPONSE = "This is a sample text response."


# Fixtures for the client
@pytest.fixture(scope="session")
def http_client():
    """Fixture for creating a shared EnhancedClient instance."""
    client = EnhancedClient()
    yield client
    client.close()


@pytest.fixture
def sample_request_data():
    """Fixture for providing sample request data."""
    return {
        "url": "https://httpbin.org/post",
        "headers": {"Content-Type": "application/json"},
        "cookies": {"session_id": "123456"},
        "body": {"key": "value"},
    }


@pytest.fixture
def sample_response_data():
    """Fixture for providing sample response data."""
    return {"json": {"key": "value"}, "status_code": 200}


@pytest.fixture
def base_url():
    """Base URL for testing."""
    return "https://api.example.com"


@pytest.fixture
def mock_router():
    """Create a mock router for httpx testing."""
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
async def client():
    """Create a client for testing."""
    client = EnhancedClient(
        base_url="https://api.example.com",
        timeout=5.0,
        max_retries=2,
        retry_backoff=0.01,  # Small backoff for faster tests
    )
    yield client
    await client.close()


@pytest.fixture
def request_id():
    """Generate a unique request ID."""
    return str(uuid.uuid4())


@pytest.fixture
def mock_auth_headers(request_id):
    """Generate mock authentication headers."""
    return {
        "Authorization": "Bearer mock-token",
        "X-Request-ID": request_id,
        "User-Agent": "EnhancedHTTPX/Test",
    }


@pytest.fixture
def mock_json_response():
    """Return mock JSON response data."""
    return MOCK_JSON_RESPONSE.copy()


@pytest.fixture
def mock_text_response():
    """Return mock text response data."""
    return MOCK_TEXT_RESPONSE


# Response fixture helpers
def create_mock_response(
    status_code: int = 200,
    content: bytes = None,
    headers: Dict[str, str] = None,
    url: str = "https://api.example.com/test",
):
    """Create a mock httpx response for testing."""
    headers = headers or {"content-type": "application/json"}
    content = content or orjson.dumps(MOCK_JSON_RESPONSE)

    return Response(
        status_code=status_code,
        content=content,
        headers=headers,
        request=None,
        extensions={},
    )


@pytest.fixture
def mock_200_json_response():
    """Mock 200 OK response with JSON content."""
    return create_mock_response(
        status_code=200,
        headers={"content-type": "application/json"},
        content=orjson.dumps(MOCK_JSON_RESPONSE),
    )


@pytest.fixture
def mock_200_text_response():
    """Mock 200 OK response with text content."""
    return create_mock_response(
        status_code=200,
        headers={"content-type": "text/plain"},
        content=MOCK_TEXT_RESPONSE.encode("utf-8"),
    )


@pytest.fixture
def mock_404_response():
    """Mock 404 Not Found response."""
    error_content = {"error": "Resource not found", "code": "not_found"}
    return create_mock_response(
        status_code=404,
        headers={"content-type": "application/json"},
        content=orjson.dumps(error_content),
    )


@pytest.fixture
def mock_500_response():
    """Mock 500 Server Error response."""
    error_content = {"error": "Internal server error", "code": "server_error"}
    return create_mock_response(
        status_code=500,
        headers={"content-type": "application/json"},
        content=orjson.dumps(error_content),
    )


# Helper function to register mock routes
def register_mock_endpoints(router, base_url: str = "https://api.example.com"):
    """Register common mock endpoints for testing."""
    # GET endpoints
    router.get(f"{base_url}/api/users").respond(
        status_code=200, json={"data": [{"id": 1, "name": "User 1"}, {"id": 2, "name": "User 2"}]}
    )

    router.get(f"{base_url}/api/users/1").respond(
        status_code=200, json={"data": {"id": 1, "name": "Test User", "email": "test@example.com"}}
    )

    # POST endpoints
    router.post(f"{base_url}/api/users").respond(
        status_code=201, json={"data": {"id": 3, "name": "New User", "email": "new@example.com"}}
    )

    # PUT endpoints
    router.put(f"{base_url}/api/users/1").respond(
        status_code=200,
        json={"data": {"id": 1, "name": "Updated User", "email": "test@example.com"}},
    )

    # DELETE endpoints
    router.delete(f"{base_url}/api/users/1").respond(status_code=204)

    # Error endpoints
    router.get(f"{base_url}/api/not-found").respond(status_code=404, json={"error": "Not found"})
    router.get(f"{base_url}/api/server-error").respond(
        status_code=500, json={"error": "Server error"}
    )
    router.get(f"{base_url}/api/timeout").respond(status_code=200, content=b"OK", after_timeout=2)

    # Redirects
    router.get(f"{base_url}/api/redirect").respond(
        status_code=302, headers={"location": f"{base_url}/api/users"}
    )

    return router


@pytest.fixture
def mock_api(mock_router, base_url):
    """Register mock API endpoints and return the router."""
    return register_mock_endpoints(mock_router, base_url)


class TestModel:
    """Sample model for response validation tests."""

    id: int
    name: str
    email: Optional[str] = None


@pytest.fixture
def test_model():
    """Return the test model class."""
    return TestModel
