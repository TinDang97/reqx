"""
Tests for the transport layer implementations.
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock

from src.transport import AiohttpTransport, HttpxTransport, HybridTransport, TransportMetrics


# Helper for mocking responses
class MockResponse:
    def __init__(self, status_code=200, content=b"", headers=None):
        self.status_code = status_code
        self._content = content
        self.headers = headers or {}

    @property
    def content(self):
        return self._content

    async def read(self):
        return self._content


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx client for tests."""
    with patch("httpx.AsyncClient") as mock:
        mock_instance = Mock()
        mock_instance.request = AsyncMock()
        mock_instance.aclose = AsyncMock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_aiohttp_session():
    """Create a mock aiohttp ClientSession for tests."""
    with patch("aiohttp.ClientSession") as mock:
        mock_instance = Mock()
        mock_instance.request = AsyncMock()
        mock_instance.close = AsyncMock()
        mock_instance.closed = False

        # Mock the context manager
        mock_resp = AsyncMock()
        mock_resp.__aenter__.return_value = MockResponse(
            200, b'{"key": "value"}', {"Content-Type": "application/json"}
        )
        mock_resp.__aexit__.return_value = None
        mock_instance.request.return_value = mock_resp

        mock.return_value = mock_instance
        yield mock_instance


class TestHttpxTransport:
    """Tests for HttpxTransport."""

    async def test_initialization(self, mock_httpx_client):
        """Test that HttpxTransport initializes correctly."""
        transport = HttpxTransport(
            base_url="https://example.com", headers={"User-Agent": "Test"}, http2=True
        )

        assert transport.supports_http2() == True
        assert transport.supports_http3() == False

    async def test_request(self, mock_httpx_client):
        """Test making a request with HttpxTransport."""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'{"success": true}'
        mock_httpx_client.request.return_value = mock_response

        # Create transport and make request
        transport = HttpxTransport()
        response = await transport.request("GET", "https://example.com")

        # Verify request was made with correct parameters
        mock_httpx_client.request.assert_called_once_with(method="GET", url="https://example.com")

        # Verify response is passed through correctly
        assert response.status_code == 200

    async def test_close(self, mock_httpx_client):
        """Test closing the HttpxTransport."""
        transport = HttpxTransport()
        await transport.close()
        mock_httpx_client.aclose.assert_called_once()


class TestAiohttpTransport:
    """Tests for AiohttpTransport."""

    async def test_initialization(self, mock_aiohttp_session):
        """Test that AiohttpTransport initializes correctly."""
        transport = AiohttpTransport(base_url="https://example.com", headers={"User-Agent": "Test"})

        assert transport.supports_http2() == False
        assert transport.supports_http3() == False

    async def test_request(self, mock_aiohttp_session):
        """Test making a request with AiohttpTransport."""
        # Create transport and make request
        transport = AiohttpTransport()
        response = await transport.request("GET", "https://example.com")

        # Verify response conversion to httpx format
        assert response.status_code == 200
        assert response.content == b'{"key": "value"}'

    async def test_close(self, mock_aiohttp_session):
        """Test closing the AiohttpTransport."""
        transport = AiohttpTransport()
        await transport.close()
        mock_aiohttp_session.close.assert_called_once()

    async def test_convert_kwargs(self):
        """Test conversion of kwargs from httpx format to aiohttp format."""
        transport = AiohttpTransport()

        # Test basic parameters
        httpx_kwargs = {
            "params": {"q": "test"},
            "headers": {"User-Agent": "Test"},
            "cookies": {"session": "abc123"},
            "timeout": 5.0,
        }

        aiohttp_kwargs = transport._convert_kwargs_to_aiohttp(httpx_kwargs)

        assert aiohttp_kwargs["params"] == {"q": "test"}
        assert aiohttp_kwargs["headers"] == {"User-Agent": "Test"}
        assert aiohttp_kwargs["cookies"] == {"session": "abc123"}

        # Verify timeout conversion
        assert hasattr(aiohttp_kwargs["timeout"], "total")


@pytest.mark.asyncio
class TestHybridTransport:
    """Tests for HybridTransport."""

    async def test_initialization(self, mock_httpx_client, mock_aiohttp_session):
        """Test that HybridTransport initializes with both transport types."""
        transport = HybridTransport(
            base_url="https://example.com", headers={"User-Agent": "Test"}, http2=True
        )

        assert hasattr(transport, "httpx")
        assert hasattr(transport, "aiohttp")
        assert transport.supports_http2() == True

    async def test_request_with_http2(self, mock_httpx_client, mock_aiohttp_session):
        """Test requests with HTTP/2 enabled use httpx."""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_httpx_client.request.return_value = mock_response

        # Create transport with HTTP/2 enabled
        transport = HybridTransport(http2=True)

        # Make request and verify httpx was used
        await transport.request("GET", "https://example.com")
        mock_httpx_client.request.assert_called_once()
        mock_aiohttp_session.request.assert_not_called()

    async def test_request_selection_by_host(self, mock_httpx_client, mock_aiohttp_session):
        """Test host-based protocol selection."""
        # Create transport with HTTP/2 disabled (so it will use host-based selection)
        transport = HybridTransport(http2=False)

        # Make requests to different hosts
        await transport.request(
            "GET", "https://www.google.com"
        )  # Should use httpx for known HTTP/2 host
        await transport.request("GET", "https://example.com")  # Should use aiohttp by default

        # Both clients should have been called once
        assert mock_httpx_client.request.call_count == 1
        assert mock_aiohttp_session.request.call_count == 1

    async def test_force_http2_parameter(self, mock_httpx_client, mock_aiohttp_session):
        """Test forcing HTTP/2 for a specific request."""
        # Create transport with HTTP/2 disabled by default
        transport = HybridTransport(http2=False)

        # Make request with force_http2=True
        await transport.request("GET", "https://example.com", force_http2=True)

        # Verify httpx was used despite HTTP/2 being disabled by default
        mock_httpx_client.request.assert_called_once()
        mock_aiohttp_session.request.assert_not_called()

    async def test_close_closes_both_transports(self, mock_httpx_client, mock_aiohttp_session):
        """Test that closing HybridTransport closes both underlying transports."""
        transport = HybridTransport()
        await transport.close()

        # Both transports should be closed
        mock_httpx_client.aclose.assert_called_once()
        mock_aiohttp_session.close.assert_called_once()

    async def test_metrics_collection(self, mock_httpx_client, mock_aiohttp_session):
        """Test that metrics are collected during requests."""
        transport = HybridTransport(enable_metrics=True)

        # Make requests to collect metrics
        await transport.request("GET", "https://www.google.com")
        await transport.request("GET", "https://example.com")

        # Get metrics summary
        metrics = transport.get_metrics_summary()

        # Verify we have metrics for both transports
        assert "transports" in metrics
        assert "hosts_analyzed" in metrics


class TestTransportMetrics:
    """Tests for TransportMetrics."""

    def test_record_request(self):
        """Test recording request metrics."""
        metrics = TransportMetrics()

        # Record some test data
        metrics.record_request("httpx", "example.com", 0.1, 200)
        metrics.record_request("httpx", "example.com", 0.2, 200)
        metrics.record_request("aiohttp", "google.com", 0.15, 200)
        metrics.record_request("aiohttp", "example.com", 0.3, 404, error=True)

        # Check global metrics
        assert metrics.metrics["httpx"]["requests"] == 2
        assert metrics.metrics["aiohttp"]["requests"] == 2
        assert metrics.metrics["aiohttp"]["errors"] == 1

        # Check host-specific metrics
        assert "example.com" in metrics.host_performance
        assert "google.com" in metrics.host_performance
        assert metrics.host_performance["example.com"]["httpx"]["count"] == 2
        assert metrics.host_performance["example.com"]["aiohttp"]["count"] == 1
        assert metrics.host_performance["example.com"]["aiohttp"]["errors"] == 1

    def test_get_preferred_transport_insufficient_data(self):
        """Test that preferred transport returns None with insufficient data."""
        metrics = TransportMetrics()

        # Add just a couple of requests (below threshold)
        metrics.record_request("httpx", "example.com", 0.1, 200)
        metrics.record_request("aiohttp", "example.com", 0.2, 200)

        # Should return None due to insufficient data
        assert metrics.get_preferred_transport("example.com") is None

    def test_get_preferred_transport_based_on_speed(self):
        """Test that preferred transport is selected based on speed."""
        metrics = TransportMetrics()

        # Add enough requests to make a decision (httpx is faster)
        for _ in range(5):
            metrics.record_request("httpx", "example.com", 0.1, 200)
            metrics.record_request("aiohttp", "example.com", 0.2, 200)

        # Should prefer httpx based on speed
        assert metrics.get_preferred_transport("example.com") == "httpx"

    def test_get_preferred_transport_considers_errors(self):
        """Test that preferred transport considers error rates."""
        metrics = TransportMetrics()

        # Add enough requests with errors to influence decision
        for _ in range(5):
            # httpx is faster but has errors
            metrics.record_request("httpx", "example.com", 0.1, 200)
            metrics.record_request("httpx", "example.com", 0.1, 0, error=True)

            # aiohttp is slower but reliable
            metrics.record_request("aiohttp", "example.com", 0.15, 200)

        # Should prefer aiohttp due to better reliability despite being slower
        assert metrics.get_preferred_transport("example.com") == "aiohttp"

    def test_get_summary(self):
        """Test generating summary metrics."""
        metrics = TransportMetrics()

        # Add some test data
        metrics.record_request("httpx", "example.com", 0.1, 200)
        metrics.record_request("httpx", "google.com", 0.2, 200)
        metrics.record_request("aiohttp", "example.com", 0.15, 200)

        # Get summary
        summary = metrics.get_summary()

        # Check summary structure
        assert "transports" in summary
        assert "hosts_analyzed" in summary
        assert summary["hosts_analyzed"] == 2
        assert "httpx" in summary["transports"]
        assert "aiohttp" in summary["transports"]
