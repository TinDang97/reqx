import json
from datetime import datetime

import pytest
from utils import (
    deserialize_json,
    format_curl_command,
    select_json_path,
    serialize_json,
)

# Sample nested JSON for testing JSON path
NESTED_JSON = {
    "store": {
        "book": [
            {
                "category": "reference",
                "author": "Nigel Rees",
                "title": "Sayings of the Century",
                "price": 8.95,
            },
            {
                "category": "fiction",
                "author": "Evelyn Waugh",
                "title": "Sword of Honour",
                "price": 12.99,
            },
            {
                "category": "fiction",
                "author": "J. R. R. Tolkien",
                "title": "The Lord of the Rings",
                "isbn": "0-395-19395-8",
                "price": 22.99,
            },
        ],
        "bicycle": {"color": "red", "price": 19.95},
    },
    "expensive": 10,
}


class TestSerializeJson:
    def test_serialize_dict(self):
        """Test serializing a dictionary to JSON."""
        data = {"name": "John", "age": 30, "city": "New York"}
        result = serialize_json(data)
        # Verify it's a valid JSON string
        parsed = json.loads(result)
        assert parsed == data

    def test_serialize_list(self):
        """Test serializing a list to JSON."""
        data = [1, 2, 3, 4, 5]
        result = serialize_json(data)
        parsed = json.loads(result)
        assert parsed == data

    def test_serialize_with_default(self):
        """Test serializing with a default handler for custom types."""
        data = {"date": datetime(2023, 1, 1, 12, 0, 0)}

        def default(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not serializable")

        result = serialize_json(data, default=default)
        parsed = json.loads(result)
        assert parsed["date"] == "2023-01-01T12:00:00"


class TestDeserializeJson:
    def test_deserialize_dict(self):
        """Test deserializing a JSON string to a dictionary."""
        json_str = '{"name": "John", "age": 30, "city": "New York"}'
        result = deserialize_json(json_str)
        assert result == {"name": "John", "age": 30, "city": "New York"}

    def test_deserialize_list(self):
        """Test deserializing a JSON string to a list."""
        json_str = "[1, 2, 3, 4, 5]"
        result = deserialize_json(json_str)
        assert result == [1, 2, 3, 4, 5]

    def test_deserialize_none(self):
        """Test deserializing None."""
        result = deserialize_json(None)
        assert result is None

    def test_deserialize_empty_string(self):
        """Test deserializing an empty string."""
        result = deserialize_json("")
        assert result is None


class TestSelectJsonPath:
    def test_simple_path(self):
        """Test a simple JSONPath expression."""
        result = select_json_path(NESTED_JSON, "$.store.bicycle.color")
        assert result == "red"

    def test_array_access(self):
        """Test accessing array elements with JSONPath."""
        result = select_json_path(NESTED_JSON, "$.store.book[0].title")
        assert result == "Sayings of the Century"

    def test_array_slice(self):
        """Test accessing array slices with JSONPath."""
        result = select_json_path(NESTED_JSON, "$.store.book[0:2].title")
        assert result == ["Sayings of the Century", "Sword of Honour"]

    def test_filter_expression(self):
        """Test using filter expressions with JSONPath."""
        result = select_json_path(NESTED_JSON, "$.store.book[?(@.price > 20)].title")
        assert result == ["The Lord of the Rings"]

    def test_wildcard(self):
        """Test using wildcards with JSONPath."""
        result = select_json_path(NESTED_JSON, "$.store.book[*].author")
        assert result == ["Nigel Rees", "Evelyn Waugh", "J. R. R. Tolkien"]

    def test_invalid_path(self):
        """Test that invalid paths raise ValueError."""
        with pytest.raises(ValueError):
            select_json_path(NESTED_JSON, "$.[invalid")

    def test_path_not_found(self):
        """Test that non-matching paths return None."""
        result = select_json_path(NESTED_JSON, "$.nonexistent.path")
        assert result is None


class TestSanitizeSensitiveData:
    def test_sanitize_password(self):
        """Test sanitizing password in dictionary."""
        data = {"username": "user", "password": "secret123"}
        sanitized = sanitize_sensitive_data(data)
        assert sanitized["username"] == "user"
        assert sanitized["password"] == "********"

    def test_sanitize_token(self):
        """Test sanitizing token in dictionary."""
        data = {"auth_token": "abcdef123456", "data": "some data"}
        sanitized = sanitize_sensitive_data(data)
        assert sanitized["auth_token"] == "********"
        assert sanitized["data"] == "some data"

    def test_sanitize_nested(self):
        """Test sanitizing nested dictionary."""
        data = {"user": {"credentials": {"api_key": "secret_api_key"}}, "data": [1, 2, 3]}
        sanitized = sanitize_sensitive_data(data)
        assert sanitized["user"]["credentials"]["api_key"] == "********"
        assert sanitized["data"] == [1, 2, 3]

    def test_sanitize_in_list(self):
        """Test sanitizing sensitive data in lists."""
        data = [
            {"username": "user1", "password": "pass1"},
            {"username": "user2", "password": "pass2"},
        ]
        sanitized = sanitize_sensitive_data(data)
        assert sanitized[0]["password"] == "********"
        assert sanitized[1]["password"] == "********"
        assert sanitized[0]["username"] == "user1"
        assert sanitized[1]["username"] == "user2"


class TestFormatCurlCommand:
    def test_format_get(self):
        """Test formatting a GET request as curl."""
        curl = format_curl_command(
            "GET",
            "https://api.example.com/users",
            {"Authorization": "Bearer token", "Accept": "application/json"},
        )
        assert curl.startswith("curl -X GET")
        assert "-H 'Authorization: Bearer token'" in curl
        assert "-H 'Accept: application/json'" in curl
        assert "'https://api.example.com/users'" in curl

    def test_format_post_with_json(self):
        """Test formatting a POST request with JSON body as curl."""
        curl = format_curl_command(
            "POST",
            "https://api.example.com/users",
            {"Content-Type": "application/json"},
            {"name": "John", "email": "john@example.com"},
        )
        assert curl.startswith("curl -X POST")
        assert "-H 'Content-Type: application/json'" in curl
        assert "-d '{" in curl
        assert '"name":"John"' in curl.replace(" ", "")
        assert '"email":"john@example.com"' in curl.replace(" ", "")
        assert "'https://api.example.com/users'" in curl

    def test_format_with_params(self):
        """Test formatting a request with query parameters."""
        curl = format_curl_command(
            "GET", "https://api.example.com/search", {}, None, {"q": "python", "page": "1"}
        )
        assert (
            "https://api.example.com/search?q=python&page=1" in curl
            or "https://api.example.com/search?page=1&q=python" in curl
        )


class TestSystemOptimization:
    """Tests for system-aware optimization utilities."""

    def test_get_system_resource_metrics(self):
        """Test retrieving system resource metrics."""
        # Use actual system metrics for basic validation
        metrics = get_system_resource_metrics()

        # Basic validation of metrics structure
        assert "cpu_count" in metrics
        assert "memory_gb" in metrics
        assert "os" in metrics
        assert metrics["cpu_count"] > 0
        assert metrics["memory_gb"] > 0

    @patch("src.utils.get_system_resource_metrics")
    def test_optimal_pool_settings_high_resources(self, mock_metrics):
        """Test optimal connection pool settings with high system resources."""
        # Mock a high-resource system
        mock_metrics.return_value = {
            "cpu_count": 16,
            "cpu_count_logical": 32,
            "memory_gb": 32.0,
            "memory_available_gb": 16.0,
            "os": "Linux",
        }

        # Get pool settings
        settings = get_optimal_connection_pool_settings()

        # For a high-resource system, should be near the upper end
        assert settings["max_connections"] >= 100
        assert settings["max_keepalive_connections"] >= 30
        assert settings["keepalive_expiry"] == 60

    @patch("src.utils.get_system_resource_metrics")
    def test_optimal_pool_settings_low_resources(self, mock_metrics):
        """Test optimal connection pool settings with limited system resources."""
        # Mock a low-resource system
        mock_metrics.return_value = {
            "cpu_count": 1,
            "cpu_count_logical": 2,
            "memory_gb": 2.0,
            "memory_available_gb": 0.5,
            "os": "Linux",
        }

        # Get pool settings
        settings = get_optimal_connection_pool_settings()

        # For a low-resource system, should be conservative
        assert settings["max_connections"] <= 20
        assert settings["keepalive_expiry"] == 30  # Shorter expiry for low memory

    @patch("src.utils.get_system_resource_metrics")
    def test_optimal_pool_settings_medium_resources(self, mock_metrics):
        """Test optimal connection pool settings with moderate system resources."""
        # Mock a medium-resource system
        mock_metrics.return_value = {
            "cpu_count": 4,
            "cpu_count_logical": 8,
            "memory_gb": 8.0,
            "memory_available_gb": 4.0,
            "os": "Windows",
        }

        # Get pool settings
        settings = get_optimal_connection_pool_settings()

        # For a medium system, should be in the middle range
        assert 20 <= settings["max_connections"] <= 100
        assert settings["keepalive_expiry"] == 60  # Normal expiry for sufficient memory


class TestHttpHelpers:
    """Tests for HTTP-related utility functions."""

    @patch("os.environ")
    def test_detect_proxy_settings(self, mock_environ):
        """Test detection of proxy settings from environment variables."""
        # Mock environment variables
        mock_environ.get.side_effect = lambda key, default=None: {
            "HTTP_PROXY": "http://proxy:8080",
            "HTTPS_PROXY": "https://proxy:8443",
            "NO_PROXY": "localhost,127.0.0.1",
        }.get(key, default)

        # Get proxy settings
        proxies = detect_proxy_settings()

        # Validate settings
        assert proxies["http"] == "http://proxy:8080"
        assert proxies["https"] == "https://proxy:8443"
        assert proxies["no_proxy"] == "localhost,127.0.0.1"

    def test_is_known_http2_host(self):
        """Test detection of known HTTP/2 hosts."""
        # Test known hosts
        assert is_known_http2_host("www.google.com") == True
        assert is_known_http2_host("github.com") == True
        assert is_known_http2_host("api.github.com") == True  # Subdomain

        # Test unknown hosts
        assert is_known_http2_host("example.com") == False
        assert is_known_http2_host("unknown-service.org") == False

    def test_calculate_backoff(self):
        """Test backoff calculation for retries."""
        # Test first retry (should be between 0 and base_delay)
        backoff1 = calculate_backoff(retry_count=1, base_delay=0.5)
        assert 0 <= backoff1 <= 0.5

        # Test second retry (should be between 0 and base_delay*2)
        backoff2 = calculate_backoff(retry_count=2, base_delay=0.5)
        assert 0 <= backoff2 <= 1.0

        # Test third retry (should be between 0 and base_delay*4)
        backoff3 = calculate_backoff(retry_count=3, base_delay=0.5)
        assert 0 <= backoff3 <= 2.0

        # Test with max_delay (should cap at max_delay)
        backoff_max = calculate_backoff(retry_count=10, base_delay=0.5, max_delay=5.0)
        assert 0 <= backoff_max <= 5.0
