import json
from datetime import datetime

import orjson
import pytest
from enhanced_httpx.utils import (
    deserialize_json,
    format_curl_command,
    sanitize_sensitive_data,
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
