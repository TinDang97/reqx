import pytest
from models import RequestModel, ResponseModel
from pydantic import ValidationError


class TestRequestModel:
    def test_valid_request_model(self):
        data = {
            "url": "https://example.com",
            "method": "GET",
            "headers": {"Authorization": "Bearer token"},
            "cookies": {"session_id": "abc123"},
            "body": None,
        }
        request = RequestModel(**data)
        assert request.url == data["url"]
        assert request.method == data["method"]
        assert request.headers == data["headers"]
        assert request.cookies == data["cookies"]
        assert request.body == data["body"]

    def test_invalid_request_model_missing_url(self):
        data = {
            "method": "GET",
            "headers": {"Authorization": "Bearer token"},
            "cookies": {"session_id": "abc123"},
            "body": None,
        }
        with pytest.raises(ValidationError):
            RequestModel(**data)

    def test_invalid_request_model_invalid_method(self):
        data = {
            "url": "https://example.com",
            "method": "INVALID_METHOD",
            "headers": {"Authorization": "Bearer token"},
            "cookies": {"session_id": "abc123"},
            "body": None,
        }
        with pytest.raises(ValidationError):
            RequestModel(**data)


class TestResponseModel:
    def test_valid_response_model(self):
        data = {
            "status_code": 200,
            "headers": {"Content-Type": "application/json"},
            "body": {"key": "value"},
        }
        response = ResponseModel(**data)
        assert response.status_code == data["status_code"]
        assert response.headers == data["headers"]
        assert response.body == data["body"]

    def test_invalid_response_model_missing_status_code(self):
        data = {"headers": {"Content-Type": "application/json"}, "body": {"key": "value"}}
        with pytest.raises(ValidationError):
            ResponseModel(**data)

    def test_invalid_response_model_invalid_status_code(self):
        data = {
            "status_code": "NOT_A_NUMBER",
            "headers": {"Content-Type": "application/json"},
            "body": {"key": "value"},
        }
        with pytest.raises(ValidationError):
            ResponseModel(**data)
