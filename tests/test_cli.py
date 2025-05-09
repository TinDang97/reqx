import pytest
from click.testing import CliRunner
from enhanced_httpx.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


def test_cli_help(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_cli_get_request(runner):
    result = runner.invoke(cli, ["get", "https://httpbin.org/get"])
    assert result.exit_code == 0
    assert '"url": "https://httpbin.org/get"' in result.output


def test_cli_post_request(runner):
    result = runner.invoke(cli, ["post", "https://httpbin.org/post", "--data", '{"key": "value"}'])
    assert result.exit_code == 0
    assert '"key": "value"' in result.output


def test_cli_invalid_url(runner):
    result = runner.invoke(cli, ["get", "invalid-url"])
    assert result.exit_code != 0
    assert "Invalid URL" in result.output


def test_cli_missing_method(runner):
    result = runner.invoke(cli, ["https://httpbin.org/get"])
    assert result.exit_code != 0
    assert "Error: Missing method" in result.output
