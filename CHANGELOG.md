# CHANGELOG.md

# Changelog

## [Unreleased]

### Added
- Initial implementation of the `ReqxClient` class for custom HTTP requests.
- Support for asynchronous operations using `httpx`.
- Pydantic models for request and response validation.
- Custom exceptions for error handling.
- Utility functions for logging and data serialization with `orjson`.
- Command-line interface for making HTTP requests.

### Changed
- Improved error handling in the `ReqxClient` class.
- Enhanced logging capabilities for better debugging.

### Fixed
- Resolved issues with session management in asynchronous requests.
- Fixed validation errors in Pydantic models.

## [0.1.0] - 2023-10-01
### Added
- Project structure and initial files.
- Basic tests for `ReqxClient` and Pydantic models.
- Example scripts for asynchronous requests and CLI usage.

### Changed
- Updated dependencies in `pyproject.toml`.

### Fixed
- Minor bugs in request handling and response parsing.
