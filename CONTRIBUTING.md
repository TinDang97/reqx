# Contributing to Enhanced-HTTPX

Thank you for your interest in contributing to Enhanced-HTTPX! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We aim to foster an inclusive and welcoming community.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up your development environment
4. Create a new branch for your feature or bugfix
5. Make your changes
6. Run tests to ensure your changes don't break existing functionality
7. Submit a pull request

## Development Environment

To set up your development environment:

```bash
# Clone the repository
git clone https://github.com/yourusername/enhanced-httpx.git
cd enhanced-httpx

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in development mode
pip install -e .[dev,test]
```

## Testing

We use `pytest` for testing. To run the tests:

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=enhanced_httpx

# Run a specific test file
pytest tests/test_client.py

# Run the test script
python scripts/run_tests.py
```

All new features should include appropriate tests, and all bugfixes should include tests that reproduce the fixed issue.

## Pull Request Process

1. Ensure your code passes all tests and linting checks
2. Update the documentation with details of changes if needed
3. Update the CHANGELOG.md with details of changes
4. The PR should include tests for new functionality or bugfixes
5. You may merge the pull request once it has been approved by at least one maintainer

## Style Guidelines

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) and use the following tools:

- `black` for code formatting
- `isort` for import sorting
- `flake8` for linting

To format and check your code:

```bash
# Format code
black src tests examples

# Sort imports
isort src tests examples

# Run linting
flake8 src tests examples
```

## Documentation

- Update the README.md if you're adding or changing features
- Docstrings should follow the Google style format
- Consider updating or adding examples for new features
- If appropriate, update the API reference documentation

## Issue Reporting

When reporting issues, please include:

1. A clear and descriptive title
2. Steps to reproduce the issue
3. Expected behavior
4. Actual behavior
5. Environment information (OS, Python version, package version)
6. Any relevant logs or error messages

## Releasing

For maintainers, to release a new version:

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create a new GitHub release
4. Deploy to PyPI using GitHub Actions

Thank you for contributing to Enhanced-HTTPX!