# Contributing to Gemini Agent Framework

Thank you for your interest in contributing to the Gemini Agent Framework! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please read it before contributing.

## How to Contribute

### 1. Reporting Bugs

If you find a bug, please create an issue with:
- A clear, descriptive title
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, etc.)
- Any relevant logs or error messages

### 2. Suggesting Features

We welcome feature suggestions! Please create an issue with:
- A clear, descriptive title
- Detailed description of the feature
- Use cases and examples
- Any relevant implementation ideas

### 3. Pull Requests

1. Fork the repository
2. Create a new branch for your changes
3. Make your changes
4. Add tests for your changes
5. Update documentation
6. Submit a pull request

### 4. Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/m7mdony/gemini-agent-framework.git
   cd gemini-agent-framework
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Run tests:
   ```bash
   pytest
   ```

### 5. Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all functions and classes
- Keep lines under 100 characters
- Use meaningful variable names

### 6. Testing

- Write unit tests for new features
- Ensure all tests pass
- Maintain or improve test coverage
- Test edge cases and error conditions

### 7. Documentation

- Update README.md if needed
- Add docstrings to new functions/classes
- Update API documentation
- Add examples for new features

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Update the documentation with any new features
3. Add tests for new functionality
4. Ensure all tests pass
5. Update the version number in pyproject.toml
6. The PR will be merged once it has been reviewed and approved

## Development Workflow

1. Create a new branch for your feature/fix
2. Make your changes
3. Run tests locally
4. Update documentation
5. Submit PR
6. Address review comments
7. PR gets merged

## Commit Messages

Follow these guidelines for commit messages:
- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally

## Review Process

1. All PRs require at least one review
2. PRs must pass all tests
3. Documentation must be updated
4. Code must follow style guidelines
5. Changes must be well-tested

## Release Process

1. Update version number
2. Update changelog
3. Create release notes
4. Tag the release
5. Build and publish to PyPI

## Getting Help

- Create an issue for bugs or feature requests
- Join our Discord community
- Check the documentation
- Ask in discussions

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License. 