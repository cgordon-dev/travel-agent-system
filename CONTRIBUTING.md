# Contributing to TravelGraph

Thank you for your interest in contributing to TravelGraph! This document outlines the guidelines for contributing to this project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please treat all community members with respect.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Create a new branch for your changes

## Development Environment Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

3. Run the development server:
   ```bash
   python api_server.py
   ```

## Making Changes

1. Create a new branch from `main` for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following the code style guidelines
3. Add tests for your changes if applicable
4. Ensure all tests pass
5. Update documentation if necessary

## Code Style Guidelines

- Follow PEP 8 style guidelines for Python code
- Use descriptive variable and function names
- Include docstrings for all functions, methods, and classes
- Keep lines under 100 characters when possible
- Sort imports alphabetically within their groups

## Submitting Changes

1. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Submit a pull request to the `main` branch of the original repository
3. In your pull request description, explain the changes and reference any related issues
4. Wait for maintainers to review your PR

## Pull Request Process

1. Update the README.md or other documentation with details of changes if needed
2. Ensure your PR passes all checks and tests
3. Your PR needs approval from at least one maintainer
4. Maintainers will merge your PR when it's ready

## Reporting Issues

- Use the GitHub Issues tab to report bugs or suggest features
- Search existing issues before creating a new one
- Include detailed steps to reproduce bugs
- Include relevant information about your environment

## Security Issues

If you discover a security vulnerability, please do NOT open an issue. Email [security@example.com](mailto:security@example.com) instead.

## License

By contributing to this project, you agree that your contributions will be licensed under the project's MIT License.