# Contributing to Disco Musica

Thank you for your interest in contributing to Disco Musica! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

1. [Code of Conduct](mdc:#code-of-conduct)
2. [Getting Started](mdc:#getting-started)
3. [Development Setup](mdc:#development-setup)
4. [Contributing Process](mdc:#contributing-process)
5. [Code Style](mdc:#code-style)
6. [Testing](mdc:#testing)
7. [Documentation](mdc:#documentation)
8. [Release Process](mdc:#release-process)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please read the [Code of Conduct](CODE_OF_CONDUCT.md) before contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/disco-musica.git
   cd disco-musica
   ```
3. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate  # On Windows
   ```

2. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Contributing Process

1. **Plan Your Contribution**
   - Check existing issues and pull requests
   - Create a new issue if needed
   - Discuss your approach with maintainers

2. **Make Your Changes**
   - Follow the code style guidelines
   - Write tests for new features
   - Update documentation
   - Commit your changes with clear messages

3. **Submit Your Changes**
   - Push your changes to your fork
   - Create a pull request
   - Provide a clear description of your changes
   - Link related issues

4. **Review Process**
   - Address review comments
   - Make requested changes
   - Ensure CI checks pass
   - Get approval from maintainers

## Code Style

We follow PEP 8 guidelines and use Black for code formatting.

1. **Python Code Style**
   - Use 4 spaces for indentation
   - Maximum line length: 88 characters (Black default)
   - Use descriptive variable names
   - Add type hints
   - Write docstrings for all public functions

2. **Documentation Style**
   - Use Google-style docstrings
   - Include examples in docstrings
   - Keep documentation up to date
   - Use clear and concise language

3. **Git Commit Messages**
   - Use present tense ("Add feature" not "Added feature")
   - First line should be 50 characters or less
   - Add detailed description if needed
   - Reference issues and pull requests

## Testing

1. **Write Tests**
   - Unit tests for all new features
   - Integration tests for complex functionality
   - Test edge cases and error conditions
   - Use pytest for testing

2. **Run Tests**
   ```bash
   # Run all tests
   pytest

   # Run specific test file
   pytest tests/test_feature.py

   # Run with coverage
   pytest --cov=disco_musica
   ```

3. **Test Coverage**
   - Maintain minimum 80% code coverage
   - Add tests for bug fixes
   - Test both success and failure cases

## Documentation

1. **Code Documentation**
   - Add docstrings to all public functions
   - Include type hints
   - Provide usage examples
   - Document exceptions

2. **API Documentation**
   - Update API documentation
   - Add new endpoints
   - Document parameters
   - Include examples

3. **User Documentation**
   - Update user guides
   - Add new features
   - Include screenshots
   - Provide troubleshooting tips

## Release Process

1. **Version Bumping**
   - Follow semantic versioning
   - Update version in setup.py
   - Update CHANGELOG.md
   - Tag releases

2. **Release Checklist**
   - Run all tests
   - Check documentation
   - Update dependencies
   - Create release notes

3. **Deployment**
   - Build distribution
   - Upload to PyPI
   - Update website
   - Announce release

## Questions?

If you have any questions, please:
1. Check the documentation
2. Search existing issues
3. Ask in discussions
4. Contact maintainers

Thank you for contributing to Disco Musica!