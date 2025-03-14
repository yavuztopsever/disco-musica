# Contributing to Disco Musica

Thank you for your interest in contributing to Disco Musica! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [How to Contribute](#how-to-contribute)
4. [Development Workflow](#development-workflow)
5. [Pull Request Process](#pull-request-process)
6. [Style Guidelines](#style-guidelines)
7. [Testing](#testing)
8. [Documentation](#documentation)
9. [Community](#community)

## Code of Conduct

We are committed to fostering an open and welcoming environment. By participating in this project, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch or TensorFlow
- Audio processing libraries (Librosa, PyDub)
- MIDI processing libraries (Music21)
- Git

### Setup

1. Fork the repository on GitHub.
2. Clone your forked repository:
   ```bash
   git clone https://github.com/yourusername/disco-musica.git
   cd disco-musica
   ```

3. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate  # On Windows
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

## How to Contribute

There are many ways to contribute to Disco Musica:

- **Code**: Implement new features, fix bugs, or improve performance.
- **Documentation**: Improve documentation, write tutorials, or fix typos.
- **Testing**: Write tests, report bugs, or help with quality assurance.
- **Design**: Improve the user interface or create visual assets.
- **Ideas**: Suggest new features or improvements.

### Finding Issues to Work On

- Check the [Issues](https://github.com/yourusername/disco-musica/issues) page for open issues.
- Look for issues labeled "good first issue" if you're new to the project.
- If you find a bug or have an idea for a new feature, create a new issue before starting work.

## Development Workflow

1. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. Make your changes, following the [Style Guidelines](#style-guidelines).

3. Write tests for your changes.

4. Run the tests to ensure they pass:
   ```bash
   pytest
   ```

5. Commit your changes:
   ```bash
   git add .
   git commit -m "Your descriptive commit message"
   ```

6. Push your branch to GitHub:
   ```bash
   git push origin feature/your-feature-name
   ```

7. Create a pull request on GitHub.

## Pull Request Process

1. Ensure your code passes all tests and follows the style guidelines.
2. Update the documentation if needed.
3. Fill out the pull request template with all required information.
4. Request a review from one of the maintainers.
5. Address any feedback from reviewers.
6. Once approved, your pull request will be merged.

## Style Guidelines

### Python Code Style

We follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code. Additionally, we use:

- [Black](https://black.readthedocs.io/) for code formatting
- [Flake8](https://flake8.pycqa.org/) for linting
- [isort](https://pycqa.github.io/isort/) for sorting imports

To check and format your code, run:
```bash
black .
isort .
flake8
```

### Docstrings

We use [Google-style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html). Example:

```python
def func(arg1, arg2):
    """Summary of function.

    Args:
        arg1: Description of arg1.
        arg2: Description of arg2.

    Returns:
        Description of return value.

    Raises:
        ValueError: Description of when ValueError is raised.
    """
    return result
```

### Commit Messages

- Use the imperative mood ("Add feature" not "Added feature").
- First line should be 50 characters or less.
- Reference issues and pull requests where appropriate.
- Consider using [Conventional Commits](https://www.conventionalcommits.org/) format.

## Testing

We use [pytest](https://pytest.org/) for testing. All new features and bug fixes should include tests.

### Running Tests

Run all tests:
```bash
pytest
```

Run tests with coverage report:
```bash
pytest --cov=modules
```

### Writing Tests

When writing tests, follow these guidelines:

1. Test files should be in the `tests/` directory.
2. Test files should start with `test_`.
3. Test functions should start with `test_`.
4. Use descriptive names for tests.
5. Each test should test one specific functionality.
6. Use fixtures for setup and teardown.

## Documentation

We use Markdown for documentation. Documentation files are in the `docs/` directory.

When adding or updating features, please update the relevant documentation. This includes:

- Update the README.md if needed.
- Update the user guide (docs/user_guide.md) for user-facing changes.
- Update the developer guide (docs/developer_guide.md) for developer-facing changes.
- Add or update API documentation for new or modified functions.

## Community

- Join our [Discord server](https://discord.gg/your-discord-invite) to chat with other contributors.
- Subscribe to our [mailing list](mailto:your-mailing-list@example.com) for updates.
- Follow us on [Twitter](https://twitter.com/your-twitter-handle).

---

Thank you for contributing to Disco Musica! Your time and expertise help make this project better for everyone.