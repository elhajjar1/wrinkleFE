# Contributing to WrinkleFE

Thank you for your interest in contributing to WrinkleFE! This document provides guidelines for contributing to the project.

## Reporting Issues

- Use the [GitHub Issues](https://github.com/elhajjar/wrinklefe/issues) tracker
- Include a minimal reproducible example when reporting bugs
- Describe your environment (OS, Python version, package versions)
- For feature requests, describe the use case and expected behavior

## Development Setup

```bash
# Clone the repository
git clone https://github.com/elhajjar/wrinklefe.git
cd wrinklefe

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install in development mode with all dependencies
pip install -e ".[all,dev]"

# Run tests
pytest
```

## Code Contributions

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run the test suite (`pytest`) and ensure all tests pass
5. Submit a pull request with a clear description

## Code Style

- Follow PEP 8 conventions
- Use type hints for function signatures
- Add docstrings to public functions and classes
- Keep functions focused and modular

## Testing

- Add tests for new functionality
- Maintain or improve test coverage
- Tests are in the `tests/` directory, organized by module

## Adding Materials

To add a new material to the built-in library, add an entry in `src/wrinklefe/core/material.py` in the `_load_builtins()` method. Include all elastic constants, strength allowables, and a literature reference.

## Questions

For questions about the science or implementation, open a GitHub issue with the "question" label.
