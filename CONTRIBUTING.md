# Contributing to WrinkleFE

Thank you for your interest in contributing to WrinkleFE! This document provides guidelines for contributing to the project.

## Reporting Issues

- Use the [GitHub Issues](https://github.com/elhajjar1/wrinkleFE/issues) tracker
- Include a minimal reproducible example when reporting bugs
- Describe your environment (OS, Python version, package versions)
- For feature requests, describe the use case and expected behavior

## Development Setup

```bash
# Clone the repository
git clone https://github.com/elhajjar1/wrinklefe.git
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

For one-off or ad-hoc materials you do not need to modify source — turn on
**Expert mode** in the Streamlit app and pick **Custom…** from the
**Material** selectbox. The inline editor exposes E1/E2/E3, G12/G13/G23,
ν12/ν13/ν23 and the Xt/Xc/Yt/Yc/Zt/Zc/S12/S13/S23 allowables, seeded from
IM7/8552. Custom materials are scoped to the current Streamlit session and
do not persist; use the source-level workflow above for anything you want
to keep. See [`DEPLOYMENT_STREAMLIT.md`](DEPLOYMENT_STREAMLIT.md) for a
full feature tour.

## Questions

For questions about the science or implementation, open a GitHub issue with the "question" label.
