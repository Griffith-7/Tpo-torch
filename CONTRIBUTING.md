# Contributing to TPO-Torch

Thanks for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/Griffith-7/tpo-torch.git
cd tpo-torch
pip install -e ".[dev]"
pre-commit install
```

## Running Tests

```bash
# Unit tests
pytest tests/ -v

# Benchmarks
pytest benchmarks/ -v -s

# Lint
ruff check tpo_torch/
ruff format --check tpo_torch/

# Integration test
python tests/integration_test.py
```

## Code Style

- Format with `ruff format`
- Lint with `ruff check`
- Keep lines under 100 characters where possible
- Add type hints to new functions
- Write docstrings for public APIs

## Pull Requests

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run tests and lint
5. Open a PR with a clear description

## Reporting Issues

Open an issue on GitHub with:
- What you expected
- What actually happened
- Steps to reproduce
- Python/PyTorch/transformers versions
