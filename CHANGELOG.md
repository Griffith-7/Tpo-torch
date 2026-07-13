# Changelog

## [0.1.0] - 2026-07-14

### Added
- Core TPO loss functions: `tpo_loss` and `tpo_loss_from_logits`
- `TPOTrainer` — HuggingFace Trainer subclass with TPO loss
- `TPODataCollator` — preserves advantages column through batching
- `TPOModel` — model wrapper with frozen reference policy
- CLI entry point (`tpo train`, `tpo bench`, `tpo info`)
- Unit tests, end-to-end tests, integration tests
- Benchmark suite
- CI/CD with GitHub Actions
- Pre-commit hooks (ruff, trailing whitespace)
- Docker support
- Full documentation

### Fixed
- Dead code in `integration_test.py` (duplicate `forward` method)

### Changed
- Upgraded `pyproject.toml` with full metadata, classifiers, and scripts
- Improved `.gitignore` to exclude training artifacts
