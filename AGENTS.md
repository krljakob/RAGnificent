# Repository Guidelines

## Project Structure & Module Organization
- `RAGnificent/`: Python package
  - `core/`, `rag/`, `utils/`: scraping, RAG pipeline, helpers
- `src/`: Rust library (exposed via PyO3/maturin as `RAGnificent.ragnificent_rs`)
- `tests/`: pytest suite (`unit/`, `integration/`, `rust/`)
- `examples/`, `docs/`, `data/`: demos, docs, local artifacts
- Key config: `.env.example`, `pyproject.toml`, `Cargo.toml`, `pytest.ini`, `.editorconfig`, `justfile`, `Makefile`

## Build, Test, and Development Commands
- Setup: `just setup` (creates venv with uv, installs deps, builds Rust ext)
- Build: `just build` or `make build` (Rust lib via maturin/cargo)
- Test: `just test` or `make test` (runs Rust + Python); quick: `./run_tests.sh fast`
- Format: `just format` or `make format` (Black+isort+Ruff, cargo fmt)
- Lint/Type: `just lint` (ruff, mypy, clippy)
- Bench: `cargo bench`; visualize: `python scripts/visualize_benchmarks.py`
Example: `python -m RAGnificent https://example.com -o output.md --save-chunks`

## Coding Style & Naming Conventions
- Python: Black line length 88, isort profile=black, Ruff rules in `pyproject.toml`
- Types: mypy (py312, strict options for defs/decorators)
- Rust: `cargo fmt`, `clippy` clean
- EditorConfig: spaces, 4-space indent, UTF-8, CRLF, trim trailing whitespace
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE`.

## Testing Guidelines
- Framework: `pytest`; discovery: files `test_*.py`, classes `Test*`, funcs `test_*`
- Markers: `unit`, `integration`, `benchmark`, `slow`, `network`, `requires_model`
- Run subsets: `pytest -m "not benchmark and not slow"`
- Rust bindings: `pytest tests/rust/test_python_bindings.py -v`

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise summary (<72 chars), scope optional
  Example: `Refactor scraper: faster sitemap parsing`
- PRs: clear description, link issues (`Closes #123`), list changes, test coverage notes; include screenshots for `webui.py` or benchmark deltas when relevant.

## Security & Configuration Tips
- Create `.env` from `.env.example`; never commit secrets
- Key vars: `OPENAI_API_KEY`, `QDRANT_*`, and pipeline tuning knobs (chunking/embedding)
- Prefer in-memory Qdrant for dev; set host/port for prod; validate via `view_qdrant_data.py`
