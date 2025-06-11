# PLANNING.md — RAGnificent Refactor Phase 2

## Milestone Breakdown

### Milestone 1: CLI & Logging Foundation

* [ ] Replace `main.py` CLI with subcommand parser (argparse or typer)
* [ ] Add `--dry-run`, `--debug`, and `--trace` flags
* [ ] Implement centralized logger with `loguru` or `rich`
* [ ] Create CLI usage examples in `/examples`

### Milestone 2: Async + Caching

* [ ] Convert `scraper.py` network code to `httpx.AsyncClient`
* [ ] Introduce connection pooling and timeout control
* [ ] Implement request + output file caching (`joblib.Memory`, `diskcache`, or `pickle`)
* [ ] Add test cases for cache hit/miss scenarios

### Milestone 3: Plugin Architecture

* [ ] Refactor vector store, embedder, and formatter classes into plugin registries
* [ ] Allow dynamic loading via `--plugin-dir`
* [ ] Register built-in plugins at startup
* [ ] Write `docs/plugins.md`

### Milestone 4: UI Enhancements

* [ ] Build `tui.py` with curses or textual for live feedback
* [ ] Add TUI views for:

  * Fetch progress (URLs scraped / total)
  * Output format preview (Markdown / JSON)
  * Warnings / errors / skips

### Milestone 5: Pipeline + Packaging

* [ ] Add `pipeline.yaml` loader with chainable steps
* [ ] Allow user config via `--config` or environment vars
* [ ] Add `pyproject.toml` with CLI `entry_points`
* [ ] Add `FastAPI` wrapper with endpoints:

  * `/scrape`
  * `/convert`
  * `/query`

### Milestone 6: Tests + Docs + Polish

* [x] Optimize test performance (separated benchmarks from unit tests)
* [x] Add test markers for categorization (benchmark, slow, integration, requires_model)
* [x] Create `run_tests.sh` for easy test execution
* [ ] Snapshot test all output formats
* [ ] Add regression tests for bad input, 404s, invalid sitemaps
* [ ] Document all CLI commands with examples
* [ ] Write `CHANGELOG.md`, `RELEASE_NOTES.md`

## Integration Timeline (3 Weeks)

* **Week 1**: Milestone 1 + 2
* **Week 2**: Milestone 3 + 4
* **Week 3**: Milestone 5 + 6
