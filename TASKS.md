# TASKS.md — RAGnificent Refactor Phase 2

## ✅ Todo List by Module

### CLI (rag/main.py)

* [ ] Implement `argparse` with subcommands: `scrape`, `convert`, `pipeline`, `chat`
* [ ] Add `--help`, `--dry-run`, `--debug` to all subcommands
* [ ] Validate CLI args with typed hints

### Logging (rag/log.py)

* [ ] Build reusable `get_logger(name)` function
* [ ] Integrate rich/loguru for color-coded logging
* [ ] Log all warnings/errors to `~/.raglog/*.log`

### Scraper (core/scraper.py)

* [ ] Refactor `requests` → `httpx.AsyncClient`
* [ ] Add `await asyncio.gather()` batch logic
* [ ] Respect robots.txt, handle 429s gracefully

### Caching (core/cache.py)

* [ ] Add memoization decorator for sitemap/content calls
* [ ] Use `joblib.Memory` or `diskcache` for local file cache
* [ ] Allow `--cache-dir` override from CLI/config

### Plugin System (core/registry.py)

* [ ] Build registries: `FORMATTERS`, `VECTORS`, `EMBEDDERS`
* [ ] Load user-defined classes from `--plugin-dir`
* [ ] Fallback to built-in plugins if not found

### TUI (ui/tui.py)

* [ ] Display status for:

  * [ ] sitemap parsing
  * [ ] fetch queue progress
  * [ ] content preview (truncated)
* [ ] Keyboard toggles: `q`, `r`, `f`, `v`, `c`

### API (api/server.py)

* [ ] Add FastAPI endpoints:

  * [ ] `/scrape` POST
  * [ ] `/convert` GET
  * [ ] `/pipeline` POST
* [ ] Serve output files as streaming responses

### Pipeline (rag/pipeline.py)

* [ ] Parse `pipeline.yaml`
* [ ] Validate step chains
* [ ] Support `embed`, `store`, `query`, `rerank`

## 🧪 Tests

* [ ] Unit: scrape, convert, plugins
* [ ] Integration: full CLI → output
* [ ] Regression: invalid sitemaps, 404s, timeouts

## 🧼 Docs & Cleanup

* [ ] Add examples to `/examples/*.md`
* [ ] Update `README.md` and `docs/*.md`
* [ ] Create `CHANGELOG.md`
* [ ] Audit for dead code or unused imports
