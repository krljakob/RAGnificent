# Product Requirements Document (PRD): RAGnificent Refactor Phase 2

## Title

**RAGnificent v2: Modular, Performant, and Extensible Document Scraper with RAG Pipeline Integration**

## Owner

@ursisterbtw

## Background

The existing refactor on the `dev` branch has significantly reduced code bloat and improved maintainability. However, the codebase remains in a transitional state: multiple UX, DX, and performance optimizations are still pending. To move from functional to production-grade, we need to further modularize, harden, and extend RAGnificent.

## Objective

Make RAGnificent a production-ready document ingestion and retrieval framework, capable of being used both interactively (via CLI/API) and programmatically (as a Python package). This phase will:

1. Finalize modular structure for core + plugins.
2. Optimize performance with parallelism + caching.
3. Enhance UX with a composable CLI and optional TUI.
4. Improve testing, traceability, and developer ergonomics.

## Scope

### Must-Have Features

* CLI: `rag scrape`, `rag convert`, `rag pipeline`, `rag chat`
* Async HTTP using `httpx.AsyncClient`
* Centralized logging (with rich/loguru support)
* Unified plugin system for:

  * Formatters
  * Vector stores
  * Embedders
* File-level + request-level caching
* TUI dashboard (optional)
* Snapshot test output formatting
* Pyproject-based packaging + installable CLI
* Optional FastAPI wrapper for headless use

### Nice-to-Haves

* Configurable pipeline chaining from YAML
* Metadata extraction using Readability
* Rust FFI integration for sitemap parsing or token chunking
* Browser scraping fallback using Playwright

## Non-Goals

* GPU-based inference integration (future phase)
* RAG model training or hosting

## Timeline

**Phase 2 duration: 3 weeks**

* Week 1: CLI, logging, async HTTP
* Week 2: plugin system, caching, tests
* Week 3: TUI, docs, polish, PR merge

## Success Metrics

* > 95% of code modules have test coverage
* > 2x scraping throughput via `asyncio`
* CLI usage documented with `--help` for all commands
* User config example YAMLs in `/examples`

## Stakeholders

* Internal: Dev team (you)
* External: Open-source contributors, plugin devs

---
