# ✨ RAGnificent ✨

A developer-first, **blazingly fast**, hypermodern Rust/Python hybrid webscraper that intuitively converts *any* accessible webpages into structured markdown, JSON, and XML formats optimized for preparing LLM retrieval augmented generation (RAG).

RAGnificent natively combines the expressiveness of Python with the *raw performance* of Rust to scrape web content and convert it into semantically meaningful document chunks. It's specifically engineered for building **high-quality** training sets and knowledge bases for RAG applications. The Rust/Python integration is achieved natively through the maturin and pyo3 crates thanks to the hardwork of the legendary engineers at Pythonium Trioxide. Links to those crates are directly below.

[Maturin](https://github.com/PyO3/maturin)  
[Pyo3](https://github.com/PyO3/pyo3)

> While the rust implementation is optional, it's worth noting that the `.xml` and `.json` formats do rely on the Rust target being built prior to scraping, while the `.md` format only requires python 3.12 and a few py dependencies.

## 🚀 Key Features:

* **Semantic Chunking**: Intelligently preserves document structure throughout the chunking and scraping process, maintaining context for LLMs
* **Multi-Format Output**: Export to clean Markdown, structured JSON, or XML formats
* **Sitemap Integration**: Automatically discover and parse XML sitemaps for systematic content crawling
* **Rust-Powered Performance**: Core processing routines implemented in Rust for **blazing-fast** operation
* **Flexible Configuration**: Customize chunk sizes, overlap, parsing strategies, and more
* **Rate Limiting & Caching**: Built-in request throttling and caching to be a good web citizen
* **Parallel Processing**: Optional multi-threaded operation for processing multiple URLs simultaneously, making larger scraping tasks a breeze

## 🔍 Ideal For:

* Building custom LLM knowledge bases from documentation sites
* Preparing web content for vector database ingestion
* Creating training sets for fine-tuning language models
* Archiving websites in clean, structured formats

RAGnificent provides both a command-line interface and a programmer-friendly API, making it suitable for both quick scraping jobs and integration into larger data processing pipelines.
