from RAGnificent.core.scraper import main

if __name__ == "__main__":
    main(
        url="https://terminaltrove.com/new/",
        output_file="terminaltrove.md",
        output_format="markdown",
        save_chunks=False,
        chunk_dir="chunks",
        chunk_format="jsonl",
        chunk_size=1000,
        chunk_overlap=200,
        requests_per_second=1.0,
        use_sitemap=False,
        min_priority=None,
        include_patterns=None,
        exclude_patterns=None,
        limit=None,
        cache_enabled=False,
        cache_max_age=3600,
        skip_cache=True,
        links_file=None,
        parallel=False,
        max_workers=4,
        worker_timeout=None,
    )
