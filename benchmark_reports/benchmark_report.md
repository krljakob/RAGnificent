# RAGnificent Rust Benchmarks Performance Report

Generated on: 2025-06-15 18:24:52

## Performance Summary

- Total benchmarks: 8
- Benchmark groups: HTML Processing
- Test sizes: medium, small

### Performance Statistics

- Fastest operation: extract_links (0.003 ms)
- Slowest operation: extract_main_content (0.031 ms)
- Average execution time: 0.014 ms
- Median execution time: 0.016 ms

## Performance by Group

### HTML Processing

- Operations: 8
- Average time: 0.014 ms
- Best performance: 0.003 ms
- Worst performance: 0.031 ms

## Detailed Results

| Group | Benchmark | Size | Mean Time (ms) | Std Dev (ms) | CV (%) |
|-------|-----------|------|----------------|--------------|--------|
| HTML Processing | extract_links | small | 0.003 | 0.000 | 1.7 |
| HTML Processing | extract_main_content | small | 0.003 | 0.000 | 0.5 |
| HTML Processing | convert_to_markdown | small | 0.006 | 0.000 | 1.7 |
| HTML Processing | clean_html | small | 0.006 | 0.000 | 1.1 |
| HTML Processing | extract_links | medium | 0.016 | 0.001 | 8.5 |
| HTML Processing | convert_to_markdown | medium | 0.022 | 0.001 | 4.6 |
| HTML Processing | clean_html | medium | 0.029 | 0.002 | 5.6 |
| HTML Processing | extract_main_content | medium | 0.031 | 0.003 | 9.1 |
