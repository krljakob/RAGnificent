use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use ragnificent_rs::{
    chunker::create_semantic_chunks,
    html_parser::{clean_html, extract_links, extract_main_content},
    markdown_converter::{convert_html, convert_to_markdown, OutputFormat},
    simd_text::{
        calculate_semantic_density_simd, clean_whitespace_simd, count_lines_simd,
        count_non_whitespace_chars_simd, count_words_simd, text_similarity_simd,
    },
};
use std::time::Duration;

fn bench_html_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("HTML Processing");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(100);

    // Test data
    let html_samples = [
        (
            "small",
            "<html><body><main><h1>Test</h1><p>Small content</p></main></body></html>",
        ),
        ("medium", include_str!("../test_data/medium.html")),
        ("large", include_str!("../test_data/large.html")),
    ];

    for (size, html) in html_samples.iter() {
        // Benchmark main content extraction
        group.bench_with_input(
            BenchmarkId::new("extract_main_content", size),
            html,
            |b, html| b.iter(|| extract_main_content(black_box(html))),
        );

        // Benchmark HTML cleaning
        group.bench_with_input(BenchmarkId::new("clean_html", size), html, |b, html| {
            b.iter(|| clean_html(black_box(html)))
        });

        // Benchmark link extraction
        group.bench_with_input(BenchmarkId::new("extract_links", size), html, |b, html| {
            b.iter(|| extract_links(black_box(html), "https://example.com"))
        });

        // Benchmark markdown conversion
        group.bench_with_input(
            BenchmarkId::new("convert_to_markdown", size),
            html,
            |b, html| b.iter(|| convert_to_markdown(black_box(html), "https://example.com")),
        );
    }

    group.finish();
}

fn bench_chunking(c: &mut Criterion) {
    let mut group = c.benchmark_group("Text Chunking");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(100);

    let chunk_sizes = [100, 500, 1000];
    let overlap_sizes = [10, 50, 100];

    let markdown = include_str!("../test_data/sample.md");

    for &chunk_size in chunk_sizes.iter() {
        for &overlap in overlap_sizes.iter() {
            group.bench_with_input(
                BenchmarkId::new(
                    format!("chunk_size_{}_overlap_{}", chunk_size, overlap),
                    chunk_size,
                ),
                &(chunk_size, overlap),
                |b, &(chunk_size, overlap)| {
                    b.iter(|| {
                        create_semantic_chunks(
                            black_box(markdown),
                            black_box(chunk_size),
                            black_box(overlap),
                        )
                    })
                },
            );
        }
    }

    group.finish();
}

// Benchmark SIMD text operations for performance comparison
fn bench_simd_text_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("SIMD Text Operations");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(1000);

    // Test texts of different sizes
    let texts = [
        ("small", "This is a short test text with a few words."),
        ("medium", include_str!("../test_data/sample.md")),
        ("large", include_str!("../test_data/large.html")),
    ];

    for (size, text) in texts.iter() {
        group.throughput(Throughput::Bytes(text.len() as u64));

        // Word counting benchmark
        group.bench_with_input(
            BenchmarkId::new("count_words_simd", size),
            text,
            |b, text| b.iter(|| count_words_simd(black_box(text))),
        );

        // Character counting benchmark  
        group.bench_with_input(
            BenchmarkId::new("count_chars_simd", size),
            text,
            |b, text| b.iter(|| count_non_whitespace_chars_simd(black_box(text))),
        );

        // Line counting benchmark
        group.bench_with_input(
            BenchmarkId::new("count_lines_simd", size),
            text,
            |b, text| b.iter(|| count_lines_simd(black_box(text))),
        );

        // Semantic density calculation benchmark
        group.bench_with_input(
            BenchmarkId::new("semantic_density_simd", size),
            text,
            |b, text| b.iter(|| calculate_semantic_density_simd(black_box(text))),
        );
    }

    group.finish();
}

// Benchmark different output formats
fn bench_output_formats(c: &mut Criterion) {
    let mut group = c.benchmark_group("Output Formats");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(100);

    let html = include_str!("../test_data/medium.html");
    let formats = [
        ("markdown", OutputFormat::Markdown),
        ("json", OutputFormat::Json),
        ("xml", OutputFormat::Xml),
    ];

    for (format_name, format) in formats.iter() {
        group.throughput(Throughput::Bytes(html.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("convert_html", format_name),
            format,
            |b, format| {
                b.iter(|| {
                    convert_html(
                        black_box(html),
                        black_box("https://example.com"),
                        black_box(*format),
                    )
                })
            },
        );
    }

    group.finish();
}

// Benchmark text cleaning operations
fn bench_text_cleaning(c: &mut Criterion) {
    let mut group = c.benchmark_group("Text Cleaning");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(500);

    let messy_texts = [
        ("short", "  hello   world  \t\n  "),
        ("medium", "  This   is   a   test   with   lots   of    extra   whitespace   and   \t\t tabs \n\n\n and newlines  "),
        ("long", &" ".repeat(100).replace(" ", "   ").repeat(50)),
    ];

    for (size, text) in messy_texts.iter() {
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("clean_whitespace_simd", size),
            text,
            |b, text| b.iter(|| clean_whitespace_simd(black_box(text))),
        );
    }

    group.finish();
}

// Benchmark text similarity calculations
fn bench_text_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("Text Similarity");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(200);

    let text_pairs = [
        ("identical", "hello world", "hello world"),
        ("similar", "hello world test", "hello world testing"),
        ("different", "hello world", "completely different text"),
        ("long_similar", include_str!("../test_data/sample.md"), include_str!("../test_data/medium.html")),
    ];

    for (case, text1, text2) in text_pairs.iter() {
        let combined_len = text1.len() + text2.len();
        group.throughput(Throughput::Bytes(combined_len as u64));
        group.bench_with_input(
            BenchmarkId::new("text_similarity_simd", case),
            &(text1, text2),
            |b, (text1, text2)| {
                b.iter(|| text_similarity_simd(black_box(text1), black_box(text2)))
            },
        );
    }

    group.finish();
}

// Benchmark parallel processing vs sequential
fn bench_parallel_vs_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("Parallel vs Sequential");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(50);

    let documents = [
        include_str!("../test_data/medium.html"),
        include_str!("../test_data/large.html"),
        include_str!("../test_data/sample.md"),
    ];

    // Benchmark batch processing of multiple documents
    group.bench_function("batch_markdown_conversion", |b| {
        b.iter(|| {
            for (i, doc) in documents.iter().enumerate() {
                convert_to_markdown(
                    black_box(doc),
                    black_box(&format!("https://example{}.com", i)),
                )
                .unwrap();
            }
        })
    });

    // Benchmark batch chunking
    group.bench_function("batch_chunking", |b| {
        b.iter(|| {
            for doc in documents.iter() {
                create_semantic_chunks(black_box(doc), black_box(1000), black_box(100))
                    .unwrap();
            }
        })
    });

    group.finish();
}

// Memory efficiency benchmarks
fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory Efficiency");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    // Test with progressively larger documents
    let sizes = [1, 5, 10, 25];
    let base_html = include_str!("../test_data/large.html");

    for size in sizes.iter() {
        let large_document = base_html.repeat(*size);
        group.throughput(Throughput::Bytes(large_document.len() as u64));
        
        group.bench_with_input(
            BenchmarkId::new("large_document_conversion", format!("{}x", size)),
            &large_document,
            |b, doc| {
                b.iter(|| {
                    convert_to_markdown(black_box(doc), black_box("https://example.com"))
                        .unwrap()
                })
            },
        );
    }

    group.finish();
}

// Comprehensive throughput benchmark
fn bench_comprehensive_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("Comprehensive Throughput");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(30);

    let html_doc = include_str!("../test_data/medium.html");
    let markdown_doc = include_str!("../test_data/sample.md");

    // Full pipeline benchmark: HTML -> Markdown -> Chunks -> Analysis
    group.bench_function("full_pipeline", |b| {
        b.iter(|| {
            // Convert HTML to markdown
            let markdown = convert_to_markdown(
                black_box(html_doc),
                black_box("https://example.com"),
            )
            .unwrap();

            // Create semantic chunks
            let chunks = create_semantic_chunks(
                black_box(&markdown),
                black_box(1000),
                black_box(100),
            )
            .unwrap();

            // Analyze each chunk
            for chunk in chunks {
                count_words_simd(black_box(&chunk));
                calculate_semantic_density_simd(black_box(&chunk));
            }
        })
    });

    // Text processing pipeline
    group.bench_function("text_processing_pipeline", |b| {
        b.iter(|| {
            let word_count = count_words_simd(black_box(markdown_doc));
            let line_count = count_lines_simd(black_box(markdown_doc));
            let char_count = count_non_whitespace_chars_simd(black_box(markdown_doc));
            let density = calculate_semantic_density_simd(black_box(markdown_doc));
            
            // Use the results to prevent optimization
            black_box((word_count, line_count, char_count, density));
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_html_processing,
    bench_chunking,
    bench_simd_text_operations,
    bench_output_formats,
    bench_text_cleaning,
    bench_text_similarity,
    bench_parallel_vs_sequential,
    bench_memory_efficiency,
    bench_comprehensive_throughput
);
criterion_main!(benches);
