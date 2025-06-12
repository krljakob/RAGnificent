// benches/chunker_bench.rs
// Criterion benchmark for semantic chunking
// Run with: cargo bench --bench chunker_bench

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ragnificent_rs::chunker::create_semantic_chunks;

fn bench_semantic_chunking(c: &mut Criterion) {
    let markdown = include_str!("../test_data/sample.md");
    let chunk_sizes = [256, 512, 1024, 2048];
    let overlap = 128;

    let mut group = c.benchmark_group("semantic_chunking");
    for &size in &chunk_sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &s| {
            b.iter(|| {
                let _ = create_semantic_chunks(markdown, s, overlap).unwrap();
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_semantic_chunking);
criterion_main!(benches);
