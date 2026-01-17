//! Quantization benchmarks for qlora-rs.

use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_quantization(c: &mut Criterion) {
    // Benchmarks will be added as quantization is implemented
    let group = c.benchmark_group("quantization");
    group.finish();
}

criterion_group!(benches, benchmark_quantization);
criterion_main!(benches);
