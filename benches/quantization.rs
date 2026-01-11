use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn benchmark_quantization(c: &mut Criterion) {
    // Benchmarks will be added as quantization is implemented
    let mut group = c.benchmark_group("quantization");
    group.finish();
}

criterion_group!(benches, benchmark_quantization);
criterion_main!(benches);
