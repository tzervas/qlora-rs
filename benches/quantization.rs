//! Quantization benchmarks for qlora-rs.

use candle_core::{Device, Tensor};
use criterion::{criterion_group, criterion_main, Criterion};
use qlora_rs::{dequantize_nf4, quantize_nf4, QLoraConfig, QuantizedLinear};
use std::hint::black_box;

fn benchmark_quantization(c: &mut Criterion) {
    let device = Device::Cpu;

    // Benchmark 1: quantize_nf4 - Quantizing a 4096x4096 tensor
    c.bench_function("quantize_nf4_4096", |b| {
        let tensor = Tensor::randn(0f32, 1f32, (4096, 4096), &device).unwrap();
        b.iter(|| black_box(quantize_nf4(black_box(&tensor), 64).unwrap()));
    });

    // Benchmark 2: dequantize_nf4 - Dequantizing back
    c.bench_function("dequantize_nf4_4096", |b| {
        let tensor = Tensor::randn(0f32, 1f32, (4096, 4096), &device).unwrap();
        let quantized = quantize_nf4(&tensor, 64).unwrap();
        b.iter(|| black_box(dequantize_nf4(black_box(&quantized), &device).unwrap()));
    });

    // Benchmark 3: qlora_forward - QLoraLayer forward pass
    c.bench_function("qlora_forward_4096", |b| {
        let config = QLoraConfig::default();
        let layer = QuantizedLinear::new(4096, 4096, &config, &device).unwrap();
        let input = Tensor::randn(0f32, 1f32, (1, 128, 4096), &device).unwrap();
        b.iter(|| black_box(layer.forward(black_box(&input)).unwrap()));
    });
}

criterion_group!(benches, benchmark_quantization);
criterion_main!(benches);
