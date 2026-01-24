# qlora-rs - 4-bit Quantized LoRA

## Overview

4-bit NormalFloat (NF4) quantization and QLoRA implementation for memory-efficient LLM fine-tuning. Depends on peft-rs for adapter management.

## Architecture

```
src/
├── lib.rs           # Public API exports
├── quantization.rs  # Core NF4/INT4 quantization algorithms
├── qlora.rs         # QLoRA layer: quantized weights + LoRA adapters
├── training.rs      # QLoRA training utilities
├── export.rs        # Export to GGUF and native formats
├── formats.rs       # Format definitions (GGUF, QNAT)
├── native.rs        # Candle-native quantized format (QNAT)
└── error.rs         # Error types

tests/
└── training_integration.rs  # End-to-end training tests
```

## Key Components

### NF4 Quantization (`quantization.rs`)
```rust
// 4-bit NormalFloat quantization optimized for neural network weights
pub struct Nf4Quantizer {
    block_size: usize,      // Typically 64 or 128
    double_quant: bool,     // Quantize the scales too
}

// Quantized tensor representation
pub struct QuantizedTensor {
    data: Vec<u8>,          // Packed 4-bit values
    scales: Tensor,         // Per-block scale factors
    zeros: Option<Tensor>,  // Zero points (for asymmetric)
}
```

### QLoRA Layer (`qlora.rs`)
```rust
// Frozen quantized base + trainable LoRA
pub struct QLoraLayer {
    quantized_weight: QuantizedTensor,  // Frozen NF4
    lora_a: Tensor,                      // Trainable
    lora_b: Tensor,                      // Trainable
    scale: f64,                          // alpha / rank
}
```

## Dependency on peft-rs

This crate requires peft-rs v1.0+ for:
- `Adapter` trait implementation
- `weights()` method for accessing trainable parameters
- `LoraConfig` for configuration

```rust
// In qlora.rs
use peft_rs::{Adapter, LoraConfig};
```

## Development Commands

```bash
# Check (will also check peft-rs dependency)
cargo check -p qlora-rs

# Test
cargo test -p qlora-rs

# Test with CUDA
cargo test -p qlora-rs --features cuda

# Integration tests
cargo test -p qlora-rs --test training_integration

# Benchmarks
cargo bench -p qlora-rs

# Clippy
cargo clippy -p qlora-rs -- -W clippy::all
```

## Critical Code Paths

### Quantization (`quantization.rs`)
Performance-critical - process large weight matrices:
```rust
pub fn quantize_nf4(tensor: &Tensor, block_size: usize) -> Result<QuantizedTensor>
pub fn dequantize_nf4(quantized: &QuantizedTensor) -> Result<Tensor>
```

### QLoRA Forward (`qlora.rs`)
```rust
// y = dequant(W_q) @ x + (x @ A @ B) * scale
// Dequantization should be fused with matmul when possible
```

### GGUF Export (`export.rs`)
Must produce llama.cpp compatible files for inference deployment.

## Export Formats

| Format | Feature Flag | Use Case |
|--------|--------------|----------|
| GGUF | `gguf-export` | llama.cpp inference |
| QNAT | `native-export` | Candle native loading |

## Testing Strategy

- Unit tests: Quantization math correctness
- Property tests: Round-trip quantize/dequantize within tolerance
- Integration: Full training step with loss computation
- GPU tests: CUDA kernel correctness (ignored by default)

## 1.0 Checklist

- [x] NF4 quantization
- [x] Double quantization
- [x] QLoRA inference layer
- [x] GGUF export basics
- [x] Published to crates.io
- [ ] Full GGUF metadata support
- [ ] Training loop with gradient checkpointing
- [ ] Memory profiling and optimization
- [ ] Benchmark against bitsandbytes
- [x] Examples directory
- [ ] 100% doc coverage

## Common Issues

### "peft-rs version mismatch"
Ensure Cargo.toml specifies `peft-rs = "1.0"` or compatible.

### Quantization produces NaN
Check for zero/negative values in scales. Add epsilon to denominators.

### GGUF file not loading in llama.cpp
Verify GGUF magic bytes and metadata format matches spec.

## Performance Notes

- NF4 quantization: ~4x memory reduction
- Block size 64: Better accuracy, more overhead
- Block size 128: Faster, slightly less accurate
- Double quantization: Additional ~0.5x reduction for scales
