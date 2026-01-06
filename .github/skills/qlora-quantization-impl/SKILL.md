---
name: qlora-quantization-impl
description: Implement and debug 4-bit quantization schemes, GGUF export, and quantized training in qlora-rs
---

# QLoRA Quantization Implementation Skill

## When to Use

Invoke when the user asks to:
- Implement new quantization schemes (Q4_1, Q5_0, etc.)
- Debug quantization accuracy issues
- Optimize quantization performance
- Add new GGUF export features
- Implement double quantization

## Quantization Fundamentals

### NF4 Format
- 16 levels optimized for N(0,1) distribution
- Block-based with shared scale per block
- Packed: 2 values per byte

### Key Operations

```rust
// Quantize single value
fn quantize_value_nf4(normalized: f32) -> u8 {
    // Find nearest NF4 level (0-15)
}

// Dequantize single value
fn dequantize_value_nf4(code: u8, scale: f32) -> f32 {
    NF4_LEVELS[code as usize] * scale
}
```

## Adding New Quantization Scheme

### Step 1: Define Levels

```rust
// src/quantization/q5_0.rs
pub const Q5_0_LEVELS: [f32; 32] = [
    // 32 uniformly spaced levels
];
```

### Step 2: Implement Quantize/Dequantize

```rust
pub fn quantize_q5_0(tensor: &Tensor, block_size: usize) -> Result<QuantizedTensor> {
    // Pack 5-bit values (8 values per 5 bytes)
}
```

### Step 3: Add GGUF Type

```rust
// In export.rs
const GGUF_TYPE_Q5_0: u32 = 8;
```

### Step 4: Test Accuracy

```rust
#[test]
fn test_q5_0_roundtrip() {
    // Verify bounded error
}
```

## Debug Checklist

- [ ] Verify scale computation (absmax vs std-based)
- [ ] Check packing/unpacking byte order
- [ ] Test edge cases (zeros, very large values)
- [ ] Compare with reference implementation
- [ ] Profile memory usage

## Key Files

- `src/quantization.rs` - Core NF4 implementation
- `src/export.rs` - GGUF export
- `tests/quantization_accuracy.rs` - Precision tests
