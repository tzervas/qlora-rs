//! 4-bit NormalFloat (NF4) quantization.
//!
//! NF4 quantization uses 16 levels optimized for normally-distributed weights,
//! providing better accuracy than uniform 4-bit quantization.
//!
//! Reference: <https://arxiv.org/abs/2305.14314> (QLoRA paper)

use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};

use crate::error::{QLoraError, Result};

/// The 16 quantization levels for NF4, optimized for N(0,1) distribution.
/// These values minimize expected quantization error for normally-distributed data.
pub const NF4_LEVELS: [f32; 16] = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
];

/// Configuration for quantization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Block size for quantization (number of values sharing a scale).
    pub block_size: usize,
    
    /// Whether to use double quantization (quantize the scales).
    pub double_quant: bool,
    
    /// Data type for computation (usually bf16 or f16).
    pub compute_dtype: ComputeDType,
}

/// Compute data type for dequantized values.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub enum ComputeDType {
    /// 32-bit float
    #[default]
    F32,
    /// 16-bit float
    F16,
    /// 16-bit brain float
    BF16,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            block_size: 64,
            double_quant: true,
            compute_dtype: ComputeDType::F32,
        }
    }
}

/// A quantized tensor with scale factors.
#[derive(Debug)]
pub struct QuantizedTensor {
    /// Packed 4-bit values (2 values per byte).
    pub data: Vec<u8>,
    /// Scale factors per block.
    pub scales: Vec<f32>,
    /// Zero points per block (for asymmetric quantization).
    pub zero_points: Option<Vec<f32>>,
    /// Original shape.
    pub shape: Vec<usize>,
    /// Block size used for quantization.
    pub block_size: usize,
}

impl QuantizedTensor {
    /// Get the number of elements in the original tensor.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the memory size in bytes.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.data.len() + self.scales.len() * 4
    }
}

/// Quantize a tensor to NF4 format.
///
/// # Arguments
/// * `tensor` - Input tensor to quantize
/// * `block_size` - Number of elements per quantization block
///
/// # Returns
/// Quantized tensor with packed 4-bit values and scales
///
/// # Errors
/// Returns error if tensor cannot be flattened or has invalid shape
pub fn quantize_nf4(tensor: &Tensor, block_size: usize) -> Result<QuantizedTensor> {
    let shape = tensor.shape().dims().to_vec();
    let flat = tensor.flatten_all()?.to_vec1::<f32>()?;
    let numel = flat.len();

    if numel % block_size != 0 {
        return Err(QLoraError::InvalidConfig(format!(
            "tensor size {} not divisible by block size {}",
            numel, block_size
        )));
    }

    let num_blocks = numel / block_size;
    let mut scales = Vec::with_capacity(num_blocks);
    let mut quantized = Vec::with_capacity((numel + 1) / 2);

    for block_idx in 0..num_blocks {
        let start = block_idx * block_size;
        let end = start + block_size;
        let block = &flat[start..end];

        // Compute absmax scale
        let absmax = block.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = if absmax > 0.0 { absmax } else { 1.0 };
        scales.push(scale);

        // Quantize each value in the block
        for chunk in block.chunks(2) {
            let q0 = quantize_value_nf4(chunk[0] / scale);
            let q1 = if chunk.len() > 1 {
                quantize_value_nf4(chunk[1] / scale)
            } else {
                0
            };
            // Pack two 4-bit values into one byte
            quantized.push((q1 << 4) | q0);
        }
    }

    Ok(QuantizedTensor {
        data: quantized,
        scales,
        zero_points: None,
        shape,
        block_size,
    })
}

/// Dequantize an NF4 tensor back to float.
///
/// # Arguments
/// * `quantized` - Quantized tensor to dequantize
/// * `device` - Device to create the output tensor on
///
/// # Returns
/// Dequantized float tensor with original shape
pub fn dequantize_nf4(quantized: &QuantizedTensor, device: &Device) -> Result<Tensor> {
    let numel = quantized.numel();
    let mut output = Vec::with_capacity(numel);

    let num_blocks = quantized.scales.len();

    for block_idx in 0..num_blocks {
        let scale = quantized.scales[block_idx];
        let start_byte = (block_idx * quantized.block_size) / 2;

        for i in 0..quantized.block_size {
            let byte_idx = start_byte + i / 2;
            let byte = quantized.data[byte_idx];
            let code = if i % 2 == 0 {
                byte & 0x0F
            } else {
                byte >> 4
            };
            let value = NF4_LEVELS[code as usize] * scale;
            output.push(value);
        }
    }

    let tensor = Tensor::from_vec(output, &quantized.shape, device)?;
    Ok(tensor)
}

/// Quantize a single value to NF4 (returns 4-bit code).
fn quantize_value_nf4(value: f32) -> u8 {
    // Find closest NF4 level
    let mut best_idx = 0;
    let mut best_dist = f32::MAX;

    for (idx, &level) in NF4_LEVELS.iter().enumerate() {
        let dist = (value - level).abs();
        if dist < best_dist {
            best_dist = dist;
            best_idx = idx;
        }
    }

    best_idx as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nf4_levels_sorted() {
        for i in 1..NF4_LEVELS.len() {
            assert!(NF4_LEVELS[i] > NF4_LEVELS[i - 1]);
        }
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let device = Device::Cpu;
        let original = Tensor::randn(0.0f32, 1.0, (64,), &device).unwrap();
        
        let quantized = quantize_nf4(&original, 64).unwrap();
        let restored = dequantize_nf4(&quantized, &device).unwrap();

        let original_vec: Vec<f32> = original.to_vec1().unwrap();
        let restored_vec: Vec<f32> = restored.to_vec1().unwrap();

        // Check that error is bounded (NF4 should have <0.5 max error for normalized data)
        for (o, r) in original_vec.iter().zip(restored_vec.iter()) {
            let error = (o - r).abs();
            assert!(error < 0.5, "Error {} too large for value {}", error, o);
        }
    }

    #[test]
    fn test_quantize_preserves_shape() {
        let device = Device::Cpu;
        let original = Tensor::zeros(&[32, 64], DType::F32, &device).unwrap();
        
        let quantized = quantize_nf4(&original, 64).unwrap();
        let restored = dequantize_nf4(&quantized, &device).unwrap();

        assert_eq!(restored.shape().dims(), &[32, 64]);
    }

    #[test]
    fn test_memory_reduction() {
        let device = Device::Cpu;
        let original = Tensor::zeros(&[1024, 1024], DType::F32, &device).unwrap();
        let original_bytes = 1024 * 1024 * 4; // f32 = 4 bytes

        let quantized = quantize_nf4(&original, 64).unwrap();
        let quantized_bytes = quantized.size_bytes();

        // Should be roughly 4x reduction (4-bit vs 32-bit) plus some overhead for scales
        let ratio = original_bytes as f64 / quantized_bytes as f64;
        assert!(ratio > 3.0, "Expected >3x reduction, got {:.2}x", ratio);
    }
}
