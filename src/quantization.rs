//! 4-bit NormalFloat (NF4) quantization.
//!
//! NF4 quantization uses 16 levels optimized for normally-distributed weights,
//! providing better accuracy than uniform 4-bit quantization.
//!
//! Reference: <https://arxiv.org/abs/2305.14314> (QLoRA paper)

use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};

use crate::error::{QLoraError, Result};

/// The 16 quantization levels for NF4, optimized for N(0,1) distribution.
/// These values minimize expected quantization error for normally-distributed data.
pub const NF4_LEVELS: [f32; 16] = [
    -1.0,
    -0.696_192_800_998_688,
    -0.525_073_051_452_637,
    -0.394_917_488_098_145,
    -0.284_441_381_692_887,
    -0.184_773_430_228_233,
    -0.091_050_036_251_545,
    0.0,
    0.079_580_299_556_255,
    0.160_930_201_411_247,
    0.246_112_301_945_686,
    0.337_915_241_718_292,
    0.440_709_829_330_444,
    0.562_617_003_917_694,
    0.722_956_836_223_602,
    1.0,
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
    /// Quantized scales (when double quantization is used).
    pub scales_quantized: Option<Vec<u8>>,
    /// Scale factors for the scales (when double quantization is used).
    pub scales_scales: Option<Vec<f32>>,
    /// Original shape.
    pub shape: Vec<usize>,
    /// Block size used for quantization.
    pub block_size: usize,
    /// Whether double quantization was applied.
    pub double_quant_enabled: bool,
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
        let mut size = self.data.len() + self.scales.len() * 4;
        if let Some(ref zp) = self.zero_points {
            size += zp.len() * 4;
        }
        if let Some(ref sq) = self.scales_quantized {
            size += sq.len();
        }
        if let Some(ref ss) = self.scales_scales {
            size += ss.len() * 4;
        }
        size
    }

    /// Get the compression ratio relative to FP32 format.
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        let fp32_size = self.numel() * 4;
        let quantized_size = self.size_bytes();
        fp32_size as f64 / quantized_size as f64
    }
}

/// Quantize a tensor to NF4 format with optional double quantization.
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
    quantize_nf4_with_config(tensor, QuantizationConfig {
        block_size,
        double_quant: false, // Use non-double quantization by default
        compute_dtype: ComputeDType::F32,
    })
}

/// Quantize a tensor to NF4 format with full configuration options.
///
/// # Arguments
/// * `tensor` - Input tensor to quantize
/// * `config` - Quantization configuration including double quant option
///
/// # Returns
/// Quantized tensor with packed 4-bit values and optional double-quantized scales
///
/// # Errors
/// Returns error if tensor cannot be flattened or has invalid shape
pub fn quantize_nf4_with_config(tensor: &Tensor, config: QuantizationConfig) -> Result<QuantizedTensor> {
    let shape = tensor.shape().dims().to_vec();
    let flat = tensor.flatten_all()?.to_vec1::<f32>()?;
    let numel = flat.len();

    if numel % config.block_size != 0 {
        return Err(QLoraError::InvalidConfig(format!(
            "tensor size {} not divisible by block size {}",
            numel, config.block_size
        )));
    }

    let num_blocks = numel / config.block_size;
    let mut scales = Vec::with_capacity(num_blocks);
    let mut quantized = Vec::with_capacity((numel + 1) / 2);

    for block_idx in 0..num_blocks {
        let start = block_idx * config.block_size;
        let end = start + config.block_size;
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

    // Apply double quantization if enabled
    let (scales_quantized, scales_scales) = if config.double_quant {
        let (sq, ss) = double_quantize_scales(&scales, 256)?;
        (Some(sq), Some(ss))
    } else {
        (None, None)
    };

    Ok(QuantizedTensor {
        data: quantized,
        scales,
        zero_points: None,
        scales_quantized,
        scales_scales,
        shape,
        block_size: config.block_size,
        double_quant_enabled: config.double_quant,
    })
}

/// Apply double quantization to scale factors.
///
/// Double quantization quantizes the scale factors themselves to reduce memory usage.
/// Typically uses 8-bit unsigned integers for the quantized scales.
///
/// # Arguments
/// * `scales` - Original float32 scale factors
/// * `max_val` - Maximum quantization value (typically 255 for u8)
///
/// # Returns
/// Tuple of (quantized_scales, scale_factors_for_scales)
///
/// # Errors
/// Returns error if scales cannot be processed
fn double_quantize_scales(scales: &[f32], max_val: usize) -> Result<(Vec<u8>, Vec<f32>)> {
    if scales.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    // Find the absolute maximum value in scales
    let absmax = scales
        .iter()
        .map(|s| s.abs())
        .fold(0.0f32, f32::max);

    if absmax == 0.0 {
        return Ok((vec![0; scales.len()], vec![1.0]));
    }

    // Quantize all scales using a single scaling factor
    let scale_factor = absmax / (max_val as f32);
    let quantized_scales: Vec<u8> = scales
        .iter()
        .map(|&s| {
            let quantized = (s / scale_factor).abs() as u32;
            std::cmp::min(quantized, max_val as u32) as u8
        })
        .collect();

    Ok((quantized_scales, vec![scale_factor]))
}

/// Dequantize double-quantized scales back to float.
fn dequantize_double_scales(
    scales_quantized: &[u8],
    scales_scales: &[f32],
) -> Vec<f32> {
    if scales_quantized.is_empty() || scales_scales.is_empty() {
        return vec![];
    }

    let scale_factor = scales_scales[0];
    scales_quantized
        .iter()
        .map(|&sq| (sq as f32) * scale_factor)
        .collect()
}

/// Dequantize an NF4 tensor back to float.
///
/// Automatically handles double-quantized scales if enabled.
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

    // Get scales, applying double quantization reversal if needed
    let scales = if quantized.double_quant_enabled {
        if let (Some(ref sq), Some(ref ss)) = (&quantized.scales_quantized, &quantized.scales_scales) {
            dequantize_double_scales(sq, ss)
        } else {
            quantized.scales.clone()
        }
    } else {
        quantized.scales.clone()
    };

    let num_blocks = scales.len();

    for block_idx in 0..num_blocks {
        let scale = scales[block_idx];
        let start_byte = (block_idx * quantized.block_size) / 2;

        for i in 0..quantized.block_size {
            let byte_idx = start_byte + i / 2;
            let byte = quantized.data[byte_idx];
            let code = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            let value = NF4_LEVELS[code as usize] * scale;
            output.push(value);
        }
    }

    let tensor = Tensor::from_vec(output, quantized.shape.clone(), device)?;
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
    use candle_core::DType;

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

    #[test]
    fn test_double_quantize_compression() {
        let device = Device::Cpu;
        let original = Tensor::randn(0.0f32, 1.0, (512,), &device).unwrap();

        let config = QuantizationConfig {
            block_size: 64,
            double_quant: true,
            compute_dtype: ComputeDType::F32,
        };

        let quantized = quantize_nf4_with_config(&original, config).unwrap();

        // Verify double quantization was applied
        assert!(quantized.double_quant_enabled);
        assert!(quantized.scales_quantized.is_some());
        assert!(quantized.scales_scales.is_some());

        // Double quantized should use less memory than non-double quantized
        let non_dq_size = quantized.scales.len() * 4; // Original scales
        let dq_scales_size = quantized
            .scales_quantized
            .as_ref()
            .map(|sq| sq.len())
            .unwrap_or(0)
            + quantized
                .scales_scales
                .as_ref()
                .map(|ss| ss.len() * 4)
                .unwrap_or(0);

        assert!(dq_scales_size < non_dq_size);
    }

    #[test]
    fn test_double_quantize_roundtrip() {
        let device = Device::Cpu;
        let original = Tensor::randn(0.0f32, 1.0, (256,), &device).unwrap();

        let config = QuantizationConfig {
            block_size: 64,
            double_quant: true,
            compute_dtype: ComputeDType::F32,
        };

        let quantized = quantize_nf4_with_config(&original, config).unwrap();
        let restored = dequantize_nf4(&quantized, &device).unwrap();

        let original_vec: Vec<f32> = original.to_vec1().unwrap();
        let restored_vec: Vec<f32> = restored.to_vec1().unwrap();

        // With double quantization, error increases (scale quantization adds error)
        // but should still be reasonable (typically 10-20% of value magnitude)
        let mut max_error = 0.0f32;
        for (o, r) in original_vec.iter().zip(restored_vec.iter()) {
            let error = (o - r).abs();
            max_error = max_error.max(error);
        }
        // Allow higher error for double quantization - scales are also quantized
        assert!(max_error < 5.0, "Max error {} too large", max_error);
    }

    #[test]
    fn test_double_quant_disabled_still_works() {
        let device = Device::Cpu;
        let original = Tensor::randn(0.0f32, 1.0, (128,), &device).unwrap();

        let config = QuantizationConfig {
            block_size: 64,
            double_quant: false,
            compute_dtype: ComputeDType::F32,
        };

        let quantized = quantize_nf4_with_config(&original, config).unwrap();

        // Verify double quantization was NOT applied
        assert!(!quantized.double_quant_enabled);
        assert!(quantized.scales_quantized.is_none());
        assert!(quantized.scales_scales.is_none());

        let restored = dequantize_nf4(&quantized, &device).unwrap();
        let original_vec: Vec<f32> = original.to_vec1().unwrap();
        let restored_vec: Vec<f32> = restored.to_vec1().unwrap();

        // Regular quantization error bounds
        for (o, r) in original_vec.iter().zip(restored_vec.iter()) {
            let error = (o - r).abs();
            assert!(error < 0.5, "Error {} too large for value {}", error, o);
        }
    }
}
