//! QLoRA layer implementation.
//!
//! Combines quantized base weights with trainable LoRA adapters.

use candle_core::{DType, Device, Tensor};
use peft_rs::{Adapter, LoraConfig, LoraLayer};
use serde::{Deserialize, Serialize};

use crate::error::{QLoraError, Result};
use crate::quantization::{dequantize_nf4, quantize_nf4, QuantizationConfig, QuantizedTensor};

/// Configuration for QLoRA.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QLoraConfig {
    /// LoRA configuration.
    pub lora: LoraConfig,
    /// Quantization configuration.
    pub quantization: QuantizationConfig,
}

impl Default for QLoraConfig {
    fn default() -> Self {
        Self {
            lora: LoraConfig::default(),
            quantization: QuantizationConfig::default(),
        }
    }
}

/// A linear layer with quantized base weights and trainable LoRA adapters.
pub struct QuantizedLinear {
    /// Quantized base weight (frozen).
    quantized_weight: QuantizedTensor,
    /// Optional bias (not quantized).
    bias: Option<Tensor>,
    /// LoRA adapter (trainable).
    lora: LoraLayer,
    /// Configuration.
    config: QLoraConfig,
    /// Device for computation.
    device: Device,
}

impl QuantizedLinear {
    /// Create a new quantized linear layer from existing weights.
    ///
    /// # Arguments
    /// * `weight` - Full-precision weight tensor to quantize
    /// * `bias` - Optional bias tensor (kept in full precision)
    /// * `config` - QLoRA configuration
    /// * `device` - Device for computation
    pub fn from_weight(
        weight: &Tensor,
        bias: Option<Tensor>,
        config: QLoraConfig,
        device: &Device,
    ) -> Result<Self> {
        let shape = weight.shape().dims();
        if shape.len() != 2 {
            return Err(QLoraError::InvalidConfig("weight must be 2D".into()));
        }
        let (out_features, in_features) = (shape[0], shape[1]);

        // Quantize the base weight
        let quantized_weight = quantize_nf4(weight, config.quantization.block_size)?;

        // Create LoRA adapter
        let lora =
            LoraLayer::new_with_zeros(in_features, out_features, config.lora.clone(), device)?;

        Ok(Self {
            quantized_weight,
            bias,
            lora,
            config,
            device: device.clone(),
        })
    }

    /// Create a new quantized linear layer with zero-initialized quantized weights.
    ///
    /// Primarily for testing; use `from_weight` for actual models.
    pub fn new(
        in_features: usize,
        out_features: usize,
        config: QLoraConfig,
        device: &Device,
    ) -> Result<Self> {
        let weight = Tensor::zeros(&[out_features, in_features], DType::F32, device)?;
        Self::from_weight(&weight, None, config, device)
    }

    /// Forward pass through the quantized linear layer.
    ///
    /// Computes: `output = x @ W_q^T + x @ (B @ A)^T * scaling + bias`
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Dequantize base weight for computation
        let weight = dequantize_nf4(&self.quantized_weight, &self.device)?;

        // Base linear: x @ W^T
        let base_output = input.matmul(&weight.t()?)?;

        // LoRA forward: adds x @ A^T @ B^T * scaling
        let output = self.lora.forward(input, Some(&base_output))?;

        // Add bias if present
        match &self.bias {
            Some(bias) => Ok(output.broadcast_add(bias)?),
            None => Ok(output),
        }
    }

    /// Get the LoRA adapter.
    #[must_use]
    pub fn lora(&self) -> &LoraLayer {
        &self.lora
    }

    /// Get mutable access to the LoRA adapter.
    pub fn lora_mut(&mut self) -> &mut LoraLayer {
        &mut self.lora
    }

    /// Get the number of trainable parameters (LoRA only).
    #[must_use]
    pub fn num_trainable_parameters(&self) -> usize {
        self.lora.num_parameters()
    }

    /// Get total memory usage in bytes.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let quantized_size = self.quantized_weight.size_bytes();
        let lora_size = self.lora.num_parameters() * 4; // f32
        let bias_size = self.bias.as_ref().map_or(0, |b| b.elem_count() * 4);
        quantized_size + lora_size + bias_size
    }
}

/// QLoRA adapter wrapping a model's linear layers.
pub struct QLoraLayer {
    /// Underlying quantized linear layer.
    linear: QuantizedLinear,
}

impl QLoraLayer {
    /// Create a new QLoRA layer.
    pub fn new(linear: QuantizedLinear) -> Self {
        Self { linear }
    }

    /// Forward pass.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.linear.forward(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qlora_creation() {
        let config = QLoraConfig::default();
        let device = Device::Cpu;
        let layer = QuantizedLinear::new(768, 768, config, &device);
        assert!(layer.is_ok());
    }

    #[test]
    fn test_qlora_forward_shape() {
        let config = QLoraConfig::default();
        let device = Device::Cpu;
        let layer = QuantizedLinear::new(768, 768, config, &device).unwrap();

        let input = Tensor::zeros(&[1, 10, 768], DType::F32, &device).unwrap();
        let output = layer.forward(&input).unwrap();

        assert_eq!(output.shape().dims(), &[1, 10, 768]);
    }

    #[test]
    fn test_qlora_memory_reduction() {
        let config = QLoraConfig::default();
        let device = Device::Cpu;
        let layer = QuantizedLinear::new(4096, 4096, config, &device).unwrap();

        // Full precision would be 4096 * 4096 * 4 = 67MB
        let full_size = 4096 * 4096 * 4;
        let actual_size = layer.memory_bytes();

        // Should be significantly smaller due to quantization
        let ratio = full_size as f64 / actual_size as f64;
        assert!(ratio > 2.0, "Expected >2x reduction, got {:.2}x", ratio);
    }
}
