//! Training utilities for `QLoRA` fine-tuning.
//!
//! This module provides:
//! - [`QLoraTrainer`] - Main trainer for `QLoRA` fine-tuning
//! - Integration with peft-rs training state and LR schedules
//! - Gradient computation and optimizer integration

use candle_core::{Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};
use peft_rs::training::{AdapterTrainingConfig, AdapterTrainingState, LrSchedule};

use crate::error::Result;
use crate::qlora::QuantizedLinear;

/// Configuration for `QLoRA` training.
#[derive(Debug, Clone)]
pub struct QLoraTrainingConfig {
    /// Adapter training configuration (from peft-rs).
    pub adapter_config: AdapterTrainingConfig,
    /// Number of training epochs.
    pub num_epochs: usize,
    /// Batch size for training.
    pub batch_size: usize,
    /// Logging frequency (steps).
    pub log_every: usize,
    /// Checkpoint save frequency (steps, None = no checkpoints).
    pub save_every: Option<usize>,
    /// Warmup steps for learning rate.
    pub warmup_steps: usize,
}

impl Default for QLoraTrainingConfig {
    fn default() -> Self {
        Self {
            adapter_config: AdapterTrainingConfig {
                learning_rate: 2e-4,
                lr_schedule: LrSchedule::LinearWarmup { warmup_steps: 100 },
                weight_decay: 0.01,
                gradient_accumulation_steps: 4,
                max_grad_norm: Some(1.0),
            },
            num_epochs: 3,
            batch_size: 4,
            log_every: 10,
            save_every: Some(500),
            warmup_steps: 100,
        }
    }
}

/// Trainer for `QLoRA` fine-tuning.
///
/// Manages the training loop, gradient computation, and optimizer updates
/// for quantized `LoRA` training.
pub struct QLoraTrainer {
    /// Training configuration.
    config: QLoraTrainingConfig,
    /// Training state tracking.
    state: AdapterTrainingState,
    /// Device for computation.
    #[allow(dead_code)]
    device: Device,
    /// Variable map for trainable parameters.
    varmap: VarMap,
    /// Optimizer.
    optimizer: Option<AdamW>,
}

impl QLoraTrainer {
    /// Create a new `QLoRA` trainer.
    ///
    /// # Arguments
    /// * `config` - Training configuration
    /// * `device` - Device for computation
    ///
    /// # Returns
    /// New trainer instance
    #[must_use]
    pub fn new(config: QLoraTrainingConfig, device: Device) -> Self {
        let state = AdapterTrainingState::new(config.adapter_config.clone());
        Self {
            config,
            state,
            device,
            varmap: VarMap::new(),
            optimizer: None,
        }
    }

    /// Initialize the optimizer with trainable parameters.
    ///
    /// # Arguments
    /// * `layers` - The `QLoRA` layers to train
    ///
    /// # Errors
    /// Returns error if optimizer initialization fails
    pub fn init_optimizer(&mut self, _layers: &[&QuantizedLinear]) -> Result<()> {
        // Get all trainable tensors from the varmap
        let params = ParamsAdamW {
            lr: self.config.adapter_config.learning_rate,
            weight_decay: self.config.adapter_config.weight_decay,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        };

        self.optimizer = Some(AdamW::new(self.varmap.all_vars(), params)?);
        Ok(())
    }

    /// Get the current training state.
    #[must_use]
    pub fn state(&self) -> &AdapterTrainingState {
        &self.state
    }

    /// Get the current learning rate.
    #[must_use]
    pub fn current_lr(&self) -> f64 {
        self.state.current_lr()
    }

    /// Get the current step.
    #[must_use]
    pub fn global_step(&self) -> usize {
        self.state.global_step
    }

    /// Get the current epoch.
    #[must_use]
    pub fn epoch(&self) -> usize {
        self.state.epoch
    }

    /// Perform a training step with loss computation.
    ///
    /// # Arguments
    /// * `layers` - The `QLoRA` layers
    /// * `input` - Input tensor `[batch, seq_len, hidden]`
    /// * `targets` - Target logits `[batch, seq_len, vocab]`
    ///
    /// # Returns
    /// The loss value
    ///
    /// # Errors
    /// Returns error if forward pass or backward pass fails
    pub fn training_step(
        &mut self,
        layers: &[&QuantizedLinear],
        input: &Tensor,
        targets: &Tensor,
    ) -> Result<f64> {
        // Forward pass through all layers
        let mut output = input.clone();
        for layer in layers {
            output = layer.forward(&output)?;
        }

        // Compute MSE loss for demonstration (in practice, use cross-entropy)
        let loss = output.sub(targets)?.sqr()?.mean_all()?;
        let loss_value = loss.to_scalar::<f32>()? as f64;

        // Backward pass
        if let Some(ref mut optimizer) = self.optimizer {
            optimizer.backward_step(&loss)?;
        }

        // Update training state
        let should_log = self.state.step();
        if should_log && self.state.global_step % self.config.log_every == 0 {
            #[cfg(feature = "logging")]
            log::info!(
                "Step {} | Loss: {:.4} | LR: {:.2e}",
                self.state.global_step,
                loss_value,
                self.current_lr()
            );
        }

        Ok(loss_value)
    }

    /// Start a new training epoch.
    pub fn start_epoch(&mut self) {
        self.state.new_epoch();
        #[cfg(feature = "logging")]
        log::info!("Starting epoch {}", self.state.epoch);
    }

    /// Check if training should continue.
    #[must_use]
    pub fn should_continue(&self) -> bool {
        self.state.epoch < self.config.num_epochs
    }

    /// Update learning rate based on schedule.
    pub fn update_lr(&mut self) {
        let lr = self.current_lr();
        if let Some(ref mut optimizer) = self.optimizer {
            optimizer.set_learning_rate(lr);
        }
    }

    /// Get training configuration.
    #[must_use]
    pub fn config(&self) -> &QLoraTrainingConfig {
        &self.config
    }
}

/// Compute cross-entropy loss for language modeling.
///
/// # Arguments
/// * `logits` - Model output logits `[batch, seq_len, vocab_size]`
/// * `targets` - Target token IDs `[batch, seq_len]`
///
/// # Returns
/// Cross-entropy loss value
///
/// # Errors
/// Returns error if tensor operations fail
pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let (batch, seq_len, vocab_size) = logits.dims3()?;

    // Reshape logits to [batch * seq_len, vocab_size]
    let flat_logits = logits.reshape(&[batch * seq_len, vocab_size])?;

    // Reshape targets to [batch * seq_len]
    let flat_targets = targets.reshape(&[batch * seq_len])?;

    // Compute log softmax
    let log_probs = candle_nn::ops::log_softmax(&flat_logits, 1)?;

    // Gather log probs at target indices
    let target_indices = flat_targets.unsqueeze(1)?;
    let gathered = log_probs.gather(&target_indices, 1)?;

    // Mean negative log likelihood
    let loss = gathered.neg()?.mean_all()?;

    Ok(loss)
}

/// Training metrics for logging.
#[derive(Debug, Clone, Default)]
pub struct TrainingMetrics {
    /// Total training loss.
    pub total_loss: f64,
    /// Number of steps.
    pub num_steps: usize,
    /// Best loss seen.
    pub best_loss: f64,
    /// Tokens processed.
    pub tokens_processed: usize,
}

impl TrainingMetrics {
    /// Create new metrics tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_loss: 0.0,
            num_steps: 0,
            best_loss: f64::MAX,
            tokens_processed: 0,
        }
    }

    /// Update metrics with a new loss value.
    pub fn update(&mut self, loss: f64, num_tokens: usize) {
        self.total_loss += loss;
        self.num_steps += 1;
        self.tokens_processed += num_tokens;
        if loss < self.best_loss {
            self.best_loss = loss;
        }
    }

    /// Get average loss.
    #[must_use]
    pub fn average_loss(&self) -> f64 {
        if self.num_steps == 0 {
            0.0
        } else {
            self.total_loss / self.num_steps as f64
        }
    }

    /// Reset metrics for new epoch.
    pub fn reset(&mut self) {
        self.total_loss = 0.0;
        self.num_steps = 0;
        self.tokens_processed = 0;
        // Keep best_loss across epochs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_training_config_default() {
        let config = QLoraTrainingConfig::default();
        assert_eq!(config.num_epochs, 3);
        assert_eq!(config.batch_size, 4);
        assert!((config.adapter_config.learning_rate - 2e-4).abs() < 1e-10);
    }

    #[test]
    fn test_trainer_creation() {
        let config = QLoraTrainingConfig::default();
        let device = Device::Cpu;
        let trainer = QLoraTrainer::new(config, device);

        assert_eq!(trainer.global_step(), 0);
        assert_eq!(trainer.epoch(), 0);
    }

    #[test]
    fn test_training_metrics() {
        let mut metrics = TrainingMetrics::new();

        metrics.update(0.5, 128);
        metrics.update(0.4, 128);
        metrics.update(0.3, 128);

        assert_eq!(metrics.num_steps, 3);
        assert!((metrics.average_loss() - 0.4).abs() < 1e-10);
        assert!((metrics.best_loss - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_cross_entropy_loss_shape() {
        let device = Device::Cpu;
        let batch = 2;
        let seq_len = 10;
        let vocab_size = 100;

        let logits = Tensor::zeros(&[batch, seq_len, vocab_size], DType::F32, &device).unwrap();
        // Random targets (0-99)
        let targets = Tensor::zeros(&[batch, seq_len], DType::U32, &device).unwrap();

        let loss = cross_entropy_loss(&logits, &targets).unwrap();
        // Loss should be scalar
        let dims: &[usize] = loss.dims();
        assert!(dims.is_empty(), "Expected scalar loss, got dims: {dims:?}");
    }

    #[test]
    fn test_trainer_epoch_progression() {
        let config = QLoraTrainingConfig {
            num_epochs: 2,
            ..Default::default()
        };
        let device = Device::Cpu;
        let mut trainer = QLoraTrainer::new(config, device);

        assert!(trainer.should_continue());
        trainer.start_epoch();
        assert_eq!(trainer.epoch(), 1);
        assert!(trainer.should_continue());
        trainer.start_epoch();
        assert_eq!(trainer.epoch(), 2);
        assert!(!trainer.should_continue());
    }
}
