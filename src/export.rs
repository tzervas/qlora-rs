//! GGUF export functionality.
//!
//! Export quantized models to GGUF format for inference with llama.cpp and similar tools.

use std::io::Write;
use std::path::Path;

use crate::error::{QLoraError, Result};
use crate::quantization::QuantizedTensor;

/// GGUF file magic number.
const GGUF_MAGIC: u32 = 0x46554747; // "GGUF"

/// GGUF version.
const GGUF_VERSION: u32 = 3;

/// GGUF tensor type for Q4_0.
const GGUF_TYPE_Q4_0: u32 = 2;

/// Export quantized tensors to GGUF format.
///
/// # Arguments
/// * `tensors` - Named quantized tensors to export
/// * `output_path` - Path to write the GGUF file
///
/// # Errors
/// Returns error if file cannot be written or format is invalid
pub fn export_gguf<P: AsRef<Path>>(
    tensors: &[(&str, &QuantizedTensor)],
    output_path: P,
) -> Result<()> {
    let mut file = std::fs::File::create(output_path)?;

    // Write header
    file.write_all(&GGUF_MAGIC.to_le_bytes())?;
    file.write_all(&GGUF_VERSION.to_le_bytes())?;
    file.write_all(&(tensors.len() as u64).to_le_bytes())?;
    file.write_all(&0u64.to_le_bytes())?; // n_kv (metadata)

    // Write tensor info
    for (name, tensor) in tensors {
        write_tensor_info(&mut file, name, tensor)?;
    }

    // Write tensor data
    for (_name, tensor) in tensors {
        file.write_all(&tensor.data)?;
    }

    Ok(())
}

fn write_tensor_info<W: Write>(
    writer: &mut W,
    name: &str,
    tensor: &QuantizedTensor,
) -> Result<()> {
    // Name length and name
    let name_bytes = name.as_bytes();
    writer.write_all(&(name_bytes.len() as u64).to_le_bytes())?;
    writer.write_all(name_bytes)?;

    // Number of dimensions
    writer.write_all(&(tensor.shape.len() as u32).to_le_bytes())?;

    // Dimensions
    for &dim in &tensor.shape {
        writer.write_all(&(dim as u64).to_le_bytes())?;
    }

    // Type (Q4_0 for NF4)
    writer.write_all(&GGUF_TYPE_Q4_0.to_le_bytes())?;

    // Offset (will be computed during actual write)
    writer.write_all(&0u64.to_le_bytes())?;

    Ok(())
}

/// Merge LoRA weights into quantized base and export.
///
/// This dequantizes the base, merges LoRA, and re-quantizes.
/// Useful for deployment without LoRA overhead.
pub fn merge_and_export_gguf<P: AsRef<Path>>(
    _output_path: P,
) -> Result<()> {
    // TODO: Implement merge + export
    Err(QLoraError::GgufExport("merge_and_export not yet implemented".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantization::quantize_nf4;
    use candle_core::{Device, Tensor};
    use std::io::Read;

    #[test]
    fn test_export_gguf_header() {
        let device = Device::Cpu;
        let tensor = Tensor::zeros(&[64, 64], candle_core::DType::F32, &device).unwrap();
        let quantized = quantize_nf4(&tensor, 64).unwrap();

        let temp_path = std::env::temp_dir().join("test_export.gguf");
        export_gguf(&[("test_tensor", &quantized)], &temp_path).unwrap();

        // Verify header
        let mut file = std::fs::File::open(&temp_path).unwrap();
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic).unwrap();
        assert_eq!(u32::from_le_bytes(magic), GGUF_MAGIC);

        std::fs::remove_file(temp_path).ok();
    }
}
