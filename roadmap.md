# QLoRA-RS Roadmap

## Success Criteria
Full QLoRA training (e.g., fine-tune LLaMA-7B on Alpaca with 4-bit weights), matching Python performance.

## Requirements
- Integrate training loop with peft-rs/unsloth-rs.
- Add paged optimizers.

## Deliverables
- Training module.
- Paged optimizer.
- Multi-GPU support.

## Remaining Tasks
- **Phase 2**: Add training loop.
- **Phase 3**: Paged optimizers and advanced quantization.
- **Phase 4**: Full parity with Python QLoRA.