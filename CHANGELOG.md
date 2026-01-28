# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.3] - 2026-01-28

### Fixed
- Fixed standalone build by replacing workspace dependencies with explicit versions
- Crates.io publish now works without workspace context

## [1.0.2] - 2026-01-25

### Changed
- Migrated NF4/FP4 GPU kernels to CubeCL 0.9 API
- Kernel position variables now use correct types
- Added proper usize casts at array index sites
- `sync_cube()` replaces deprecated `sync_units()`
- Wrapped kernel launches in unsafe blocks with SAFETY comments

### Known Limitations
- Scale computation kernels assume exact block alignment (documented)
- TILE_SIZE constants defined in multiple places (may consolidate in future)

## [1.0.1] - 2026-01-24

### Added
- CPU fallback warning when CUDA is unavailable

### Changed
- Bumped minimum Rust version to 1.92

## [1.0.0] - 2026-01-24

### Added
- **merge_and_export_gguf()** - Complete LoRA weight merging and GGUF export
  - Dequantizes base weights, merges LoRA adapters, re-quantizes, exports
  - New accessor methods on QLoraLayer: `quantized_weight()`, `lora_weights()`, `lora_scale()`, `device()`, `config()`
- **Examples directory** with 3 runnable examples:
  - `basic_quantization.rs` - NF4 quantization demonstration
  - `qlora_inference.rs` - QLoRA forward pass example
  - `qlora_training.rs` - Training loop setup example
- **Benchmark suite** with Criterion benchmarks:
  - `quantize_nf4_4096` - 4096x4096 tensor quantization
  - `dequantize_nf4_4096` - Dequantization performance
  - `qlora_forward_4096` - Forward pass benchmark
- CLAUDE.md for Claude Code development workflow

### Changed
- Bumped to stable 1.0.0 release
- All tests passing (59 tests: 52 unit + 7 integration)
- Full clippy compliance

### Fixed
- Doc comment formatting for clippy doc_markdown lint

## [0.3.0] - 2026-01-17

### Added
- Complete training infrastructure with `QLoraTrainer` and `QLoraTrainingConfig` modules (769 lines)
- `PagedAdamW` optimizer with memory-efficient GPU/CPU paging for large model training
- `PagedAdamWState` with LRU-based GPU memory management and automatic state eviction
- Full gradient accumulation support with configurable accumulation steps
- Learning rate scheduling with linear warmup and cosine annealing presets
- Comprehensive training configuration system with 10+ parameters and preset builders
- Integration test suite with 7 comprehensive training validation tests
- `from_weight_with_varbuilder()` method for training-aware layer initialization
- Accessor methods `lora_weights()` and `lora_shapes()` for parameter inspection
- Training support for both on-the-fly and cached dequantization modes

### Changed
- Updated peft-rs dependency to use dev branch (includes custom weights() methods)
- Upgraded thiserror from 1.0 to 2.0 for improved error handling
- Upgraded criterion from 0.5 to 0.8 for enhanced benchmark infrastructure
- Modernized GitHub Actions: checkout@v4 → v6, rust-cache to v2.8.2 latest
- Replaced deprecated rustsec/audit-check-action with manual cargo-audit
- Improved merge conflict resolution from PR #10 code review

### Fixed
- Removed broken submodule entries (gemm-fork, paste-fork) from git tracking
- Fixed duplicate method definitions from merge conflict resolution
- Resolved 5 critical CI/CD failures (formatting, clippy, tests, benchmarks, security audit)
- Updated security audit handling for RUSTSEC-2024-0436 compatibility

### Test Coverage
- Expanded to 57 total tests: 50 unit tests + 7 integration tests (100% pass rate)
- Added comprehensive training integration tests covering:
  - Standard optimizer weight training
  - Paged optimizer memory tracking and limits
  - Optimizer state management and LRU eviction
  - Full training loop with gradient accumulation
  - VarBuilder requirement validation

## [0.1.1] - 2026-01-10

### Added
- Dual MIT OR Apache-2.0 licensing with proper LICENSE files ([LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE))
- Comprehensive PR resolution documentation ([RESOLUTION_SUMMARY.md](RESOLUTION_SUMMARY.md))
- Support for batch dimensions (2D and 3D inputs) in QLoRA forward pass
- CHANGELOG.md for tracking version changes

### Fixed
- QLoRA batch dimension handling bug (test_qlora_forward_shape now passes)
- Documentation inaccuracies in [ANALYSIS.md](ANALYSIS.md) (corrected false claims about unimplemented features)
- Documentation inaccuracies in [README.md](README.md) (clarified alpha status, training vs inference)
- peft-rs dependency documentation (now correctly shows crates.io source)

### Changed
- Updated [Cargo.toml](Cargo.toml) license field from "MIT" to "MIT OR Apache-2.0"
- Improved [README.md](README.md) to accurately reflect inference capabilities vs training (planned)
- Updated all documentation to reflect 100% test pass rate (24/24 tests)
- Toned down "production-ready" language to "alpha quality" throughout documentation
- PR documentation marked as historical reference after fast-track merge

### Test Coverage
- All 24 tests passing (100% coverage)
  - quantization: 13/13 ✅
  - export: 3/3 ✅
  - native: 3/3 ✅
  - formats: 4/4 ✅
  - qlora: 3/3 ✅

## [0.1.0-alpha] - 2026-01-09

### Added
- Initial alpha release with core NF4 quantization
- Double quantization for scale compression (40% reduction)
- Per-channel and per-tensor quantization strategies
- Zero-point asymmetric quantization
- Dual export formats: GGUF (llama.cpp compatible) and Candle native (QNAT)
- QLoRA inference layer with LoRA adapter support
- Comprehensive test suite (23/24 tests passing at time of release)
- CI/CD pipeline with format, lint, test, and security checks
- Development workflow documentation

### Known Issues
- QLoRA batch dimension handling bug (fixed in 0.1.1)
- Training support not implemented (planned for Phase 2)
- Model merge functionality not implemented

## [0.1.0] - 2026-01-09

Initial scaffold and infrastructure setup.
