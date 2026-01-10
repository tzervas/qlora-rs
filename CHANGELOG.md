# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
