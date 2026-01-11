# Version 0.1.1 Release Notes

**Release Date**: January 10, 2026  
**Tag**: v0.1.1  
**Status**: Alpha - Active Development

---

## Overview

Version 0.1.1 is a maintenance release that adds proper licensing, fixes a critical bug, and improves documentation accuracy throughout the project.

## What's New

### 🔒 Licensing
- Added [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE) files
- Updated [Cargo.toml](Cargo.toml) to properly declare dual `MIT OR Apache-2.0` licensing
- Project is now legally compliant with stated license

### 🐛 Bug Fixes
- **Critical**: Fixed QLoRA batch dimension handling in forward pass
  - Now supports both 2D `[batch, features]` and 3D `[batch, seq, features]` inputs
  - Previously failed on 3D inputs with shape mismatch error
  - All 24 tests now pass (100% coverage) ✅

### 📚 Documentation Improvements
- Corrected false claims in [ANALYSIS.md](ANALYSIS.md) about unimplemented features
- Updated [README.md](README.md) to accurately describe alpha status and capabilities
- Clarified inference vs training support (training planned for Phase 2)
- Changed memory reduction claims from stated fact to "expected/theoretical"
- Fixed peft-rs dependency documentation (crates.io vs local path)
- Added [CHANGELOG.md](CHANGELOG.md) following Keep a Changelog format
- Created [RESOLUTION_SUMMARY.md](RESOLUTION_SUMMARY.md) documenting PR resolution process

### ✨ Other Changes
- Updated version references across all documentation files
- Toned down "production-ready" language to "alpha quality"
- Professional, grounded language throughout (no overstated claims)

---

## Test Coverage

All 24 tests passing (100%):

| Module | Tests | Status |
|--------|-------|--------|
| quantization | 13/13 | ✅ |
| export | 3/3 | ✅ |
| native | 3/3 | ✅ |
| formats | 4/4 | ✅ |
| qlora | 3/3 | ✅ |

---

## Upgrade Notes

### Breaking Changes
None. Version 0.1.1 is fully backward compatible with 0.1.0.

### Dependency Changes
None. All dependencies remain the same:
- candle-core 0.9
- peft-rs 0.4

### API Changes
None. The QLoRA batch dimension fix is internal and doesn't change the public API.

---

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
qlora-rs = "0.1.1"
```

Or use the git repository:

```toml
[dependencies]
qlora-rs = { git = "https://github.com/tzervas/qlora-rs", tag = "v0.1.1" }
```

---

## Known Limitations

- Training support not implemented (planned for Phase 2)
- Model merge functionality not implemented
- Benchmarks not implemented (memory claims are theoretical)
- One dead code warning (config field intentionally unused, reserved for future training)

---

## What's Next

Version 0.2.0 will focus on:
- Training support implementation
- Optimizer integration
- Backward pass for QLoRA layers
- Training loop examples

See [DEVELOPMENT.md](DEVELOPMENT.md) for full roadmap.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for detailed changes between versions.

## License

Dual licensed under MIT OR Apache-2.0. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).

---

**Contributors**: Tyler Zervas and GitHub Copilot  
**Project**: https://github.com/tzervas/qlora-rs
