# Development Guide - QLoRA Rust Port

## Branch Strategy

This project follows a strict branching model for quality and release management:

```
feature/* / fix/* / refactor/*  (working branches)
    ↓ (PR)
dev (integration branch)
    ↓ (PR)
testing (staging/pre-release)
    ↓ (PR)
main (production/release)
```

### Branch Types

- **`main`**: Production-ready releases. Tagged with semantic versions (e.g., `v0.1.0`).
- **`testing`**: Pre-release staging. Testing and QA before merging to main.
- **`dev`**: Integration branch for features. All working branches target `dev`.
- **`feature/*`**: New feature implementation. Naming: `feature/description-of-feature`
- **`fix/*`**: Bug fixes. Naming: `fix/issue-description`
- **`refactor/*`**: Code refactoring. Naming: `refactor/area-of-change`

## Commit Message Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): subject line
(blank line)
body (optional)
(blank line)
footer (optional, for references like fixes #123)
```

### Commit Types

- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code refactoring (no functional change)
- `docs`: Documentation changes
- `test`: Test additions/modifications
- `perf`: Performance improvements
- `style`: Code style/formatting
- `chore`: Build, dependencies, or tooling

### Examples

```
feat(export): implement Candle native format export

Added support for Candle native quantized format alongside GGUF.
Includes manual format selection API with auto-detect for future.

feat(quantization): implement double quantization

fix(gguf): correct tensor offset calculations in GGUF export

docs(api): add export format selection documentation

perf(dequantization): optimize forward pass dequantization
```

## Pull Request (PR) Workflow

### Before Submitting a PR

1. **Ensure code quality**:
   ```bash
   cargo fmt --all
   cargo clippy --all-targets --all-features
   cargo test --all
   cargo doc --all --no-deps
   ```

2. **Check for security issues**:
   ```bash
   cargo audit
   ```

3. **Run benchmarks** (if relevant):
   ```bash
   cargo bench
   ```

4. **Verify PR is ready**:
   - [ ] All tests pass locally
   - [ ] Clippy shows no warnings
   - [ ] Code formatted with `rustfmt`
   - [ ] Documentation updated
   - [ ] Commit messages follow conventions
   - [ ] No security vulnerabilities

### PR Title and Description

**Title**: Follow conventional commits format.

**Description**: Include:
- What does this PR do?
- Why is this change needed?
- How were changes tested?
- Any breaking changes?
- Closes/Fixes references (e.g., `Closes #42`)

**Mark as Ready**: Only create the PR when all checks pass and code is ready for review.

### Example PR Title

```
feat(export): implement dual GGUF/Candle native format support

Implements both GGUF and Candle native quantized format exports with
manual format selection (auto-detect planned for v0.2).

- Add ExportFormat enum for format selection
- Implement GGUF exporter with fixed tensor offsets
- Implement Candle native binary format exporter
- Add integration tests for both formats

Closes #15
```

## Code Quality Standards

### Linting

- **Clippy**: Must pass with zero warnings
  ```bash
  cargo clippy --all-targets --all-features -- -D warnings
  ```

- **Format**: Must match rustfmt output
  ```bash
  cargo fmt --all
  ```

### Testing

- Minimum 80% code coverage for critical paths
- All public APIs must have tests
- Integration tests must validate against Python reference implementation

### Documentation

- All public items must have documentation comments (`///`)
- Examples in docstrings for complex types/functions
- SAFETY comments required for all `unsafe` blocks

### Security

- No dependencies with known CVEs
- `cargo audit` must pass
- Minimize `unsafe` code usage

## Versioning

This project follows [Semantic Versioning 2.0.0](https://semver.org/).

### Current Version: 0.1.0

- **0.x.y**: Pre-release development. Breaking changes allowed.
- **1.0.0**: First stable release (blocked until all Phase features complete).

### Release Process

1. Update `Cargo.toml` version
2. Update `CHANGELOG.md` with changes
3. Create annotated git tag: `git tag -a v0.1.0 -m "Release 0.1.0"`
4. Push tag: `git push origin v0.1.0`
5. GitHub Actions publishes to crates.io automatically

## Development Phases

### Phase 1: Dual Export Infrastructure (Current)
- [ ] Fix GGUF export
- [ ] Implement double quantization
- [ ] Design/implement Candle native format
- [ ] Add format selection API
- [ ] Integration tests

### Phase 2: Training Support
- [ ] Gradient computation
- [ ] Optimizer support
- [ ] Training loop
- [ ] Checkpointing

### Phase 3: Performance Optimization
- [ ] Lazy dequantization
- [ ] Vectorization
- [ ] Benchmarking

### Phase 4: Model Management & Release Prep
- [ ] Model save/load
- [ ] Documentation
- [ ] Release notes

## Useful Commands

```bash
# Format code
cargo fmt --all

# Run linter
cargo clippy --all-targets --all-features

# Run tests with output
cargo test -- --nocapture

# Generate documentation
cargo doc --all --open

# Security audit
cargo audit

# Benchmark
cargo bench

# Clean build
cargo clean && cargo build

# Check (without building)
cargo check --all
```

## CI/CD Pipeline

GitHub Actions automatically runs on all pushes and PRs:
- ✅ Code formatting check
- ✅ Clippy linting
- ✅ Security audit
- ✅ Unit tests
- ✅ Benchmark compilation
- ✅ Documentation generation

All checks must pass before merging to `main`.

## Questions?

Refer to the project README or open an issue on GitHub.
