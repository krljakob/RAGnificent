# RAGnificent CI/CD Pipeline Documentation

This document provides an overview of the CI/CD pipelines for the RAGnificent project, explaining the purpose of each workflow and how to use them.

## Overview

The CI/CD pipeline has been completely rewritten with a focus on:

- Rust-first development approach
- Multiple platform testing (Linux and Windows)
- Comprehensive code quality checks
- Performance benchmarking
- Security scanning
- Automated releases
- Documentation generation

## Workflow Files

### 1. `ragnificent-ci.yml`

The main continuous integration workflow that runs on every push to main and pull requests.

**Jobs:**
- **rust-checks**: Code quality checks for Rust (formatting, linting)
- **rust-tests**: Unit tests for Rust components on both Linux and Windows
- **python-checks**: Code quality checks for Python (ruff, mypy)
- **python-tests**: Unit tests for Python components on both Linux and Windows
- **integration-tests**: End-to-end tests that exercise the full system

**When it runs:**
- On push to main branch
- On pull requests to main
- Manually via workflow dispatch

### 2. `ragnificent-release.yml`

Handles the creation of releases, building distribution packages, and publishing to PyPI.

**Jobs:**
- **create-release**: Creates a GitHub Release
- **build-wheels**: Builds Python wheels for multiple platforms
- **publish-pypi**: Publishes packages to PyPI
- **build-docs**: Generates and publishes documentation

**When it runs:**
- On tag push with 'v*' pattern (e.g., v1.0.0)
- Manually via workflow dispatch

### 3. `ragnificent-benchmark.yml`

Runs performance benchmarks and tracks changes over time.

**Jobs:**
- **rust-benchmarks**: Runs Criterion benchmarks for Rust code
- **python-benchmarks**: Runs pytest-benchmark tests for Python code

**When it runs:**
- On push to main branch
- On pull requests to main
- Weekly on Monday at 3 AM UTC
- Manually via workflow dispatch

### 4. `ragnificent-security.yml`

Scans dependencies for security vulnerabilities.

**Jobs:**
- **rust-security-audit**: Uses cargo-audit and cargo-deny to check Rust dependencies
- **python-security-audit**: Uses safety and bandit to check Python code and dependencies

**When it runs:**
- When dependency files change (Cargo.toml, requirements.txt, etc.)
- Weekly on Monday at 6 AM UTC
- Manually via workflow dispatch

### 5. `ragnificent-docs.yml`

Builds and publishes documentation.

**Jobs:**
- **build-docs**: Generates API documentation and builds a documentation site

**When it runs:**
- When documentation or source code files change
- Manually via workflow dispatch

## Workflow Integration

These workflows are designed to work together:

1. **Development Cycle**:
   - Make changes to the codebase
   - Push to a branch and create a PR
   - The CI pipeline automatically runs tests and checks
   - Merge to main once CI passes

2. **Release Process**:
   - Create and push a tag (e.g., `git tag v1.0.0 && git push --tags`)
   - The release workflow automatically:
     - Creates a GitHub release
     - Builds distribution packages
     - Publishes to PyPI
     - Updates documentation

## Configuration

### Secrets Required

The workflows use the following GitHub secrets:

- `GITHUB_TOKEN` (automatically provided)
- `PYPI_API_TOKEN` (needed for PyPI publishing)

### Environment Variables

The main configuration happens in the workflow files themselves, but you can customize behavior with:

- Branch protection rules on GitHub
- Selective workflow triggers in workflow files

## Local Development

For local development, the CI workflows align with these commands:

```bash
# Rust checks (from project root)
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test

# Python checks (from project root)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
uv pip install -e .[dev,test]
ruff check RAGnificent/ tests/
mypy RAGnificent/
pytest

# Build Rust extension
maturin build --release
maturin develop --release
```

## Best Practices

1. **Commit Messages**: Use clear, descriptive commit messages to help with release changelogs.
2. **Branch Protection**: Enable branch protection on the main branch to ensure CI passes before merging.
3. **Versioning**: Follow semantic versioning for releases.
4. **Pull Requests**: Include tests for new features and bug fixes.
5. **Documentation**: Update documentation along with code changes.

## Troubleshooting

If CI workflows fail:

1. Check the specific job that failed in the GitHub Actions interface
2. Look at the error messages and logs
3. Run the same commands locally to reproduce the issue
4. Fix the issue and push changes

Common issues:
- Formatting errors (run `cargo fmt` and `black`)
- Linting errors (run `cargo clippy` and `ruff`)
- Type checking errors (run `mypy`)
- Test failures (run the specific failing test locally)

## Customization

To customize the CI/CD pipeline:

1. Edit the workflow files in `.github/workflows/`
2. Commit and push changes
3. The updated workflows will be used for future runs

## Monitoring

- GitHub Actions dashboard shows all workflow runs
- Benchmark results are published to GitHub Pages
- Security scan results are uploaded as artifacts
