<!--
Thank you for contributing to LightRFT!
Please fill in the PR description below, ensuring all checklist items have been considered.
-->

## 📋 Summary

<!-- Provide a clear and concise description of your changes -->

**Purpose:** <!-- e.g., Fix bug #123, Add support for new algorithm, Improve performance -->

**Type of Change:**
- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] 🎨 Code refactoring (no functional changes)
- [ ] ⚡ Performance improvement
- [ ] ✅ Test addition/modification
- [ ] 🔧 Configuration/Build changes

## 🔗 Related Issues

<!-- Link to related issues, discussions, or PRs -->

Fixes #(issue number)
Related to #(issue number)

## 📝 Changes

<!-- Describe your changes in detail -->

### What changed:
-
-
-

### Why these changes:
-
-

### Key implementation details:
-
-

## 🧪 Testing

### Test Plan

<!-- Describe how you tested your changes -->

- [ ] **Unit tests:** Added/updated unit tests
- [ ] **Integration tests:** Tested with full training pipeline
- [ ] **Manual testing:** Describe what you tested manually

**Test commands:**
```bash
# Commands used to test the changes
```

**Test environment:**
- Python Version:
- PyTorch Version:
- CUDA Version:
- GPU Model:
- Number of GPUs:

### Test Results

<!-- Paste test results, benchmarks, or comparison data -->

<details>
<summary>Test Output</summary>

```
Paste test output here
```

</details>

**Before this PR:**
```
# Baseline metrics/behavior
```

**After this PR:**
```
# New metrics/behavior
```

## 📊 Performance Impact

<!-- If applicable, describe the performance impact -->

- [ ] No performance impact
- [ ] Performance improved: <!-- describe improvement -->
- [ ] Performance regression: <!-- describe and justify -->

**Benchmark results (if applicable):**
```
Baseline: X samples/sec, Y GB memory
After PR: X samples/sec, Y GB memory
```

## 📚 Documentation

<!-- Check all that apply -->

- [ ] Docstrings updated for new/modified functions
- [ ] README.md updated (if user-facing changes)
- [ ] Documentation in `docs/` updated (if applicable)
- [ ] Examples updated/added (if applicable)
- [ ] Configuration reference updated (if new parameters added)
- [ ] CHANGELOG.md updated

## ✅ Checklist

<!-- Please check all items before submitting -->

### Code Quality
- [ ] Code follows the project's style guidelines (run `make format` and `make fcheck`)
- [ ] Self-review of code completed
- [ ] Code is well-commented, especially in complex areas
- [ ] No unnecessary debug logs or commented-out code

### Compatibility
- [ ] Changes are backward compatible (or breaking changes are documented)
- [ ] Existing tests pass with changes
- [ ] No new warnings introduced

### Testing
- [ ] Tested with FSDP (if applicable)
- [ ] Tested with DeepSpeed (if applicable)
- [ ] Tested with inference engines (vLLM/SGLang) (if applicable)
- [ ] Tested on multiple GPU configurations (if applicable)

### Documentation
- [ ] All public APIs are documented
- [ ] User-facing changes are documented
- [ ] Migration guide provided (if breaking changes)

## 🎯 Algorithm/Model Specific (if applicable)

<!-- Fill this section if adding new algorithm or model support -->

**New Algorithm:**
- [ ] Algorithm implementation follows existing patterns
- [ ] Algorithm is configurable via CLI arguments
- [ ] Example training script provided
- [ ] Algorithm documentation added to `docs/source/quick_start/algorithms.md`

**New Model Support:**
- [ ] Model architecture properly integrated
- [ ] Tested with representative datasets
- [ ] Model-specific documentation added

## 💭 Additional Notes

<!-- Any additional information, concerns, or discussion points -->

## 🔍 Review Checklist for Maintainers

<!-- For maintainer use -->

- [ ] Code quality and style verified
- [ ] Tests are adequate and passing
- [ ] Documentation is complete and clear
- [ ] Performance impact is acceptable
- [ ] Breaking changes are properly documented
- [ ] Ready to merge

---

**BEFORE SUBMITTING, PLEASE READ:**
- [Contributing Guide](https://github.com/opendilab/LightRFT/tree/main/docs/source/best_practice/contributing.md)
- [Code Style Guidelines](https://github.com/opendilab/LightRFT/tree/main/docs/source/best_practice/contributing.md)

<!-- Anything below this line will be visible in the PR -->
