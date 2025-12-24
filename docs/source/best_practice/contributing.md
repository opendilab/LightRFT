# Contributing to LightRFT

Thank you for your interest in contributing to LightRFT! This guide will help you get started.

## Ways to Contribute

- 🐛 **Report Bugs**: Open an issue with detailed reproduction steps
- 💡 **Suggest Features**: Propose new features or improvements
- 📝 **Improve Documentation**: Fix typos, add examples, clarify instructions
- 🔧 **Submit Code**: Implement features, fix bugs, optimize performance
- ✅ **Write Tests**: Add test cases for better coverage
- 🌍 **Translate**: Help translate documentation

## Getting Started

1. **Fork the Repository**
   ```bash
   git clone https://github.com/yourusername/LightRFT.git
   cd LightRFT
   ```

2. **Install Development Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-doc.txt  # For documentation
   pip install -e .  # Editable install
   ```

3. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to all public functions/classes
- Keep functions focused and modular

## Pull Request Process

1. **Before Submitting**:
   - Test your changes thoroughly
   - Update documentation if needed
   - Add tests for new features
   - Ensure all tests pass

2. **Submit PR**:
   - Write a clear PR title and description
   - Reference related issues (e.g., "Fixes #123")
   - Request review from maintainers

3. **After Submission**:
   - Respond to review feedback
   - Make requested changes
   - Keep PR updated with main branch

## Documentation

Documentation is built with Sphinx:

```bash
# Build documentation
make docs

# Live preview
make docs-live
```

## Testing

```bash
# Run specific tests
python test_trajectory_saver_fix.py
python test_action_mask_indexing.py

# Add your own tests in tests/ directory
```

## Questions?

- Open a [GitHub Issue](https://github.com/opendilab/LightRFT/issues)
- Join our discussions
- Check the [FAQ](faq.md)

Thank you for contributing to LightRFT! 🎉
