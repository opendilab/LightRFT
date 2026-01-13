# GitHub Issue and PR Templates

This directory contains issue templates and pull request templates for the LightRFT project.

## 📋 Issue Templates

The issue templates are numbered to control their display order on GitHub:

### 100-documentation.yml
**Purpose:** Report documentation issues or suggest improvements
- **Use for:** Missing docs, typos, unclear explanations, outdated information
- **Label:** `documentation`

### 200-installation.yml
**Purpose:** Report installation problems
- **Use for:** Errors during pip install, dependency conflicts, environment setup issues
- **Label:** `installation`

### 300-usage.yml
**Purpose:** Ask questions about how to use LightRFT
- **Use for:** General usage questions, configuration help, feature understanding
- **Label:** `usage`
- **Note:** Consider using Discussions for general questions

### 400-bug-report.yml
**Purpose:** Report bugs or unexpected behavior
- **Use for:** Crashes, errors, incorrect outputs, unexpected behavior
- **Label:** `bug`
- **Important:** Include full environment details and reproducible examples

### 500-training-issue.yml
**Purpose:** Report training-specific problems
- **Use for:** OOM errors, NaN losses, convergence issues, distributed training problems
- **Label:** `training`
- **Important:** Include complete training configuration and GPU memory usage

### 600-performance.yml
**Purpose:** Discuss performance improvements or report regressions
- **Use for:** Performance proposals, benchmark sharing, speed optimization discussions
- **Label:** `performance`

### 700-feature-request.yml
**Purpose:** Request new features or enhancements
- **Use for:** New algorithms, model support, API improvements, new capabilities
- **Label:** `enhancement`

## 📝 Pull Request Template

**File:** `PULL_REQUEST_TEMPLATE.md`

The PR template includes sections for:
- Summary and type of change
- Related issues
- Detailed description of changes
- Testing plan and results
- Performance impact analysis
- Documentation updates
- Comprehensive checklist for contributors

## 🎯 Template Features

All templates include:
- ✅ Security warnings about sensitive information
- ✅ Links to search existing issues before creating new ones
- ✅ Structured fields for easy information gathering
- ✅ Required and optional fields appropriately marked
- ✅ Code blocks with syntax highlighting
- ✅ Dropdowns for categorization
- ✅ Checkboxes for confirmations

## 📚 References

These templates are inspired by best practices from:
- [vLLM](https://github.com/vllm-project/vllm) - High-quality issue templates
- GitHub's recommended template structures
- LightRFT-specific needs as a training framework

## 🔧 Customization

To modify templates:
1. Edit the YAML files in `.github/ISSUE_TEMPLATE/`
2. Test changes by creating a new issue/PR
3. Refer to [GitHub's documentation](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests)

## 📞 Contact

For questions about these templates or to suggest improvements, please:
- Open a discussion in [Discussions](https://github.com/opendilab/LightRFT/discussions)
- Contact the maintainers at opendilab@pjlab.org.cn
