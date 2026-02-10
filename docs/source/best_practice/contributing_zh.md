# 参与贡献 LightRFT

感谢您对 LightRFT 项目的关注！本指南将帮助您快速开始贡献。

## 贡献方式

- 🐛 **报告 Bug**：提交 Issue 并提供详细的复现步骤。
- 💡 **功能建议**：提出新功能或改进建议。
- 📝 **改进文档**：纠正拼写错误、添加示例或使说明更清晰。
- 🔧 **提交代码**：实现新功能、修复 Bug 或优化性能。
- ✅ **编写测试**：添加测试用例以提高覆盖率。
- 🌍 **翻译文档**：由于项目正在全球化，欢迎各种语言的翻译贡献。

## 快速开始

1. **Fork 代码仓库**
   ```bash
   git clone https://github.com/yourusername/LightRFT.git
   cd LightRFT
   ```

2. **安装开发依赖**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-doc.txt  # 文档构建所需
   pip install -e .  # 以可编辑模式安装
   ```

3. **创建分支**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 代码规范

- Python 代码请遵循 PEP 8 规范。
- 使用具有明确意义的变量和函数名。
- 为所有公共函数和类添加 Docstrings（文档字符串）。
- 保持函数功能单一且模块化。

## Pull Request (PR) 流程

1. **提交前自检**：
   - 彻底测试您的更改。
   - 如有必要，请更新相关文档。
   - 为新功能添加测试用例。
   - 确保所有现有测试均能通过。

2. **提交 PR**：
   - 填写清晰的 PR 标题和描述。
   - 引用相关 Issue（例如："Fixes #123"）。
   - 请求维护者进行 Review。

3. **提交后跟进**：
   - 响应 Review 反馈。
   - 根据建议进行修改。
   - 保持 PR 与主分支同步。

## 文档构建

本项目文档使用 Sphinx 构建：

```bash
# 构建文档
make docs

# 实时预览
make docs-live
```

## 测试

```bash
# 运行特定测试
python test_trajectory_saver_fix.py
python test_action_mask_indexing.py

# 在 tests/ 目录下添加您自己的测试用例
```

## 有疑问？

- 提交 [GitHub Issue](https://github.com/opendilab/LightRFT/issues)
- 加入我们的社区讨论
- 查看 [常见问题解答 (FAQ)](faq_zh.md)

感谢您为 LightRFT 做出贡献！🎉
