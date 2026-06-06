# rag-codebase

用于本地代码库和 Markdown 文档语义检索的 AI Skill，支持 C/C++、Python、Go 和 Markdown。基于 tree-sitter、本地嵌入模型和 ChromaDB 构建索引，并通过内容哈希进行增量更新，无需独立服务。

## 快速开始

需要 Python 3.10+ 和支持 `flock` 的 POSIX 本地文件系统，不支持原生 Windows。在本项目目录执行，将 `/path/to/repo` 替换为目标仓库路径：

```bash
# 安装依赖并激活虚拟环境
bash scripts/install-chroma.sh
. "$HOME/.rag/.venv/bin/activate"

# 预览索引计划，不创建数据库或下载模型
python scripts/rag_builder.py --repo /path/to/repo --plan

# 构建或增量更新索引（默认仅使用已缓存模型）
python scripts/rag_builder.py --repo /path/to/repo

# 检索代码或文档
python scripts/rag_query.py --repo /path/to/repo \
  --query "authentication middleware" --top-k 8
```

## 注意事项

- 模型未缓存时，需明确同意下载后，在构建命令中添加 `--download-model`。
- 索引默认保存在 `~/.rag/rag_db/` 下，包含源码原文，请保持私有；自定义 `--db` 必须位于目标仓库之外，构建和查询时保持一致。
- 语义检索不保证结果完整；精确符号、字面量或穷举搜索请使用 `rg`，引用结果前核对当前源码。
- 安装脚本仅安装依赖，不会向 AI 宿主注册 Skill。完整使用流程和边界见 [SKILL.md](SKILL.md)。
