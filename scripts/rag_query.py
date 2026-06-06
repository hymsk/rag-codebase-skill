#!/usr/bin/env python3
"""
RAG 知识库检索器 — 查询 ChromaDB 向量知识库并返回相关代码或文档片段。

使用方式:
    python rag_query.py --repo /path/to/project --query "如何初始化 RGW？"
    python rag_query.py --repo /path/to/project --query "rgw_main 函数" --top-k 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from rag_common import project_id
from rag_store import active_index, operation_lock, validate_owner
from rag_index import _embedding, embedding_fingerprint

RAG_BASE_DIR = os.path.join(os.path.expanduser("~"), ".rag")
# 兼容新旧集合命名：优先多语言名称，其次历史 C++ 名称
COLLECTION_CANDIDATES = ["multilang_codebase", "cpp_codebase"]


def _repo_to_project_id(repo_dir: str) -> str:
    """将仓库绝对路径转换为 project_id，例如 /home/workspace/stor -> home-workspace-stor"""
    return project_id(repo_dir)


def _default_db_path(repo_dir: str) -> str:
    """返回默认的向量数据库路径: $HOME/.rag/rag_db/<project_id>"""
    return os.path.join(RAG_BASE_DIR, "rag_db", _repo_to_project_id(repo_dir))


def init_chromadb_readonly(db_path: str):
    """Open an existing collection; PersistentClient itself is not read-only."""
    import chromadb

    manifest = active_index(db_path)
    if manifest is None:
        raise RuntimeError("No active index generation; build first or migrate a legacy index with --full")
    if not os.path.isfile(os.path.join(db_path, "chroma.sqlite3")):
        print(f"[ERROR] 知识库不存在: {db_path}")
        print("请先使用 rag-codebase 构建知识库。")
        sys.exit(1)

    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))
    ef = _embedding()

    collection = client.get_collection(name=manifest['collection'], embedding_function=ef)
    if collection.count() != sum(record['chunks'] for record in manifest['files'].values()):
        raise RuntimeError('Active generation count mismatch; preserve database for diagnosis')

    return client, collection


def query_knowledge_base(
    db_path: str,
    query_text: str,
    top_k: int = 5,
    filter_file: Optional[str] = None,
    repo_dir: Optional[str] = None,
    download_model: bool = False,
) -> List[Dict]:
    if top_k < 1 or not query_text.strip():
        raise ValueError('top_k must be positive and query must not be empty')
    with operation_lock(db_path):
        if repo_dir is not None:
            validate_owner(repo_dir, db_path)
        from rag_chunking import TokenChunker
        manifest = active_index(db_path)
        if manifest is None:
            raise RuntimeError('No active generation; build first or migrate with --full')
        chunker = TokenChunker.from_default_model(allow_download=download_model)
        fingerprint = dict(chunker.fingerprint, **embedding_fingerprint())
        if any(manifest['fingerprint'].get(key) != value for key, value in fingerprint.items()):
            raise RuntimeError('Embedding fingerprint mismatch; use matching model or rebuild with --full')
        if chunker._count(query_text) > chunker.max_tokens:
            raise ValueError('Query exceeds the embedding token budget; use concise search terms')
        return _query_collection(db_path, query_text, top_k, filter_file)


def _query_collection(db_path, query_text, top_k=5, filter_file=None):
    """
    查询知识库，返回最相关的代码或文档片段。

    返回 list of dict:
        {
            "document": "代码文本",
            "metadata": { "file_path", "type", "name", "start_line", "end_line", ... },
            "distance": float,
        }
    """
    if top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if not query_text.strip():
        raise ValueError("query must not be empty")
    _, collection = init_chromadb_readonly(db_path)

    collection_count = collection.count()
    if collection_count <= 0:
        return []
    candidate_count = min(top_k, collection_count)
    query_options = {}
    if filter_file:
        # Resolve substring matches over metadata BEFORE vector ranking, not
        # over an arbitrary global top-50 shortlist (which loses valid hits).
        paths = set()
        matched_count = 0
        for offset in range(0, collection_count, 1000):
            page = collection.get(limit=1000, offset=offset, include=["metadatas"])
            for metadata in page.get("metadatas") or []:
                path = (metadata or {}).get("file_path", "")
                if filter_file in path:
                    paths.add(path)
                    matched_count += 1
        if not paths:
            return []
        query_options["where"] = {"file_path": {"$in": sorted(paths)}}
        candidate_count = min(top_k, matched_count)

    results = collection.query(
        query_texts=[query_text],
        n_results=candidate_count,
        include=["documents", "metadatas", "distances"],
        **query_options,
    )

    items = []
    if results and results["ids"] and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            item = {
                "id": doc_id,
                "document": results["documents"][0][i] if results["documents"] else "",
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else 0.0,
            }
            if 'source_text' in item['metadata']:
                item['document'] = item['metadata']['source_text']
            if filter_file:
                file_path = item["metadata"].get("file_path", "")
                if filter_file not in file_path:
                    continue
            items.append(item)
            if len(items) >= top_k:
                break

    return items


def format_results_for_ai(results: List[Dict], query: str) -> str:
    """
    将检索结果格式化为适合 AI 阅读的上下文文本。
    """
    if not results:
        return f"未找到与 \"{query}\" 相关的代码/文档片段。"

    fence_map = {
        "cpp": "cpp",
        "c": "c",
        "python": "python",
        "go": "go",
        "markdown": "markdown",
    }

    parts = [f"## 知识库检索结果（查询: \"{query}\"）\n"]
    parts.append(f"共找到 {len(results)} 个相关代码/文档片段:\n")

    for i, item in enumerate(results, 1):
        meta = item.get("metadata", {})
        file_path = meta.get("file_path", "unknown")
        chunk_type = meta.get("type", "unknown")
        language = meta.get("language", "text")
        name = meta.get("name", "")
        start_line = meta.get("start_line", "?")
        end_line = meta.get("end_line", "?")
        signature = meta.get("signature", "")
        distance = item.get("distance", 0)
        similarity = max(0, 1 - distance)

        parts.append(f"### 片段 {i} — {name or file_path}")
        parts.append(f"- 文件: `{file_path}`")
        parts.append(f"- 类型: {chunk_type}")
        parts.append(f"- 语言: {language}")
        line_label = '关联符号行号（不是注释精确位置）' if meta.get('line_range_kind') == 'associated_symbol' else '行号'
        parts.append(f"- {line_label}: {start_line}-{end_line}")
        parts.append(f"- 检索分数: {similarity:.4f}（1 − distance，不是置信概率）")
        if signature:
            parts.append(f"- 签名: `{signature}`")
        fence = fence_map.get(language, "text")
        parts.append(f"\n```{fence}\n{item.get('document', '')}\n```\n")

    return "\n".join(parts)


def format_results_json(results: List[Dict]) -> str:
    """将检索结果格式化为 JSON。"""
    output = []
    for item in results:
        output.append({
            "file_path": item.get("metadata", {}).get("file_path", ""),
            "type": item.get("metadata", {}).get("type", ""),
            "name": item.get("metadata", {}).get("name", ""),
            "start_line": item.get("metadata", {}).get("start_line", 0),
            "end_line": item.get("metadata", {}).get("end_line", 0),
            "line_range_kind": item.get("metadata", {}).get("line_range_kind", "source"),
            "source_field": item.get("metadata", {}).get("source_field", "body"),
            "char_start": item.get("metadata", {}).get("char_start", 0),
            "char_end": item.get("metadata", {}).get("char_end", 0),
            "char_offset_scope": item.get("metadata", {}).get("char_offset_scope", "body"),
            "signature": item.get("metadata", {}).get("signature", ""),
            "language": item.get("metadata", {}).get("language", "unknown"),
            "similarity": max(0, 1 - item.get("distance", 0)),
            "code": item.get("document", ""),
        })
    return json.dumps(output, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="多语言代码知识库检索")
    ap.add_argument("--repo", required=True, help="代码仓库根目录")
    ap.add_argument("--db", default=None, help="向量数据库路径（默认: $HOME/.rag/rag_db/<project_id>）")
    ap.add_argument("--query", required=True, help="检索问题")
    ap.add_argument("--top-k", type=int, default=5, help="返回最相关的 K 个结果（默认: 5）")
    ap.add_argument("--file", default=None, help="过滤指定文件路径（模糊匹配）")
    ap.add_argument("--format", choices=["text", "json"], default="text", help="输出格式")
    ap.add_argument("--download-model", action="store_true", help="Allow model download if uncached")
    args = ap.parse_args()

    if args.top_k < 1 or not args.query.strip():
        ap.error("--top-k must be positive and --query must not be empty")

    db_path = args.db if args.db else _default_db_path(args.repo)

    results = query_knowledge_base(
        db_path=db_path,
        query_text=args.query,
        top_k=args.top_k,
        filter_file=args.file,
        repo_dir=args.repo,
        download_model=args.download_model,
    )

    if args.format == "json":
        print(format_results_json(results))
    else:
        print(format_results_for_ai(results, args.query))


if __name__ == "__main__":
    main()
