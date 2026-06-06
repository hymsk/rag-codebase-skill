#!/usr/bin/env python3
"""
Python 代码解析器 — 基于 tree-sitter 提取函数、类等语义块。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional
from parser.coverage import uncovered_chunks

_ts_parser = None
_ts_language = None

PYTHON_EXTENSIONS = {".py"}

SKIP_DIRS = {
    "test", "tests", "testing", "unittest", "unittests", "unit_test", "unit_tests",
    "benchmark", "benchmarks", "example", "examples", "sample", "samples", "demo", "demos",
    "doc", "docs", "documentation", "third_party", "thirdparty", "3rdparty", "external",
    "vendor", "deps", "build", "dist", "out", "output", ".git", ".svn", ".hg",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".tox", ".venv", "venv", "node_modules",
}

SKIP_FILE_PATTERNS = [
    re.compile(r"test_.*\.py$", re.IGNORECASE),
    re.compile(r".*_test\.py$", re.IGNORECASE),
    re.compile(r".*_benchmark\.py$", re.IGNORECASE),
    re.compile(r".*_example\.py$", re.IGNORECASE),
]


def _init_treesitter():
    global _ts_parser, _ts_language
    if _ts_parser is not None:
        return
    try:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser

        _ts_language = Language(tspython.language())
        _ts_parser = Parser(_ts_language)
    except ImportError:
        raise ImportError(
            "请安装 tree-sitter 和 tree-sitter-python:\n"
            "  uv pip install tree-sitter tree-sitter-python"
        )


def _node_text(node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _find_preceding_comment(node, lines: List[str]) -> str:
    start_idx = node.start_point[0] - 1
    if start_idx < 0:
        return ""

    comments = []
    idx = start_idx
    while idx >= 0:
        line = lines[idx].rstrip()
        if not line:
            if comments:
                break
            idx -= 1
            continue
        stripped = line.lstrip()
        if stripped.startswith("#"):
            comments.insert(0, line)
            idx -= 1
            continue
        break
    return "\n".join(comments)


class PythonParser:
    TOP_LEVEL_TYPES = {"function_definition", "class_definition", "decorated_definition"}
    NODE_TYPE_MAP = {
        "function_definition": "function",
        "class_definition": "class",
        "decorated_definition": "decorated",
    }

    def __init__(
        self,
        max_chunk_chars: int = 2000,
        overlap_chars: int = 200,
        skip_dirs: Optional[set] = None,
        skip_file_patterns: Optional[list] = None,
    ):
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars
        self.skip_dirs = SKIP_DIRS if skip_dirs is None else skip_dirs
        self.skip_file_patterns = SKIP_FILE_PATTERNS if skip_file_patterns is None else skip_file_patterns
        self.extensions = PYTHON_EXTENSIONS
        self.language = "python"

    def should_skip_file(self, file_path: str) -> bool:
        path = Path(file_path)
        if path.suffix.lower() not in self.extensions:
            return True

        parts = set(p.lower() for p in path.parts)
        if parts & self.skip_dirs:
            return True

        name = path.name
        for pattern in self.skip_file_patterns:
            if pattern.search(name):
                return True
        return False

    def parse_file(self, file_path: str) -> List[Dict]:
        _init_treesitter()
        file_path = os.path.abspath(file_path)

        with open(file_path, "rb") as f:
            source_bytes = f.read()

        source_text = source_bytes.decode("utf-8", errors="replace")
        source_lines = source_text.splitlines()

        tree = _ts_parser.parse(source_bytes)
        root = tree.root_node
        chunks = uncovered_chunks(self, root, source_bytes, file_path, self.TOP_LEVEL_TYPES)

        for child in root.children:
            if child.type in self.TOP_LEVEL_TYPES:
                chunks.extend(self._extract_node(child, source_bytes, source_lines, file_path))

        if not chunks and source_text.strip():
            chunks = self._split_plain_text(source_text, file_path, 1)

        return chunks

    def _extract_node(self, node, source_bytes: bytes, source_lines: List[str], file_path: str) -> List[Dict]:
        base_node = node
        if node.type == "decorated_definition":
            for child in node.children:
                if child.type in {"function_definition", "class_definition"}:
                    base_node = child
                    break

        body = _node_text(node, source_bytes)
        node_type = self.NODE_TYPE_MAP.get(base_node.type, "other")
        name_node = base_node.child_by_field_name("name")
        name = _node_text(name_node, source_bytes) if name_node else "<anonymous>"
        doc = _find_preceding_comment(node, source_lines)
        sig = body.splitlines()[0].strip() if body.splitlines() else name

        base_chunk = {
            "type": node_type,
            "name": name,
            "signature": sig,
            "doc": doc,
            "file_path": file_path,
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
            "language": self.language,
        }

        if len(body) <= self.max_chunk_chars:
            base_chunk["body"] = body
            return [base_chunk]

        sub_chunks = self._split_plain_text(body, file_path, node.start_point[0] + 1)
        for sc in sub_chunks:
            sc.update({
                "type": node_type,
                "name": name,
                "signature": sig,
                "doc": doc,
                "language": self.language,
            })
        return sub_chunks

    def _split_plain_text(self, text: str, file_path: str, start_line: int) -> List[Dict]:
        chunks = []
        step = self.max_chunk_chars - self.overlap_chars
        if step <= 0:
            step = self.max_chunk_chars

        for i in range(0, len(text), step):
            segment = text[i: i + self.max_chunk_chars]
            if not segment.strip():
                continue

            lines_before = text[:i].count("\n")
            lines_in = segment.count("\n")
            chunks.append({
                "type": "other",
                "name": "",
                "signature": "",
                "body": segment,
                "doc": "",
                "file_path": file_path,
                "start_line": start_line + lines_before,
                "end_line": start_line + lines_before + lines_in,
                "language": self.language,
            })
        return chunks

    def collect_files(self, root_dir: str) -> List[str]:
        root = Path(root_dir)
        files = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if self.should_skip_file(str(rel)):
                continue
            files.append(str(path))
        return files
