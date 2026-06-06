#!/usr/bin/env python3
"""
C++ 代码解析器 — 基于 tree-sitter 提取函数、类、结构体等语义块。

使用方式:
    from parser.cpp import CppParser
    parser = CppParser()
    chunks = parser.parse_file("/path/to/file.cpp")

每个 chunk 是一个 dict:
    {
        "type":       "function" | "class" | "struct" | "namespace" | "enum" | "macro" | "other",
        "name":       "函数/类名",
        "signature":  "完整签名（含参数、返回值）",
        "body":       "完整代码文本",
        "doc":        "注释/文档（如有）",
        "start_line": 起始行号,
        "end_line":   结束行号,
        "file_path":  "源文件路径",
    }
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Dict, Optional
from parser.coverage import uncovered_chunks

# ---------------------------------------------------------------------------
# tree-sitter 懒加载（首次调用时初始化）
# ---------------------------------------------------------------------------
_ts_parser = None
_ts_language = None

# C++ 文件扩展名
CPP_EXTENSIONS = {".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hxx", ".hh", ".inl", ".ipp"}

# 需要跳过的目录模式
SKIP_DIRS = {
    "test", "tests", "testing", "unittest", "unittests", "unit_test", "unit_tests",
    "gtest", "gmock", "mock", "mocks", "benchmark", "benchmarks",
    "example", "examples", "sample", "samples", "demo", "demos",
    "doc", "docs", "documentation",
    "third_party", "thirdparty", "3rdparty", "external", "vendor", "deps", "boost",
    "build", "cmake-build", "cmake-build-debug", "cmake-build-release",
    "out", "output", "bin", "lib", "obj",
    ".git", ".svn", ".hg", "__pycache__", "node_modules",
}

# 需要跳过的文件名模式
SKIP_FILE_PATTERNS = [
    re.compile(r"_test\.(?:cpp|cc|cxx|c|h|hpp)$", re.IGNORECASE),
    re.compile(r"_unittest\.(?:cpp|cc|cxx|c|h|hpp)$", re.IGNORECASE),
    re.compile(r"test_.*\.(?:cpp|cc|cxx|c|h|hpp)$", re.IGNORECASE),
    re.compile(r"_mock\.(?:cpp|cc|cxx|c|h|hpp)$", re.IGNORECASE),
    re.compile(r"_benchmark\.(?:cpp|cc|cxx|c|h|hpp)$", re.IGNORECASE),
    re.compile(r"_example\.(?:cpp|cc|cxx|c|h|hpp)$", re.IGNORECASE),
]


def _init_treesitter():
    """初始化 tree-sitter C++ 解析器（仅首次调用）。"""
    global _ts_parser, _ts_language
    if _ts_parser is not None:
        return

    try:
        import tree_sitter_cpp as tscpp
        from tree_sitter import Language, Parser

        _ts_language = Language(tscpp.language())
        _ts_parser = Parser(_ts_language)
    except ImportError:
        raise ImportError(
            "请安装 tree-sitter 和 tree-sitter-cpp:\n"
            "  uv pip install tree-sitter tree-sitter-cpp\n"
            "或:\n"
            "  pip install tree-sitter tree-sitter-cpp"
        )


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _node_text(node, source_bytes: bytes) -> str:
    """提取节点对应的源码文本。"""
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _find_preceding_comment(node, source_bytes: bytes) -> str:
    """查找紧挨着当前节点前面的注释（Doxygen / 行注释块）。"""
    if node.prev_named_sibling is None:
        return ""

    prev = node.prev_named_sibling
    # 如果前一个兄弟节点是 comment，收集连续的注释
    if prev.type != "comment":
        return ""

    comments = []
    cursor = prev
    while cursor is not None and cursor.type == "comment":
        comments.insert(0, _node_text(cursor, source_bytes))
        cursor = cursor.prev_named_sibling

    return "\n".join(comments)


def _get_name_from_node(node, source_bytes: bytes) -> str:
    """从各类声明节点中提取名称。"""
    # 尝试常见的子节点类型
    for child in node.children:
        if child.type in (
            "identifier",
            "field_identifier",
            "type_identifier",
            "namespace_identifier",
            "destructor_name",
            "qualified_identifier",
        ):
            return _node_text(child, source_bytes)
        # 函数声明器
        if child.type in ("function_declarator", "reference_declarator", "pointer_declarator"):
            return _get_name_from_node(child, source_bytes)
    return "<anonymous>"


# ---------------------------------------------------------------------------
# 主解析器
# ---------------------------------------------------------------------------

class CppParser:
    """C++ 语义解析器。"""

    # tree-sitter 节点类型 → chunk 类型的映射
    NODE_TYPE_MAP = {
        "function_definition":     "function",
        "declaration":             "function",   # 可能是函数声明或变量声明
        "class_specifier":         "class",
        "struct_specifier":        "struct",
        "enum_specifier":          "enum",
        "namespace_definition":    "namespace",
        "template_declaration":    "template",
        "preproc_def":             "macro",
        "preproc_function_def":    "macro",
        "preproc_ifdef":           "macro",
        "preproc_if":              "macro",
        "type_definition":         "typedef",
        "using_declaration":       "using",
        "alias_declaration":       "using",
    }

    # 我们关注的顶层节点类型
    TOP_LEVEL_TYPES = set(NODE_TYPE_MAP.keys())

    def __init__(
        self,
        max_chunk_chars: int = 2000,
        overlap_chars: int = 200,
        skip_dirs: Optional[set] = None,
        skip_file_patterns: Optional[list] = None,
        extra_extensions: Optional[set] = None,
    ):
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars
        self.skip_dirs = SKIP_DIRS if skip_dirs is None else skip_dirs
        self.skip_file_patterns = SKIP_FILE_PATTERNS if skip_file_patterns is None else skip_file_patterns
        self.extensions = CPP_EXTENSIONS | (extra_extensions or set())
        self.language = "cpp"

    def should_skip_file(self, file_path: str) -> bool:
        """判断文件是否应跳过。"""
        path = Path(file_path)

        # 检查扩展名
        if path.suffix.lower() not in self.extensions:
            return True

        # 检查目录
        parts = set(p.lower() for p in path.parts)
        if parts & self.skip_dirs:
            return True

        # 检查文件名模式
        name = path.name
        for pattern in self.skip_file_patterns:
            if pattern.search(name):
                return True

        return False

    def parse_file(self, file_path: str) -> List[Dict]:
        """
        解析单个 C++ 文件，返回语义代码块列表。

        对大的顶层声明（超过 max_chunk_chars），会自动拆分子块。
        """
        _init_treesitter()

        file_path = os.path.abspath(file_path)
        with open(file_path, "rb") as f:
            source_bytes = f.read()

        tree = _ts_parser.parse(source_bytes)
        root = tree.root_node
        chunks = uncovered_chunks(self, root, source_bytes, file_path, self.TOP_LEVEL_TYPES)

        for child in root.children:
            if child.type in self.TOP_LEVEL_TYPES:
                chunks.extend(self._extract_node(child, source_bytes, file_path))
            elif child.type == "comment":
                # 独立的顶层注释（如文件头注释），暂不作为单独 chunk
                pass

        # 如果没解析出任何语义块，将整个文件作为一个 chunk（兜底）
        if not chunks:
            full_text = source_bytes.decode("utf-8", errors="replace")
            if full_text.strip():
                chunks = self._split_plain_text(full_text, file_path, 1)

        return chunks

    def _extract_node(self, node, source_bytes: bytes, file_path: str) -> List[Dict]:
        """从一个顶层节点提取 chunk(s)。"""
        body = _node_text(node, source_bytes)
        chunk_type = self.NODE_TYPE_MAP.get(node.type, "other")
        name = _get_name_from_node(node, source_bytes)
        doc = _find_preceding_comment(node, source_bytes)

        if node.type == "declaration" and "(" not in body:
            chunk_type = "variable"

        # 生成签名（取第一行或到 { 之前）
        sig_match = re.match(r"^([^{;]+)", body)
        signature = sig_match.group(1).strip() if sig_match else name

        base_chunk = {
            "type": chunk_type,
            "name": name,
            "signature": signature,
            "doc": doc,
            "file_path": file_path,
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
            "language": self.language,
        }

        # 如果代码块不大，直接返回一整个 chunk
        if len(body) <= self.max_chunk_chars:
            base_chunk["body"] = body
            return [base_chunk]

        # 大代码块 → 按子节点拆分（类/结构体内部方法等）
        sub_chunks = []
        if chunk_type in ("class", "struct", "namespace"):
            sub_chunks = self._split_compound(node, source_bytes, file_path, name, chunk_type)

        # 如果子节点拆分未产出结果，按文本硬拆
        if not sub_chunks:
            sub_chunks = self._split_plain_text(body, file_path, node.start_point[0] + 1)
            for sc in sub_chunks:
                sc["type"] = chunk_type
                sc["name"] = name
                sc["signature"] = signature
                sc["doc"] = doc

        return sub_chunks

    def _split_compound(
        self, node, source_bytes: bytes, file_path: str, parent_name: str, parent_type: str
    ) -> List[Dict]:
        """拆分类/结构体/命名空间的子节点。"""
        chunks = []
        # 找到 body（field_declaration_list / declaration_list）
        body_node = None
        for child in node.children:
            if child.type in ("field_declaration_list", "declaration_list"):
                body_node = child
                break

        if body_node is None:
            return []

        for child in body_node.children:
            if child.type in self.TOP_LEVEL_TYPES or child.type in (
                "function_definition", "field_declaration", "friend_declaration",
                "access_specifier", "template_declaration",
            ):
                body_text = _node_text(child, source_bytes)
                if len(body_text.strip()) < 10:
                    continue

                child_name = _get_name_from_node(child, source_bytes)
                full_name = f"{parent_name}::{child_name}" if child_name != "<anonymous>" else parent_name

                chunk = {
                    "type": self.NODE_TYPE_MAP.get(child.type, "member"),
                    "name": full_name,
                    "signature": re.match(r"^([^{;]+)", body_text).group(1).strip() if re.match(r"^([^{;]+)", body_text) else full_name,
                    "body": body_text,
                    "doc": _find_preceding_comment(child, source_bytes),
                    "file_path": file_path,
                    "start_line": child.start_point[0] + 1,
                    "end_line": child.end_point[0] + 1,
                    "language": self.language,
                }

                if len(body_text) > self.max_chunk_chars:
                    sub = self._split_plain_text(body_text, file_path, child.start_point[0] + 1)
                    for s in sub:
                        s.update({
                            "type": chunk["type"],
                            "name": full_name,
                            "doc": chunk["doc"],
                            "language": self.language,
                        })
                    chunks.extend(sub)
                else:
                    chunks.append(chunk)

        return chunks

    def _split_plain_text(self, text: str, file_path: str, start_line: int) -> List[Dict]:
        """按字符数硬拆文本，带重叠。"""
        chunks = []
        step = self.max_chunk_chars - self.overlap_chars
        if step <= 0:
            step = self.max_chunk_chars

        for i in range(0, len(text), step):
            segment = text[i: i + self.max_chunk_chars]
            if not segment.strip():
                continue

            # 估算行号
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
        """递归收集目录下所有 C++ 源文件（排除跳过规则命中的）。"""
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


# ---------------------------------------------------------------------------
# 直接运行时的简单测试
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("用法: python cpp.py <file_or_dir>")
        sys.exit(1)

    target = sys.argv[1]
    parser = CppParser()

    if os.path.isfile(target):
        result = parser.parse_file(target)
    else:
        files = parser.collect_files(target)
        print(f"[INFO] 共收集到 {len(files)} 个 C++ 文件")
        result = []
        for f in files:
            result.extend(parser.parse_file(f))

    print(f"[INFO] 共提取 {len(result)} 个代码块")
    for i, chunk in enumerate(result[:5]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"  type:      {chunk['type']}")
        print(f"  name:      {chunk['name']}")
        print(f"  signature: {chunk['signature'][:80]}")
        print(f"  lines:     {chunk['start_line']}-{chunk['end_line']}")
        print(f"  body_len:  {len(chunk['body'])} chars")
