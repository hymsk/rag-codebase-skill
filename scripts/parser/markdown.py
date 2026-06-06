#!/usr/bin/env python3
"""
Markdown 文档解析器 — 按标题层级切分 README、设计文档和操作手册。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MARKDOWN_EXTENSIONS = {".md", ".markdown", ".mdown", ".mkd", ".mdx"}

SKIP_DIRS = {
    ".git", ".svn", ".hg",
    ".pytest_cache", ".mypy_cache", ".tox", ".venv", "venv", "__pycache__",
    "node_modules", "bower_components",
    "third_party", "thirdparty", "3rdparty", "external", "vendor", "deps",
    "build", "dist", "out", "output", "target", "bazel-bin", "bazel-out",
    "_build", "site", ".docusaurus", ".next", ".nuxt",
}

SKIP_FILE_PATTERNS = [
    re.compile(r".*\.generated\.md$", re.IGNORECASE),
]

ATX_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
SETEXT_HEADING_RE = re.compile(r"^[ \t]*(=+|-+)[ \t]*$")
FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


class MarkdownParser:
    """Markdown 文档解析器。"""

    def __init__(
        self,
        max_chunk_chars: int = 3000,
        overlap_lines: int = 2,
        skip_dirs: Optional[set] = None,
        skip_file_patterns: Optional[list] = None,
        extra_extensions: Optional[set] = None,
    ):
        self.max_chunk_chars = max_chunk_chars
        self.overlap_lines = overlap_lines
        self.skip_dirs = SKIP_DIRS if skip_dirs is None else skip_dirs
        self.skip_file_patterns = SKIP_FILE_PATTERNS if skip_file_patterns is None else skip_file_patterns
        self.extensions = MARKDOWN_EXTENSIONS | (extra_extensions or set())
        self.language = "markdown"

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
        file_path = os.path.abspath(file_path)
        with open(file_path, "rb") as f:
            source_bytes = f.read()

        text = source_bytes.decode("utf-8", errors="replace")
        if not text.strip():
            return []

        lines = text.splitlines(keepends=True)
        sections = self._collect_sections(lines, file_path)
        chunks: List[Dict] = []
        for section in sections:
            chunks.extend(self._section_to_chunks(section))
        return chunks

    def _collect_sections(self, lines: List[str], file_path: str) -> List[Dict]:
        headings = self._find_headings(lines)
        if not headings:
            return [{
                "title": Path(file_path).name,
                "heading_path": Path(file_path).name,
                "level": 0,
                "start_line": 1,
                "end_line": len(lines),
                "lines": lines,
                "file_path": file_path,
            }]

        sections = []
        if headings[0][2] > 1:
            preface = lines[:headings[0][2] - 1]
            if "".join(preface).strip():
                sections.append({
                    "title": Path(file_path).name,
                    "heading_path": Path(file_path).name,
                    "level": 0,
                    "start_line": 1,
                    "end_line": headings[0][2] - 1,
                    "lines": preface,
                    "file_path": file_path,
                })

        stack: List[Tuple[int, str]] = []
        for idx, (level, title, start_line) in enumerate(headings):
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))

            end_line = headings[idx + 1][2] - 1 if idx + 1 < len(headings) else len(lines)
            section_lines = lines[start_line - 1:end_line]
            if not "".join(section_lines).strip():
                continue

            sections.append({
                "title": title,
                "heading_path": " > ".join(item[1] for item in stack),
                "level": level,
                "start_line": start_line,
                "end_line": end_line,
                "lines": section_lines,
                "file_path": file_path,
            })
        return sections

    def _find_headings(self, lines: List[str]) -> List[Tuple[int, str, int]]:
        headings: List[Tuple[int, str, int]] = []
        fence_char = None
        fence_length = 0

        for idx, line in enumerate(lines):
            fence = FENCE_RE.match(line.rstrip("\r\n"))
            if fence_char is not None:
                if (fence and fence.group(1)[0] == fence_char
                        and len(fence.group(1)) >= fence_length
                        and not fence.group(2).strip()):
                    fence_char = None
                continue
            if fence:
                marker, info = fence.groups()
                if marker[0] != "`" or "`" not in info:
                    fence_char, fence_length = marker[0], len(marker)
                    continue

            match = ATX_HEADING_RE.match(line.rstrip("\n\r"))
            if match:
                level = len(match.group(1))
                title = self._clean_heading_title(match.group(2))
                if title:
                    headings.append((level, title, idx + 1))
                continue

            if idx > 0 and SETEXT_HEADING_RE.match(line.rstrip("\n\r")):
                prev = lines[idx - 1].strip()
                if prev and not ATX_HEADING_RE.match(prev):
                    level = 1 if line.lstrip().startswith("=") else 2
                    title = self._clean_heading_title(prev)
                    if title:
                        headings.append((level, title, idx))
        return headings

    def _section_to_chunks(self, section: Dict) -> List[Dict]:
        body = "".join(section["lines"])
        base = {
            "type": "document_section",
            "name": section["title"],
            "signature": section["heading_path"],
            "doc": "",
            "file_path": section["file_path"],
            "start_line": section["start_line"],
            "end_line": section["end_line"],
            "language": self.language,
        }

        if len(body) <= self.max_chunk_chars:
            base["body"] = body
            return [base]

        chunks = []
        for idx, (chunk_lines, start_line, end_line) in enumerate(
            self._split_lines(section["lines"], section["start_line"]),
            1,
        ):
            chunk = dict(base)
            chunk["name"] = f"{section['title']} part {idx}"
            chunk["start_line"] = start_line
            chunk["end_line"] = end_line
            chunk["body"] = "".join(chunk_lines)
            chunks.append(chunk)
        return chunks

    def _split_lines(self, lines: List[str], first_line: int) -> List[Tuple[List[str], int, int]]:
        chunks: List[Tuple[List[str], int, int]] = []
        current: List[str] = []
        current_start = first_line
        current_len = 0

        for offset, line in enumerate(lines):
            line_no = first_line + offset
            if len(line) > self.max_chunk_chars:
                if current:
                    chunks.append((current, current_start, line_no - 1))
                    current = []
                    current_len = 0
                for start in range(0, len(line), self.max_chunk_chars):
                    chunks.append(([line[start:start + self.max_chunk_chars]], line_no, line_no))
                current_start = line_no + 1
                continue
            if current and current_len + len(line) > self.max_chunk_chars:
                chunks.append((current, current_start, line_no - 1))
                overlap = current[-self.overlap_lines:] if self.overlap_lines > 0 else []
                current = list(overlap)
                current_start = line_no - len(current)
                current_len = sum(len(item) for item in current)
                while current and current_len + len(line) > self.max_chunk_chars:
                    current_len -= len(current.pop(0))
                    current_start += 1

            current.append(line)
            current_len += len(line)

            if line.strip() == "" and current_len >= self.max_chunk_chars * 0.8:
                chunks.append((current, current_start, line_no))
                current = []
                current_start = line_no + 1
                current_len = 0

        if current and "".join(current).strip():
            chunks.append((current, current_start, first_line + len(lines) - 1))
        return chunks

    def _clean_heading_title(self, title: str) -> str:
        title = title.strip()
        title = re.sub(r"\s+", " ", title)
        title = title.strip("`*_ ")
        return title or "<untitled>"

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
