#!/usr/bin/env python3
"""Plan and build token-bounded, content-hash repository index generations."""

import argparse
import json
import os
from pathlib import Path

from parser.cpp import CppParser
from parser.go import GoParser
from parser.markdown import MarkdownParser
from parser.python import PythonParser
from rag_common import project_id, source_files
from rag_index import build, plan

RAG_BASE_DIR = os.path.join(os.path.expanduser('~'), '.rag')


def _repo_to_project_id(repo_dir):
    return project_id(repo_dir)


def _default_db_path(repo_dir):
    return os.path.join(RAG_BASE_DIR, 'rag_db', project_id(repo_dir))


class MultiLangParser:
    def __init__(self):
        self.parsers = [CppParser(), PythonParser(), GoParser(), MarkdownParser()]

    def get_parser_for_path(self, filename):
        return next((parser for parser in self.parsers if not parser.should_skip_file(filename)), None)

    def collect_files(self, root_dir):
        root = Path(root_dir).resolve()
        return [filename for filename in source_files(root)
                if self.get_parser_for_path(str(Path(filename).relative_to(root))) is not None]


def get_build_plan(repo_dir, db_path=None):
    return plan(repo_dir, db_path or _default_db_path(repo_dir), MultiLangParser())


def print_build_plan(result):
    print(json.dumps(result, indent=2, ensure_ascii=True))


def build_knowledge_base(repo_dir, db_path=None, full_rebuild=False, download_model=False):
    return build(repo_dir, db_path or _default_db_path(repo_dir), MultiLangParser(),
                 full=full_rebuild, download_model=download_model)


def main():
    parser = argparse.ArgumentParser(description='Local content-hash code index builder')
    parser.add_argument('--repo', required=True, help='Source repository or directory')
    parser.add_argument('--db', help='Index directory outside the source repository')
    parser.add_argument('--plan', action='store_true', help='Read-only file hash plan; no model download')
    parser.add_argument('--full', action='store_true', help='Build a new full generation; retain old collections')
    parser.add_argument('--download-model', action='store_true', help='Explicitly allow Chroma model download if uncached')
    args = parser.parse_args()
    if args.plan:
        print_build_plan(get_build_plan(args.repo, args.db))
    else:
        print(json.dumps(build_knowledge_base(args.repo, args.db, args.full, args.download_model),
                         indent=2, ensure_ascii=True))


if __name__ == '__main__':
    main()
