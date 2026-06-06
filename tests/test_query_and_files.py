import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import rag_query
from rag_common import project_id, source_files, safe_source
from parser.markdown import MarkdownParser


class QueryTests(unittest.TestCase):
    def test_comment_output_retains_associated_range(self):
        result = [{'document': '# comment', 'metadata': {'line_range_kind': 'associated_symbol',
                   'source_field': 'doc', 'start_line': 2, 'end_line': 3}, 'distance': 0.1}]
        self.assertIn('关联符号行号（不是注释精确位置）', rag_query.format_results_for_ai(result, 'q'))
        data = json.loads(rag_query.format_results_json(result))[0]
        self.assertEqual(data['line_range_kind'], 'associated_symbol')
        self.assertEqual(data['source_field'], 'doc')

    def test_filter_is_applied_before_vector_ranking(self):
        collection = Mock()
        collection.count.return_value = 101
        collection.get.return_value = {'metadatas':
            [{'file_path': 'noise.py'}] * 100 + [{'file_path': 'target.py'}]}
        collection.query.return_value = {'ids': [['hit']], 'documents': [['target']],
            'metadatas': [[{'file_path': 'target.py'}]], 'distances': [[0.1]]}
        with patch.object(rag_query, 'init_chromadb_readonly', return_value=(None, collection)):
            result = rag_query._query_collection('unused', 'needle', 8, 'target')
        self.assertEqual(len(result), 1)
        self.assertEqual(collection.query.call_args.kwargs['where'],
                         {'file_path': {'$in': ['target.py']}})
        self.assertEqual(collection.query.call_args.kwargs['n_results'], 1)

    def test_invalid_input_does_not_open_database(self):
        with patch.object(rag_query, 'init_chromadb_readonly') as init:
            for query, top_k in [('valid', 0), (' ', 5), ('valid', -1)]:
                with self.assertRaises(ValueError):
                    rag_query.query_knowledge_base('unused', query, top_k)
            init.assert_not_called()

    def test_missing_path_match_does_not_query(self):
        collection = Mock()
        collection.count.return_value = 1
        collection.get.return_value = {'metadatas': [{'file_path': 'other.py'}]}
        with patch.object(rag_query, 'init_chromadb_readonly', return_value=(None, collection)):
            self.assertEqual(rag_query._query_collection('unused', 'q', 5, 'absent'), [])
        collection.query.assert_not_called()


class FileTests(unittest.TestCase):
    def test_path_identity_does_not_flatten_collisions(self):
        self.assertNotEqual(project_id('/tmp/a-b/c'), project_id('/tmp/a/b-c'))

    def test_git_ignore_links_special_files_and_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            (root / '.gitignore').write_text('ignored.py\n')
            for name in ['ignored.py', 'visible.py']:
                (root / name).write_text('pass')
            (root / 'nested').mkdir()
            (root / 'nested/.gitignore').write_text('hidden.py\n')
            (root / 'nested/hidden.py').write_text('pass')
            (root / 'linked.py').symlink_to(root / 'visible.py')
            (root / 'large.py').write_bytes(b'x' * (1024 * 1024 + 1))
            os.mkfifo(root / 'fifo.py')
            files = {Path(path).relative_to(root).as_posix() for path in source_files(root)}
            self.assertIn('visible.py', files)
            self.assertFalse(files & {'ignored.py', 'nested/hidden.py', 'linked.py', 'large.py', 'fifo.py'})

    def test_non_git_directory_link_is_not_followed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'real').mkdir()
            (root / 'real/code.py').write_text('pass')
            (root / 'link').symlink_to(root / 'real', target_is_directory=True)
            self.assertFalse(safe_source(root, root / 'link/code.py'))
            self.assertEqual(list(source_files(root)), [str(root / 'real/code.py')])

    def test_missing_file_is_error_not_empty_success(self):
        with self.assertRaises(FileNotFoundError):
            MarkdownParser().parse_file('/nonexistent-rag-source.md')


class MarkdownTests(unittest.TestCase):
    def test_fence_character_and_length(self):
        for body in ['```text\n~~~\n# fake\n```\n# real\n',
                     '````text\n```\n# fake\n````\n# real\n']:
            headings = MarkdownParser()._find_headings(body.splitlines(keepends=True))
            self.assertEqual([title for _, title, _ in headings], ['real'])

    def test_empty_skip_configuration_is_respected(self):
        self.assertFalse(MarkdownParser(skip_dirs=set(), skip_file_patterns=[])
                         .should_skip_file('vendor/test.generated.md'))


if __name__ == '__main__':
    unittest.main()
