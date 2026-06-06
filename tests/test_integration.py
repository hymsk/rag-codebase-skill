"""Optional real dependencies; deterministic vectors, no model download."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import rag_builder
import rag_query
import rag_index
from rag_store import active_index
from parser.cpp import CppParser
from parser.go import GoParser
from parser.python import PythonParser
from parser.markdown import MarkdownParser

HAS_PARSERS = all(importlib.util.find_spec(name) for name in
    ['tree_sitter', 'tree_sitter_cpp', 'tree_sitter_python', 'tree_sitter_go'])
HAS_CHROMA = importlib.util.find_spec('chromadb') is not None


@unittest.skipUnless(HAS_PARSERS, 'install requirements.txt for real parser tests')
class ParserIntegrationTests(unittest.TestCase):
    def test_top_level_gaps_and_syntax_errors(self):
        fixtures = [
            (PythonParser(), '.py', 'import os\nMARKER = 1\ndef f():\n    return 1\n', '\ndef broken('),
            (GoParser(), '.go', 'package main\nimport "fmt"\nvar MARKER = 1\nfunc f() { fmt.Println(MARKER) }\n', '\nfunc broken('),
            (CppParser(), '.cpp', '#include <stdio.h>\nint MARKER = 1;\nint f() { return MARKER; }\n', '\nvoid broken('),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for parser, suffix, valid, invalid in fixtures:
                with self.subTest(language=parser.language):
                    path = Path(directory) / ('source' + suffix)
                    path.write_text(valid)
                    chunks = parser.parse_file(str(path))
                    bodies = '\n'.join(c['body'] for c in chunks)
                    self.assertIn('MARKER', bodies)
                    self.assertIn(valid.splitlines()[0], bodies)
                    path.write_text(valid + invalid)
                    with self.assertRaisesRegex(ValueError, 'syntax error'):
                        parser.parse_file(str(path))

    def test_read_errors_propagate_for_all_languages(self):
        for parser in [PythonParser(), GoParser(), CppParser()]:
            with self.subTest(language=parser.language), self.assertRaises(OSError):
                parser.parse_file('/nonexistent-rag-source')


class MarkdownStorageTests(unittest.TestCase):
    def test_long_line_keeps_tail_and_bounded_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'long.md'
            path.write_text('x' * 12000 + 'TAIL_MARKER')
            chunks = MarkdownParser().parse_file(str(path))
            self.assertTrue(any('TAIL_MARKER' in c['body'] for c in chunks))
            self.assertTrue(all(len(c['body']) <= 3000 for c in chunks))
            self.assertTrue(all(c['start_line'] == c['end_line'] == 1 for c in chunks))


@unittest.skipUnless(HAS_CHROMA, 'install requirements.txt for real Chroma tests')
class ChromaIntegrationTests(unittest.TestCase):
    def test_incomplete_model_never_implicitly_downloads(self):
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2 as EF
        with tempfile.TemporaryDirectory() as directory, patch.object(EF, 'DOWNLOAD_PATH', Path(directory)), \
                patch.object(EF, '_download', side_effect=AssertionError('network forbidden')) as download:
            embedding = rag_index._embedding()
            with self.assertRaisesRegex(FileNotFoundError, '--download-model'):
                embedding(['test'])
            download.assert_not_called()

    def test_model_weight_hash_changes_fingerprint(self):
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2 as EF
        with tempfile.TemporaryDirectory() as directory, patch.object(EF, 'DOWNLOAD_PATH', Path(directory)):
            folder = Path(directory) / EF.EXTRACTED_FOLDER_NAME
            folder.mkdir()
            for name in ('model.onnx', 'config.json', 'special_tokens_map.json',
                         'tokenizer_config.json', 'tokenizer.json', 'vocab.txt'):
                (folder / name).write_bytes(b'synthetic')
            before = rag_index.embedding_fingerprint()
            (folder / 'model.onnx').write_bytes(b'different-synthetic-weights')
            after = rag_index.embedding_fingerprint()
            self.assertNotEqual(before['embedding_assets']['model.onnx'], after['embedding_assets']['model.onnx'])
            self.assertEqual(set(after['embedding_packages']), {'chromadb', 'tokenizers', 'onnxruntime'})

    def test_real_build_query_filter_and_recovery(self):
        import chromadb
        from chromadb.api.types import EmbeddingFunction
        from rag_chunking import TokenChunker
        from tokenizers import Tokenizer, models, pre_tokenizers
        tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0, 'target': 1}, unk_token='[UNK]'))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        chunker = TokenChunker(tokenizer)

        class TestEmbedding(EmbeddingFunction):
            def __init__(self, **kwargs):
                pass

            def __call__(self, input):
                return [[float('target' in text), 1.0, 0.5] for text in input]

            @staticmethod
            def name():
                return 'rag-test-deterministic'

            def get_config(self):
                return {}

            @staticmethod
            def build_from_config(config):
                return TestEmbedding()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, db = root / 'repo', root / 'db'
            repo.mkdir()
            (repo / 'readme.md').write_text('# target\n' + 'x' * 12000 + 'TAIL_MARKER')
            with patch('rag_index._embedding', TestEmbedding), patch('rag_query._embedding', TestEmbedding), \
                    patch('rag_index.embedding_fingerprint', return_value={'embedding_assets': {'model': 'synthetic'}}), \
                    patch('rag_query.embedding_fingerprint', return_value={'embedding_assets': {'model': 'synthetic'}}), \
                    patch.object(TokenChunker, 'from_default_model', return_value=chunker):
                with contextlib.redirect_stdout(io.StringIO()):
                    stats = rag_builder.build_knowledge_base(str(repo), str(db))
                self.assertGreater(stats['collection_count'], 1)
                results = rag_query.query_knowledge_base(str(db), 'target', 8, 'readme.md')
                self.assertTrue(any('TAIL_MARKER' in r['document'] for r in results))
                for item in results:
                    self.assertEqual(item['document'], item['metadata']['source_text'])
                    self.assertNotIn('# Section:', item['document'])
                self.assertEqual(len({r['id'] for r in results}), len(results))
                client, collection = rag_query.init_chromadb_readonly(str(db))
                self.assertFalse(client.get_settings().anonymized_telemetry)
                with patch.object(chromadb, 'PersistentClient') as factory:
                    factory.return_value.get_collection.side_effect = RuntimeError('database failure')
                    with self.assertRaisesRegex(RuntimeError, 'database failure'):
                        rag_query.init_chromadb_readonly(str(db))
                old = active_index(db)
                (repo / 'new.md').write_text('# new\nfresh target text')
                stats = rag_builder.build_knowledge_base(str(repo), str(db))
                self.assertEqual((stats['parsed_files'], stats['reused_files']), (1, 1))
                self.assertNotEqual(active_index(db)['collection'], old['collection'])
                before_failure = active_index(db)
                (repo / 'new.md').write_text('# changed\nnow failing')
                def fail_embedding(self, input):
                    raise RuntimeError('embedding failure')
                with patch.object(TestEmbedding, '__call__', fail_embedding):
                    with self.assertRaisesRegex(RuntimeError, 'embedding failure'):
                        rag_builder.build_knowledge_base(str(repo), str(db))
                self.assertEqual(active_index(db), before_failure)
                self.assertTrue(rag_query.query_knowledge_base(str(db), 'target', repo_dir=str(repo)))
                self.assertTrue((db / 'build_in_progress.json').exists())
                rag_builder.build_knowledge_base(str(repo), str(db))
                self.assertFalse((db / 'build_in_progress.json').exists())


if __name__ == '__main__':
    unittest.main()
