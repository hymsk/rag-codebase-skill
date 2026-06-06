"""Generation and hash regressions; no embedding/model dependency required."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import rag_builder
import rag_index
from rag_store import ACTIVE, PENDING, active_index, atomic_json, operation_lock


class TestChunker:
    fingerprint = {'model': 'synthetic', 'max_tokens': 256, 'chunker_version': 1}

    def prepare(self, chunks):
        return [dict(chunk, document=chunk['body']) for chunk in chunks]


class Collection:
    def __init__(self):
        self.rows = {}
        self.embedded = 0

    def count(self):
        return len(self.rows)

    def upsert(self, ids, documents, metadatas, embeddings=None):
        if embeddings is None:
            self.embedded += len(ids)
        for i, key in enumerate(ids):
            self.rows[key] = (documents[i], metadatas[i], [0.1, 0.2] if embeddings is None else embeddings[i])

    def get(self, where, include, limit, offset):
        rows = [(key, row) for key, row in self.rows.items()
                if all(row[1].get(k) == v for k, v in where.items())][offset:offset + limit]
        return {'ids': [key for key, _ in rows], 'documents': [row[0] for _, row in rows],
                'metadatas': [row[1] for _, row in rows], 'embeddings': [row[2] for _, row in rows]}


class Client:
    def __init__(self):
        self.collections = {}

    def create_collection(self, name, **kwargs):
        self.collections[name] = Collection()
        return self.collections[name]

    def get_collection(self, name, **kwargs):
        return self.collections[name]


class BuilderTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.repo, self.db = self.root / 'repo', self.root / 'db'
        self.repo.mkdir()
        self.file = self.repo / 'readme.md'
        self.file.write_text('# title\noriginal\n')
        self.router = rag_builder.MultiLangParser()
        self.client = Client()
        for target, value in [('_client', self.client), ('_embedding', None),
                              ('embedding_fingerprint', {'embedding_assets': {'model': 'synthetic-v1'}})]:
            patcher = patch.object(rag_index, target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def build(self, full=False, chunker=None):
        return rag_index.build(self.repo, self.db, self.router, full=full, chunker=chunker or TestChunker())

    def active(self):
        return active_index(self.db)

    def test_first_build_and_unchanged_skip(self):
        self.assertEqual(self.build()['mode'], 'full')
        old = self.active()
        self.assertEqual(self.build()['mode'], 'skip')
        self.assertEqual(self.active(), old)
        self.assertFalse((self.db / PENDING).exists())

    def test_dirty_update_restore_delete_and_reuse_vectors(self):
        other = self.repo / 'other.md'
        other.write_text('# other\nconstant\n')
        self.build()
        old = copy.deepcopy(self.client.collections[self.active()['collection']].rows)
        original = self.file.read_text()
        self.file.write_text('# title\ndirty\n')
        result = self.build()
        self.assertEqual((result['parsed_files'], result['reused_files']), (1, 1))
        new = self.client.collections[result['collection_name']]
        self.assertEqual(new.embedded, 1)
        self.assertTrue(any(key in new.rows for key in old))
        self.file.write_text(original)
        self.assertEqual(self.build()['parsed_files'], 1)
        other.unlink()
        self.assertEqual(self.build()['deleted_files'], 1)
        self.assertNotIn('other.md', self.active()['files'])

    def test_rename_and_unusual_paths(self):
        self.build()
        path = self.repo / '新\t名\n.md'
        self.file.rename(path)
        result = self.build()
        self.assertEqual((result['parsed_files'], result['deleted_files']), (1, 1))
        self.assertEqual(set(self.active()['files']), {path.name})

    def test_parse_failure_preserves_active_and_retry_needs_no_full(self):
        self.build()
        old = self.active()
        old_rows = copy.deepcopy(self.client.collections[old['collection']].rows)
        self.file.write_text('changed')
        parser = self.router.get_parser_for_path('readme.md')
        with patch.object(parser, 'parse_file', side_effect=OSError('read failed')):
            with self.assertRaisesRegex(OSError, 'read failed'):
                self.build()
        self.assertEqual(self.active(), old)
        self.assertEqual(self.client.collections[old['collection']].rows, old_rows)
        self.assertTrue((self.db / PENDING).exists())
        self.assertEqual(self.build()['mode'], 'incremental')

    def test_embedding_failure_preserves_active_even_full(self):
        self.build()
        old = self.active()
        with patch.object(Collection, 'upsert', side_effect=RuntimeError('embedding failed')):
            with self.assertRaisesRegex(RuntimeError, 'embedding failed'):
                self.build(full=True)
        self.assertEqual(self.active(), old)

    def test_activation_failure_preserves_active(self):
        self.build()
        old = self.active()
        original = rag_index.atomic_json
        def fail(path, value):
            if Path(path).name == ACTIVE:
                raise OSError('activation failed')
            original(path, value)
        with patch.object(rag_index, 'atomic_json', side_effect=fail):
            with self.assertRaisesRegex(OSError, 'activation failed'):
                self.build(full=True)
        self.assertEqual(self.active(), old)

    def test_source_changes_during_build_dont_activate(self):
        self.build()
        old = self.active()
        original = rag_index._write_file
        def changing(*args):
            original(*args)
            self.file.write_text('concurrent edit')
        with patch.object(rag_index, '_write_file', side_effect=changing):
            with self.assertRaisesRegex(RuntimeError, 'Source changed'):
                self.build(full=True)
        self.assertEqual(self.active(), old)

    def test_same_mtime_content_change_is_detected(self):
        import os
        self.build()
        info = self.file.stat()
        self.file.write_text('changed')
        os.utime(self.file, ns=(info.st_atime_ns, info.st_mtime_ns))
        self.assertEqual(self.build()['parsed_files'], 1)

    def test_fingerprint_mismatch_requires_full(self):
        self.build()
        changed = TestChunker()
        changed.fingerprint = dict(changed.fingerprint, max_tokens=128)
        with self.assertRaisesRegex(RuntimeError, 'fingerprint'):
            self.build(chunker=changed)
        self.assertEqual(self.build(full=True, chunker=changed)['mode'], 'full')

    def test_legacy_requires_explicit_migration(self):
        self.db.mkdir()
        (self.db / 'chroma.sqlite3').touch()
        with self.assertRaisesRegex(RuntimeError, 'Legacy'):
            self.build()
        self.assertEqual(rag_builder.get_build_plan(str(self.repo), str(self.db))['status'], 'migration_required')
        self.assertEqual(self.build(full=True)['mode'], 'full')

    def test_plan_is_readonly_and_needs_no_model(self):
        result = rag_builder.get_build_plan(str(self.repo), str(self.db))
        self.assertEqual(result['changed_files'], {'readme.md': 'A'})
        self.assertFalse(self.db.exists())

    def test_owner_and_db_inside_source_rejected(self):
        self.build()
        sibling = self.root / 'other-repo'
        sibling.mkdir()
        with self.assertRaisesRegex(RuntimeError, 'different repository'):
            rag_index.build(sibling, self.db, self.router, chunker=TestChunker())
        with self.assertRaisesRegex(RuntimeError, 'outside'):
            rag_index.build(self.repo, self.repo / 'db', self.router, chunker=TestChunker())

    def test_file_becomes_symlink_removed_not_read(self):
        self.build()
        self.file.unlink()
        outside = self.root / 'secret.md'
        outside.write_text('not indexed')
        self.file.symlink_to(outside)
        self.assertEqual(self.build()['deleted_files'], 1)
        self.assertEqual(self.active()['files'], {})

    def test_git_scope_and_nested_ignore(self):
        subprocess.run(['git', 'init', '-q', str(self.repo)], check=True)
        scope = self.repo / 'scope'
        scope.mkdir()
        (scope / 'inside.md').write_text('inside')
        (scope / '.gitignore').write_text('hidden.md\n')
        (scope / 'hidden.md').write_text('hidden')
        result = rag_builder.get_build_plan(str(scope), str(self.db))
        self.assertEqual(result['changed_files'], {'inside.md': 'A'})

    def test_lock_blocks_other_process_and_releases(self):
        scripts = str(Path(rag_index.__file__).parent)
        code = ('import sys; sys.path.insert(0, sys.argv[1]); from rag_store import operation_lock; '
                '\nwith operation_lock(sys.argv[2], create=True): pass')
        with operation_lock(self.db, create=True):
            proc = subprocess.run([sys.executable, '-c', code, scripts, str(self.db)], capture_output=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn(b'busy', proc.stderr)
            with self.assertRaisesRegex(RuntimeError, 'busy'):
                self.build()
        with operation_lock(self.db):
            pass

    def test_corrupt_manifest_does_not_rebuild_silently(self):
        self.build()
        (self.db / ACTIVE).write_text('{broken')
        with self.assertRaises(json.JSONDecodeError):
            self.build(full=True)

    def test_stale_pending_is_not_authority(self):
        self.build()
        old = self.active()
        atomic_json(self.db / PENDING, {'repo_dir': str(self.repo), 'status': 'building'})
        self.assertEqual(self.build()['mode'], 'skip')
        self.assertEqual(self.active(), old)

    def test_atomic_write_error_preserves_previous_manifest(self):
        self.build()
        old = (self.db / ACTIVE).read_bytes()
        with patch('rag_store.os.replace', side_effect=OSError('disk failure')):
            with self.assertRaises(OSError):
                atomic_json(self.db / ACTIVE, {})
        self.assertEqual((self.db / ACTIVE).read_bytes(), old)

    def test_missing_lock_fails_closed_for_existing_index(self):
        self.build()
        (self.db / '.rag.lock').unlink()
        with self.assertRaisesRegex(RuntimeError, 'no operation lock'):
            with operation_lock(self.db):
                self.fail('entered without a lock')
        self.assertEqual(self.build()['mode'], 'skip')

    def test_changed_embedding_assets_require_full(self):
        self.build()
        with patch.object(rag_index, 'embedding_fingerprint', return_value={'embedding_assets': {'model': 'v2'}}):
            with self.assertRaisesRegex(RuntimeError, 'fingerprint'):
                self.build()
            self.assertEqual(self.build(full=True)['mode'], 'full')


if __name__ == '__main__':
    unittest.main()
