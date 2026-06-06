"""Content-hash updates with immutable Chroma generations and atomic activation."""

import hashlib
import importlib.metadata
import os
from pathlib import Path
import uuid

from rag_common import safe_source
from rag_store import (ACTIVE, PENDING, SCHEMA, active_index, atomic_json,
                       operation_lock, read_json, validate_db_location, validate_owner)


def snapshot(repo, router):
    result = {}
    for filename in router.collect_files(str(repo)):
        path = Path(filename)
        if not safe_source(repo, path):
            raise RuntimeError(f'Source became unsafe: {path}')
        result[path.relative_to(repo).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def parser_fingerprint():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in [root / 'rag_common.py', *sorted((root / 'parser').glob('*.py'))]:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    versions = {}
    for package in ('tree-sitter', 'tree-sitter-cpp', 'tree-sitter-python', 'tree-sitter-go'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'unavailable'
    return {'parser_sha256': digest.hexdigest(), 'parser_packages': versions}


def plan(repo, db, router):
    repo, db = Path(repo).resolve(), Path(db).absolute()
    if not repo.is_dir():
        raise ValueError(f'Repository does not exist: {repo}')
    validate_db_location(repo, db)
    with operation_lock(db):
        validate_owner(repo, db)
        previous = active_index(db)
        files = snapshot(repo, router)
        old_files = previous['files'] if previous else {}
        changed = {name: ('M' if name in old_files else 'A') for name, digest in files.items()
                   if old_files.get(name, {}).get('sha256') != digest}
        changed.update({name: 'D' for name in old_files if name not in files})
        legacy = previous is None and ((db / 'chroma.sqlite3').exists() or (db / 'git_info.json').exists())
        parser_changed = previous is not None and any(
            previous['fingerprint'].get(key) != value for key, value in parser_fingerprint().items())
        status = ('migration_required' if legacy else 'first_build' if previous is None
                  else 'fingerprint_changed' if parser_changed else 'content_changes' if changed else 'no_changes')
        return {'repo_dir': str(repo), 'db_path': str(db), 'status': status,
                'recommended_mode': 'full' if previous is None or parser_changed else 'incremental' if changed else 'skip',
                'changed_files': changed, 'file_count': len(files),
                'pending_build': read_json(db / PENDING) is not None,
                'active_collection': previous['collection'] if previous else None,
                'note': 'Content-hash plan; embedding fingerprint is checked at build time; no model download.'}


def _client(db):
    import chromadb
    from chromadb.config import Settings
    return chromadb.PersistentClient(path=str(db), settings=Settings(anonymized_telemetry=False))


def _embedding():
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
    class CachedMiniLM(ONNXMiniLM_L6_V2):
        def _download_model_if_not_exists(self):
            # Chroma calls this on every embedding. Override its automatic
            # downloader so an incomplete cache cannot silently access network.
            folder = Path(self.DOWNLOAD_PATH) / self.EXTRACTED_FOLDER_NAME
            required = ('config.json', 'model.onnx', 'special_tokens_map.json',
                        'tokenizer_config.json', 'tokenizer.json', 'vocab.txt')
            if not all((folder / name).is_file() for name in required):
                raise FileNotFoundError('MiniLM cache incomplete; explicitly rerun with --download-model')
    return CachedMiniLM(preferred_providers=['CPUExecutionProvider'])


def embedding_fingerprint():
    """Hash actual inference assets and implementation, not just a model label."""
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
    import inspect
    folder = Path(ONNXMiniLM_L6_V2.DOWNLOAD_PATH) / ONNXMiniLM_L6_V2.EXTRACTED_FOLDER_NAME
    assets = {}
    for name in ('model.onnx', 'config.json', 'special_tokens_map.json',
                 'tokenizer_config.json', 'tokenizer.json', 'vocab.txt'):
        path = folder / name
        if not path.is_file():
            raise FileNotFoundError('MiniLM cache incomplete; explicitly rerun with --download-model')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        assets[name] = digest.hexdigest()
    return {'embedding_assets': assets,
            'embedding_implementation': hashlib.sha256(inspect.getsource(ONNXMiniLM_L6_V2).encode()).hexdigest(),
            'embedding_packages': {name: importlib.metadata.version(name)
                                   for name in ('chromadb', 'tokenizers', 'onnxruntime')},
            'provider': 'CPUExecutionProvider'}


def _copy_file(old, new, name, count):
    copied = 0
    for offset in range(0, count, 100):
        page = old.get(where={'relative_path': name}, include=['documents', 'metadatas', 'embeddings'],
                       limit=100, offset=offset)
        if not page['ids']:
            raise RuntimeError(f'Active generation has missing chunks: {name}')
        new.upsert(ids=page['ids'], documents=page['documents'], metadatas=page['metadatas'],
                   embeddings=page['embeddings'])
        copied += len(page['ids'])
    if copied != count:
        raise RuntimeError(f'Active generation chunk count mismatch: {name}')


def _write_file(collection, chunks, name, digest):
    for start in range(0, len(chunks), 100):
        batch = chunks[start:start + 100]
        ids, documents, metadata = [], [], []
        for ordinal, chunk in enumerate(batch, start):
            ids.append(hashlib.sha256(f'{name}\0{digest}\0{ordinal}'.encode()).hexdigest())
            documents.append(chunk['document'])
            meta = {key: value for key, value in chunk.items()
                    if key not in ('body', 'document', 'doc') and isinstance(value, (str, int, float, bool))}
            meta.update(relative_path=name, file_sha256=digest, chunk_ordinal=ordinal,
                        source_text=chunk['body'])
            metadata.append(meta)
        collection.upsert(ids=ids, documents=documents, metadatas=metadata)


def build(repo, db, router, full=False, download_model=False, chunker=None):
    repo, db = Path(repo).resolve(), Path(db).absolute()
    if not repo.is_dir():
        raise ValueError(f'Repository does not exist: {repo}')
    validate_db_location(repo, db)
    with operation_lock(db, create=True):
        validate_owner(repo, db)
        previous = active_index(db)
        legacy = previous is None and ((db / 'chroma.sqlite3').exists() or (db / 'git_info.json').exists())
        if legacy and not full:
            raise RuntimeError('Legacy index requires explicit --full migration; old collections are retained')
        if chunker is None:
            from rag_chunking import TokenChunker
            chunker = TokenChunker.from_default_model(allow_download=download_model)
        fingerprint = dict(chunker.fingerprint, **parser_fingerprint(), **embedding_fingerprint())
        if previous and previous['fingerprint'] != fingerprint and not full:
            raise RuntimeError('Model/tokenizer/parser fingerprint changed; use explicit --full')
        files = snapshot(repo, router)
        old_files = previous['files'] if previous else {}
        changed = {name for name, digest in files.items()
                   if full or old_files.get(name, {}).get('sha256') != digest}
        deleted = sorted(set(old_files) - set(files))
        client = _client(db)
        ef = _embedding()
        old = client.get_collection(previous['collection'], embedding_function=ef) if previous and not full else None
        if old is not None and old.count() != sum(record['chunks'] for record in old_files.values()):
            raise RuntimeError('Active generation count mismatch; preserve DB for diagnosis')
        if previous and not full and not changed and not deleted:
            return {'mode': 'skip', 'reason': 'content_unchanged', 'collection_count': old.count(),
                    'db_path': str(db), 'collection_name': previous['collection']}
        generation = 'rag_gen_' + uuid.uuid4().hex
        state = {'repo_dir': str(repo), 'schema': SCHEMA, 'collection': generation, 'status': 'building'}
        atomic_json(db / PENDING, state)
        try:
            # No active collection is ever changed, even by --full.
            collection = client.create_collection(generation, embedding_function=ef,
                                                  metadata={'hnsw:space': 'cosine'})
            manifest_files = {}
            for name, digest in files.items():
                if name not in changed:
                    count = old_files[name]['chunks']
                    _copy_file(old, collection, name, count)
                else:
                    path = repo / name
                    if not safe_source(repo, path):
                        raise RuntimeError(f'Source became unsafe: {name}')
                    chunks = router.get_parser_for_path(name).parse_file(str(path))
                    chunks = chunker.prepare(chunks)
                    _write_file(collection, chunks, name, digest)
                    count = len(chunks)
                manifest_files[name] = {'sha256': digest, 'chunks': count}
            if snapshot(repo, router) != files:
                raise RuntimeError('Source changed during build; old generation remains active, retry')
            expected = sum(item['chunks'] for item in manifest_files.values())
            if collection.count() != expected:
                raise RuntimeError('Staging generation is incomplete; old generation remains active')
            manifest = {'schema': SCHEMA, 'repo_dir': str(repo), 'collection': generation,
                        'fingerprint': fingerprint, 'files': manifest_files}
            # This replace is the sole commit point; exceptions before it cannot
            # expose the new collection. No fallible state write after it.
            atomic_json(db / ACTIVE, manifest)
        except BaseException:
            # Marker/staging retained for diagnosis; next build gets a fresh UUID.
            raise
        try:
            (db / PENDING).unlink()
        except OSError:
            pass  # Stale marker is not the authority; ACTIVE is.
        return {'mode': 'full' if full or previous is None else 'incremental',
                'parsed_files': len(changed), 'reused_files': len(files) - len(changed),
                'deleted_files': len(deleted), 'collection_count': expected,
                'collection_name': generation, 'db_path': str(db), 'errors': 0}
