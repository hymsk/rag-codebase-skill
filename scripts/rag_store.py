"""Generation manifests and local POSIX operation locking (no Chroma import)."""

import contextlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile

ACTIVE = 'active_index.json'
PENDING = 'build_in_progress.json'
SCHEMA = 2


def read_json(path):
    path = Path(path)
    if path.is_symlink():
        raise RuntimeError(f'Refusing linked metadata: {path}')
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return None
    if not isinstance(value, dict):
        raise RuntimeError(f'Invalid metadata object: {path}')
    return value


def active_index(db):
    info = read_json(Path(db) / ACTIVE)
    if info is None:
        return None
    if (info.get('schema') != SCHEMA or not isinstance(info.get('repo_dir'), str)
            or not re.fullmatch(r'rag_gen_[a-f0-9]{32}', str(info.get('collection', '')))
            or not isinstance(info.get('files'), dict)
            or not isinstance(info.get('fingerprint'), dict)):
        raise RuntimeError('Invalid or unsupported active index manifest; preserve DB for diagnosis')
    for name, record in info['files'].items():
        path = Path(name)
        if (path.is_absolute() or '..' in path.parts or not name
                or not isinstance(record, dict)
                or not re.fullmatch(r'[a-f0-9]{64}', str(record.get('sha256', '')))
                or type(record.get('chunks')) is not int or record['chunks'] < 0):
            raise RuntimeError('Invalid file record in active index manifest')
    return info


def validate_owner(repo, db):
    repo = str(Path(repo).resolve())
    for name in (ACTIVE, PENDING, 'git_info.json'):
        info = read_json(Path(db) / name)
        if info is not None and (not info.get('repo_dir') or
                                 str(Path(info['repo_dir']).resolve()) != repo):
            raise RuntimeError('Database belongs to a different repository or has no owner')


def atomic_json(path, value):
    """Atomic visibility; requires an already-owned operation lock."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.rag-state-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=True, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_db_location(repo, db):
    repo, db = Path(repo).resolve(), Path(db).absolute()
    for part in (db, *db.parents):
        if part.is_symlink():
            raise RuntimeError('Database path must not contain symlinks')
    if db.resolve() == repo or repo in db.resolve().parents:
        raise RuntimeError('Keep the database outside the source repository')


@contextlib.contextmanager
def operation_lock(db, create=False):
    """Fail-fast exclusive flock. Never unlink locks or guess stale ownership.

    Query also uses exclusive mode: PersistentClient may write on open. Planning
    uses an existing lock without creating a database or lock file.
    """
    try:
        import fcntl
    except ImportError as exc:
        raise RuntimeError('POSIX flock is required; native Windows is not supported') from exc
    db = Path(db).absolute()
    for part in (db, *db.parents):
        if part.is_symlink():
            raise RuntimeError('Database path must not contain symlinks')
    if create:
        db.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock = db / '.rag.lock'
    flags = (os.O_RDWR | os.O_CREAT) if create else os.O_RDONLY
    try:
        fd = os.open(lock, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    except FileNotFoundError:
        if not create:
            if db.exists() and any(db.iterdir()):
                raise RuntimeError('Existing index has no operation lock; run an authorized build to restore locking')
            yield
            return
        raise
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise RuntimeError('Operation lock is not a regular file')
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Index is busy; retry after the active operation finishes') from exc
        yield
    finally:
        os.close(fd)
