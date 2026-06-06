"""Shared repository identity and conservative file discovery (stdlib only)."""

import hashlib
import os
import stat
import subprocess
from pathlib import Path

MAX_FILE_BYTES = 1024 * 1024
SKIP_DIRS = {'.git', '.svn', '.hg', '.venv', 'venv', '__pycache__',
             'node_modules', 'vendor', 'third_party', 'build', 'dist'}


def project_id(repo_dir):
    canonical = str(Path(repo_dir).resolve())
    label = Path(canonical).name or 'root'
    return label + '-' + hashlib.sha256(os.fsencode(canonical)).hexdigest()[:16]


def safe_source(root, path):
    """Reject links (including parent links), special files and oversized files.

    This is a static-worktree check, not a sandbox against concurrent replacement.
    """
    root = Path(root).resolve()
    path = Path(os.path.abspath(path))
    try:
        relative = path.relative_to(root)
        current = root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                return False
        info = path.stat()
        return stat.S_ISREG(info.st_mode) and info.st_size <= MAX_FILE_BYTES
    except (ValueError, OSError):
        return False


def source_files(root):
    """Use Git's nested ignore semantics; non-Git uses conservative os.walk."""
    root = Path(root).resolve()
    try:
        check = subprocess.run(['git', 'rev-parse', '--is-inside-work-tree'],
                               cwd=root, capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        check = None
    if check and check.stdout.strip() == b'true':
        result = subprocess.run(
            ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z', '--', '.'],
            cwd=root, capture_output=True, check=True)
        candidates = (root / os.fsdecode(name) for name in sorted(set(result.stdout.split(b'\0'))) if name)
        for path in candidates:
            if safe_source(root, path):
                yield str(path)
        return
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d.lower() not in SKIP_DIRS
                         and not (Path(directory) / d).is_symlink())
        for name in sorted(files):
            path = Path(directory) / name
            if safe_source(root, path):
                yield str(path)
