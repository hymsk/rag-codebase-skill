"""Lossless source slicing with the embedding tokenizer's actual token budget.

Only ``document`` is embedding input. Other fields remain provenance/context,
not text to prepend again at storage time. Offsets are Python character offsets
relative to the input chunk's ``source_field``, not file bytes. Source lines are
1-based and inclusive (a trailing newline belongs to the preceding line).
"""

from __future__ import annotations

from bisect import bisect_left
from copy import deepcopy
import hashlib
from pathlib import Path


class TokenChunker:
    """Prepare parser dictionaries without mutating them or a shared tokenizer.

    ``tokenizer`` implements the tokenizers.Tokenizer API (including to_str,
    no_truncation, no_padding and encode). Custom tokenizers must be deepcopyable.
    Documents never use token decode: normalization must not rewrite source.
    """

    CHUNKER_VERSION = "1"

    def __init__(self, tokenizer, max_tokens=256):
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        # Hash before disabling anything. A supplied object has no original file;
        # from_default_model replaces this serialization hash with raw file SHA256.
        serialized = tokenizer.to_str().encode("utf-8")
        self._tokenizer_sha256 = hashlib.sha256(serialized).hexdigest()
        self._tokenizer = deepcopy(tokenizer)
        if self._tokenizer is tokenizer:
            raise TypeError("tokenizer must support an independent deepcopy")
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        self.max_tokens = max_tokens
        if self._count("") > max_tokens:
            raise ValueError("max_tokens cannot fit the tokenizer's special tokens")

    @classmethod
    def from_default_model(cls, allow_download=False) -> TokenChunker:
        """Load cached MiniLM tokenizer only; network access requires opt-in.

        No ONNX inference session or EF shared tokenizer property is accessed.
        Chroma's downloader (and its archive verification) owns model downloads.
        """
        from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
        from tokenizers import Tokenizer

        path = (Path(ONNXMiniLM_L6_V2.DOWNLOAD_PATH)
                / ONNXMiniLM_L6_V2.EXTRACTED_FOLDER_NAME / "tokenizer.json")
        if allow_download:
            ef = ONNXMiniLM_L6_V2()
            ef._download_model_if_not_exists()
        if not path.is_file():
            raise FileNotFoundError(
                "MiniLM tokenizer cache is missing; rerun with --download-model "
                "to explicitly allow the model download"
            )
        raw = path.read_bytes()
        result = cls(Tokenizer.from_str(raw.decode("utf-8")))
        result._tokenizer_sha256 = hashlib.sha256(raw).hexdigest()
        return result

    @property
    def fingerprint(self) -> dict:
        """JSON-compatible embedding/chunking identity (a fresh dictionary)."""
        return {
            "model": "all-MiniLM-L6-v2",
            "dimensions": 384,
            "distance": "cosine",
            "tokenizer_sha256": self._tokenizer_sha256,
            "max_tokens": self.max_tokens,
            "chunker_version": self.CHUNKER_VERSION,
        }

    def _count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=True).ids)

    def _slices(self, text: str):
        """Yield safe, non-overlapping character spans, including empty input.

        Token counts need not be monotonic (e.g. WordPiece unknown words). Binary
        search only keeps actually encoded safe candidates; maximal packing is
        deliberately not promised. Each iteration consumes >=1 character or
        raises an explicit budget error rather than spinning or dropping text.
        """
        if not text:
            yield 0, 0
            return
        start = 0
        while start < len(text):
            # Bound character work per probe, not token count. Re-encoding the
            # entire remaining megabyte for every tiny chunk would be quadratic.
            # This may underfill whitespace/[UNK] chunks but never drops text.
            limit = min(len(text), start + max(1024, self.max_tokens * 16))
            if self._count(text[start:limit]) <= self.max_tokens:
                yield start, limit
                start = limit
                continue
            low, high = start, limit
            while low + 1 < high:
                middle = (low + high) // 2
                if self._count(text[start:middle]) <= self.max_tokens:
                    low = middle
                else:
                    high = middle
            if low == start:
                raise ValueError(
                    f"max_tokens={self.max_tokens} cannot fit a source character "
                    f"at character offset {start} including special tokens"
                )
            yield start, low
            start = low

    @staticmethod
    def _text(chunk: dict, field: str) -> str:
        value = chunk.get(field, "")
        if value is None:
            return ""
        if not isinstance(value, str):
            raise TypeError(f"chunk {field} must be a string")
        return value

    def prepare(self, chunks: list[dict]) -> list[dict]:
        """Return embedding-ready chunks; preserve body and doc losslessly.

        Body gets the entire budget first. Signature context is included only
        when its complete labeled form takes <=1/4 of the budget and the final
        document fits; otherwise context_omitted='signature' records omission.
        Original signature/doc metadata is retained on body records. Every
        nonempty doc is separately sliced into comment records; their line range
        is explicitly the associated symbol's range, NOT a guessed doc location.
        Join bodies by source_field to reconstruct each input field exactly.
        """
        prepared = []
        for source in chunks:
            body = self._text(source, "body")
            doc = self._text(source, "doc")
            signature = self._text(source, "signature")
            label = "# Section: " if source.get("language") == "markdown" else "// Signature: "
            context = label + signature + "\n" if signature else ""
            bounded_context = context and self._count(context) <= self.max_tokens // 4
            fields = [("body", body)]
            if doc:
                fields.append(("doc", doc))
            for field, text in fields:
                newlines = [index for index, char in enumerate(text) if char == "\n"]
                for start, end in self._slices(text):
                    chunk = deepcopy(source)
                    piece = text[start:end]
                    document = piece
                    omitted = "signature" if signature else ""
                    if bounded_context and self._count(context + piece) <= self.max_tokens:
                        document = context + piece
                        omitted = ""
                    chunk.update({
                        "body": piece,
                        "document": document,
                        "source_field": field,
                        "char_start": start,
                        "char_end": end,
                        "char_offset_scope": field,
                        "context_omitted": omitted,
                    })
                    if field == "doc":
                        chunk.update({
                            "type": "comment",
                            "associated_type": source.get("type", "unknown"),
                            "doc": "",
                            "line_range_kind": "associated_symbol",
                        })
                    else:
                        base = source.get("start_line", 1)
                        chunk.update({
                            "start_line": base + bisect_left(newlines, start),
                            "end_line": base + bisect_left(newlines, max(start, end - 1)),
                            "line_range_kind": "source",
                            "doc_context": "separate_comment" if doc else "none",
                        })
                    # Verify exactly what the caller will embed, after composition.
                    chunk["token_count"] = self._count(document)
                    if chunk["token_count"] > self.max_tokens:
                        raise ValueError("Final document exceeds max_tokens")
                    prepared.append(chunk)
        return prepared
