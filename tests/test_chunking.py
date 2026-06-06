"""Offline chunker regressions; real tokenizers use a synthetic local vocab."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rag_chunking import TokenChunker


class CharacterTokenizer:
    """Strict double: one token per code point plus two special tokens."""

    def __init__(self):
        self.truncation = 5
        self.padding = 12

    def to_str(self):
        return json.dumps(vars(self), sort_keys=True)

    def no_truncation(self):
        self.truncation = None

    def no_padding(self):
        self.padding = None

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens
        ids = [0] * (len(text) + 2)
        if self.truncation:
            ids = ids[:self.truncation]
        if self.padding:
            ids += [0] * max(0, self.padding - len(ids))
        return types.SimpleNamespace(ids=ids)

    def decode(self, *args, **kwargs):
        raise AssertionError("Source must never be reconstructed by token decode")


def source(body="", **extra):
    result = {"body": body, "signature": "", "doc": "", "file_path": "/synthetic/a.py",
              "type": "function", "name": "sample", "language": "python",
              "start_line": 7, "end_line": 40, "custom": {"keep": True}}
    result.update(extra)
    return result


class ChunkerTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()
        self.chunker = TokenChunker(self.tokenizer, max_tokens=18)

    def assert_lossless(self, text):
        chunks = self.chunker.prepare([source(text)])
        self.assertEqual("".join(c["body"] for c in chunks), text)
        offset = 0
        for chunk in chunks:
            self.assertEqual(chunk["char_start"], offset)
            self.assertEqual(chunk["body"], text[chunk["char_start"]:chunk["char_end"]])
            offset = chunk["char_end"]
            self.assertEqual(chunk["start_line"], 7 + text.count("\n", 0, chunk["char_start"]))
            self.assertEqual(chunk["end_line"], 7 + text.count("\n", 0, max(chunk["char_start"], offset - 1)))
            self.assertEqual(chunk["token_count"], len(chunk["document"]) + 2)
            self.assertLessEqual(chunk["token_count"], 18)
        self.assertEqual(offset, len(text))
        return chunks

    def test_unicode_whitespace_and_lines_lossless(self):
        for text in ["", " \t\r\n" * 100, "中文单行" * 700, "🙂👨‍👩‍👧‍👦" * 200,
                     "  first\r\nsecond\n\n last \t\n", "x" * 16 + "\nnext\n"]:
            with self.subTest(text=text[:30]):
                self.assert_lossless(text)

    def test_deterministic_randomized_spans(self):
        rng = random.Random(4)
        for _ in range(60):
            self.assert_lossless("".join(rng.choice("ab中🙂 \r\n\t") for _ in range(rng.randrange(400))))

    def test_shared_tokenizer_and_input_unchanged(self):
        original = source("many lines\n" * 30, doc="documentation" * 20)
        before = deepcopy(original)
        state = self.tokenizer.to_str()
        result = self.chunker.prepare([original])
        self.assertEqual(original, before)
        self.assertEqual(self.tokenizer.to_str(), state)
        result[0]["custom"]["keep"] = False
        self.assertTrue(original["custom"]["keep"])

    def test_doc_never_displaces_body_or_disappears(self):
        body = "body with padding   \n" * 40
        doc = " 中文说明🙂\n" * 100
        chunks = self.chunker.prepare([source(body, doc=doc, signature="long signature" * 90)])
        body_chunks = [c for c in chunks if c["source_field"] == "body"]
        comments = [c for c in chunks if c["source_field"] == "doc"]
        self.assertEqual([c["body"] for c in body_chunks],
                         [c["body"] for c in self.chunker.prepare([source(body)])])
        self.assertEqual("".join(c["body"] for c in body_chunks), body)
        self.assertEqual("".join(c["body"] for c in comments), doc)
        for c in chunks:
            self.assertLessEqual(c["token_count"], 18)
            self.assertEqual(c["context_omitted"], "signature")
        for c in comments:
            self.assertEqual((c["start_line"], c["end_line"]), (7, 40))
            self.assertEqual(c["line_range_kind"], "associated_symbol")
            self.assertEqual(c["type"], "comment")
            self.assertEqual(c["associated_type"], "function")
            self.assertEqual(c["body"], doc[c["char_start"]:c["char_end"]])

    def test_short_context_composed_with_final_budget_check(self):
        chunker = TokenChunker(self.tokenizer, 128)
        for language, label in [("python", "// Signature: "), ("markdown", "# Section: ")]:
            c = chunker.prepare([source(" body ", signature="f()", language=language)])[0]
            self.assertEqual(c["document"], label + "f()\n body ")
            self.assertEqual(c["context_omitted"], "")
        chunks = chunker.prepare([source("a" * 126, signature="f()")])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["context_omitted"], "signature")
        self.assertEqual(chunks[0]["token_count"], 128)

    def test_empty_input_and_null_optional_fields(self):
        self.assertEqual(self.chunker.prepare([]), [])
        self.assertEqual(self.chunker.prepare([source(None, signature=None, doc=None)])[0]["body"], "")

    def test_invalid_field_types_fail_explicitly(self):
        for field in ("body", "doc", "signature"):
            with self.assertRaisesRegex(TypeError, field):
                self.chunker.prepare([source(**{field: ["not text"]})])

    def test_tiny_budgets_fail_without_looping(self):
        for budget in (0, -1, 1, True, 1.5, "18"):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                TokenChunker(self.tokenizer, budget)
        chunker = TokenChunker(self.tokenizer, 2)
        self.assertEqual(chunker.prepare([source("")])[0]["token_count"], 2)
        with self.assertRaisesRegex(ValueError, "cannot fit a source character"):
            chunker.prepare([source("🙂")])
        chunks = TokenChunker(self.tokenizer, 3).prepare([source("中🙂a")])
        self.assertEqual([c["body"] for c in chunks], ["中", "🙂", "a"])

    def test_long_input_uses_bounded_probes(self):
        class BoundedTokenizer(CharacterTokenizer):
            def encode(self, text, add_special_tokens=True):
                if len(text) > 1024:
                    raise AssertionError("Do not repeatedly encode the entire remaining source")
                return super().encode(text, add_special_tokens=add_special_tokens)

        body = "中" * 10000
        chunks = TokenChunker(BoundedTokenizer(), 18).prepare([source(body)])
        self.assertEqual("".join(c["body"] for c in chunks), body)
        self.assertTrue(all(c["token_count"] <= 18 for c in chunks))

    def test_fingerprint_hashes_unmodified_serialization(self):
        expected = hashlib.sha256(self.tokenizer.to_str().encode()).hexdigest()
        value = self.chunker.fingerprint
        self.assertEqual(value["tokenizer_sha256"], expected)
        self.assertEqual(value["model"], "all-MiniLM-L6-v2")
        self.assertEqual(value["dimensions"], 384)
        self.assertEqual(value["distance"], "cosine")
        self.assertEqual(value["max_tokens"], 18)
        self.assertEqual(json.loads(json.dumps(value)), value)
        value["max_tokens"] = 0
        self.assertEqual(self.chunker.fingerprint["max_tokens"], 18)


class DefaultModelTests(unittest.TestCase):
    def test_cache_policy_hash_and_no_session_with_doubles(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic-extracted" / "tokenizer.json"
            calls = []

            class EF:
                DOWNLOAD_PATH = temporary
                EXTRACTED_FOLDER_NAME = "synthetic-extracted"

                def __init__(self):
                    calls.append("init")

                def _download_model_if_not_exists(self):
                    calls.append("download")
                    path.parent.mkdir(exist_ok=True)
                    path.write_bytes(b'  {"synthetic": true}\n')

                @property
                def model(self):
                    raise AssertionError("ONNX session must not be loaded")

                @property
                def tokenizer(self):
                    raise AssertionError("Shared tokenizer must not be touched")

            ef_module = types.ModuleType("chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2")
            ef_module.ONNXMiniLM_L6_V2 = EF
            token_module = types.ModuleType("tokenizers")
            token_module.Tokenizer = types.SimpleNamespace(from_str=lambda raw: CharacterTokenizer())
            with patch.dict(sys.modules, {ef_module.__name__: ef_module, "tokenizers": token_module}):
                with self.assertRaisesRegex(FileNotFoundError, "--download-model"):
                    TokenChunker.from_default_model()
                self.assertEqual(calls, [])
                chunker = TokenChunker.from_default_model(allow_download=True)
                self.assertEqual(calls, ["init", "download"])
                expected = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(chunker.fingerprint["tokenizer_sha256"], expected)
                self.assertEqual(chunker.fingerprint["max_tokens"], 256)
                calls.clear()
                self.assertEqual(TokenChunker.from_default_model().fingerprint, chunker.fingerprint)
                self.assertEqual(calls, [])


try:
    import tokenizers
except ImportError:
    tokenizers = None


@unittest.skipIf(tokenizers is None, "optional tokenizers dependency is not installed")
class RealTokenizerTests(unittest.TestCase):
    def make_tokenizer(self):
        vocab = {token: index for index, token in enumerate(
            ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "##a", "中", "文", "🙂", "/", ":"])}
        t = tokenizers.Tokenizer(tokenizers.models.WordPiece(vocab, unk_token="[UNK]"))
        t.normalizer = tokenizers.normalizers.BertNormalizer()
        t.pre_tokenizer = tokenizers.pre_tokenizers.BertPreTokenizer()
        t.post_processor = tokenizers.processors.TemplateProcessing(
            single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
        t.decoder = tokenizers.decoders.WordPiece()
        t.enable_truncation(max_length=8)
        t.enable_padding(length=32, pad_id=0, pad_token="[PAD]")
        return t

    def test_real_encode_budget_and_preservation(self):
        tokenizer = self.make_tokenizer()
        state = tokenizer.to_str()
        chunker = TokenChunker(tokenizer, 24)
        checker = tokenizers.Tokenizer.from_str(state)
        checker.no_truncation()
        checker.no_padding()
        for body in ("中文" * 1000, "🙂\t  " * 300, "  a\r\n" * 200,
                     "a" * 1000, " \r\n\t" * 100, ""):
            doc = "中文 documentation\n" * 100
            chunks = chunker.prepare([source(body, doc=doc, signature="a")])
            for field, original in (("body", body), ("doc", doc)):
                self.assertEqual("".join(c["body"] for c in chunks if c["source_field"] == field), original)
            for c in chunks:
                self.assertEqual(len(checker.encode(c["document"], add_special_tokens=True).ids), c["token_count"])
                self.assertLessEqual(c["token_count"], 24)
        self.assertEqual(tokenizer.to_str(), state)

    def test_nonmonotonic_wordpiece_counts_still_safe(self):
        tokenizer = self.make_tokenizer()
        tokenizer.no_truncation()
        tokenizer.no_padding()
        self.assertGreater(len(tokenizer.encode("a" * 100).ids), len(tokenizer.encode("a" * 101).ids))
        chunks = TokenChunker(tokenizer, 8).prepare([source("a" * 101 + " 中" * 150)])
        self.assertEqual("".join(c["body"] for c in chunks), "a" * 101 + " 中" * 150)
        self.assertTrue(all(len(tokenizer.encode(c["document"]).ids) <= 8 for c in chunks))

    def test_actual_chroma_constants_cache_only_and_original_file_hash(self):
        try:
            from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2 as EF
        except ImportError:
            self.skipTest("optional chromadb dependency is not installed")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / EF.EXTRACTED_FOLDER_NAME / "tokenizer.json"
            path.parent.mkdir()
            raw = (" \n" + self.make_tokenizer().to_str(pretty=True) + "\n").encode()
            path.write_bytes(raw)
            with patch.object(EF, "DOWNLOAD_PATH", Path(temporary)), \
                 patch.object(EF, "__init__", side_effect=AssertionError("Do not instantiate EF in cache-only mode")):
                chunker = TokenChunker.from_default_model()
            self.assertEqual(chunker.fingerprint["tokenizer_sha256"], hashlib.sha256(raw).hexdigest())
            chunks = chunker.prepare([source("中文" * 400)])
            self.assertGreater(len(chunks), 1)
            self.assertTrue(all(c["token_count"] <= 256 for c in chunks))


if __name__ == "__main__":
    unittest.main()
