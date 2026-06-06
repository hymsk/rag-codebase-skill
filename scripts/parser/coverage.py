"""Preserve top-level source outside semantic nodes and report syntax errors."""


def uncovered_chunks(parser, root, source_bytes, file_path, accepted_types):
    if root.has_error:
        raise ValueError(f"tree-sitter syntax error in {file_path}; index not updated")
    cursor = 0
    chunks = []
    for node in root.children:
        if node.type not in accepted_types:
            continue
        gap = source_bytes[cursor:node.start_byte].decode('utf-8', errors='replace')
        if gap.strip():
            line = source_bytes[:cursor].count(b'\n') + 1
            chunks.extend(parser._split_plain_text(gap, file_path, line))
        cursor = node.end_byte
    gap = source_bytes[cursor:].decode('utf-8', errors='replace')
    if gap.strip():
        line = source_bytes[:cursor].count(b'\n') + 1
        chunks.extend(parser._split_plain_text(gap, file_path, line))
    return chunks
