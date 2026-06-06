"""Split long text into chunks that fit within the model's 512-token input limit.

Splits at sentence boundaries first, then groups greedily up to MAX_CHUNK_CHARS.
1 500 chars ≈ 350–420 tokens for English/Vietnamese, safely under the 512-token ceiling.

Paragraph structure (blank lines) is preserved so the worker can reassemble
translated chunks into a coherent document.
"""
from __future__ import annotations

import re

_MAX_CHUNK_CHARS = 1_500
_PARA_SEP = re.compile(r"\n{2,}")
_SENT_SEP = re.compile(r"(?<=[.!?。！？])\s+")


def _group_sentences(sentences: list[str], max_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sent in sentences:
        sep = 1 if current else 0
        if current and current_len + sep + len(sent) > max_chars:
            chunks.append(" ".join(current))
            current = [sent]
            current_len = len(sent)
        else:
            current.append(sent)
            current_len += sep + len(sent)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_text(
    text: str, max_chars: int = _MAX_CHUNK_CHARS
) -> tuple[list[str], list[bool]]:
    """Split text into translatable chunks.

    Returns:
        chunks: flat list of text strings, each ≤ max_chars characters.
        is_para_start: parallel bool list; True when the chunk starts a new paragraph.
    """
    paragraphs = _PARA_SEP.split(text)
    chunks: list[str] = []
    is_para_start: list[bool] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        sentences = [s for s in _SENT_SEP.split(para) if s.strip()]
        if not sentences:
            sentences = [para]
        para_chunks = _group_sentences(sentences, max_chars)
        for i, chunk in enumerate(para_chunks):
            chunks.append(chunk)
            is_para_start.append(i == 0 and bool(chunks) and len(chunks) > 1)

    if not chunks:
        return [text], [False]

    # First chunk is never a paragraph break relative to nothing.
    is_para_start[0] = False
    return chunks, is_para_start


def reassemble(translated: list[str], is_para_start: list[bool]) -> str:
    """Join translated chunks, inserting paragraph breaks where indicated."""
    parts: list[str] = []
    for i, (text, new_para) in enumerate(zip(translated, is_para_start)):
        if i == 0:
            parts.append(text)
        elif new_para:
            parts.append("\n\n" + text)
        else:
            parts.append(" " + text)
    return "".join(parts)
