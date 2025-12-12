"""
intelligent_rag_chunker_fixed.py

Production-ready IntelligentChunker:
- Semantic chunking (embeddings or naive fallback)
- Table/CSV/TSV row-level atomic chunking (dynamic detection)
- Deterministic chunk IDs (stable across re-index)
- Refinement pass that prevents merging atomic chunks (table rows/code/list)
- Overlap generation
- Debug helpers
"""

from __future__ import annotations

import os
import re
import json
import hashlib
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Optional dependencies (kept optional to allow graceful fallback)
try:
    import numpy as np
except Exception:
    np = None

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    SentenceTransformer = None
    cosine_similarity = None


# ----------------------------- Utilities & Data Classes -----------------------------

def _deterministic_id(prefix: str = "chunk", source_path: Optional[str] = None, text: Optional[str] = None) -> str:
    """
    Deterministic id generator: stable when source_path+text provided.
    Falls back to uuid when not provided (backwards compatible).
    """
    if source_path is not None and text is not None:
        data = (str(source_path) + "|" + text).encode("utf-8")
        return f"{prefix}_{hashlib.sha1(data).hexdigest()[:16]}"
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    words = len(text.split())
    return max(1, int(words * 1.3))


def _split_sentences(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if s.strip()]


@dataclass
class Block:
    type: str
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)


# ----------------------------- Extractor -----------------------------

# Broader table detector: catches pipes, tabs, commas
_TABLE_LINE_RE = re.compile(r"[|\t,]")

_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6}\s+|[A-Z][A-Za-z0-9\- ]{2,120}:$)")
_LIST_RE = re.compile(r"^(\s*[-*+]\s+|\s*\d+\.|\s*•\s+)")
_CODE_FENCE_RE = re.compile(r"^\s*```")


def extract_structured_blocks_from_text(text: str) -> List[Block]:
    """
    Splits raw text into structural blocks:
    - headers
    - lists
    - code blocks
    - tables (keeps entire table as block for table transformer)
    - paragraphs
    """
    lines = text.splitlines()
    blocks: List[Block] = []
    buf: List[str] = []
    cur_type = "paragraph"

    def flush():
        nonlocal buf, cur_type
        if not buf:
            return
        txt = "\n".join(buf).rstrip()
        if txt:
            blocks.append(Block(type=cur_type, text=txt))
        buf = []
        cur_type = "paragraph"

    in_code = False
    for line in lines:
        s = line.rstrip()
        if s.strip() == "":
            flush()
            continue

        if _CODE_FENCE_RE.match(s):
            if in_code:
                buf.append(s)
                flush()
                in_code = False
            else:
                flush()
                in_code = True
                cur_type = "code"
                buf.append(s)
            continue

        if in_code:
            buf.append(s)
            continue

        if _HEADING_RE.match(s):
            flush()
            blocks.append(Block(type="header", text=s.strip()))
            continue

        if _LIST_RE.match(s):
            if cur_type != "list":
                flush()
                cur_type = "list"
            buf.append(s)
            continue

        # Table detection: if line contains any table separator char OR line composed of table separator chars
        if _TABLE_LINE_RE.search(s) or all(c in ('-', '|', ' ', ':') for c in s.strip()):
            if cur_type != "table":
                flush()
                cur_type = "table"
            buf.append(s)
            continue

        if cur_type != "paragraph":
            flush()
            cur_type = "paragraph"

        buf.append(s)

    flush()
    return blocks


# ----------------------------- Embedding helper -----------------------------

_embedding_model = None


def load_embedding_model(name: str = "all-MiniLM-L6-v2"):
    global _embedding_model
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed; pip install sentence-transformers")
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(name)
    return _embedding_model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Returns dense vectors for texts. Falls back to deterministic synthetic vectors when
    sentence-transformers is not present.
    """
    if SentenceTransformer is not None:
        model = load_embedding_model()
        embs = model.encode(texts, convert_to_numpy=True)
        return embs
    if np is not None:
        out = []
        for t in texts:
            v = np.zeros(128, dtype=float)
            for i, ch in enumerate(t[:1024]):
                v[i % 128] += ord(ch)
            n = np.linalg.norm(v) + 1e-12
            out.append((v / n).tolist())
        return out
    return [[len(t)] for t in texts]


# ----------------------------- IntelligentChunker -----------------------------

class IntelligentChunker:
    def __init__(
            self,
            max_tokens: int = 1200,
            target_tokens: int = 800,
            overlap_tokens: int = 150,
            semantic_threshold: float = 0.65,
            embed_model_name: Optional[str] = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.semantic_threshold = semantic_threshold
        self.embed_model_name = embed_model_name

    # -------------------------
    # Table transformer
    # -------------------------
    def _transform_universal_table(self, table_text: str) -> List[str]:
        """
        Return a list of raw data row strings for the table_text.
        - auto-detect delimiter (pipe, tab, comma)
        - preserve empty columns (do not filter them)
        - handle tables with or without explicit separator lines
        """
        lines = [line.rstrip("\n") for line in table_text.splitlines() if line.strip()]

        if not lines:
            return []

        # explore candidates and score them on first N lines
        sep_candidates = ["|", "\t", ","]
        scores = {sep: sum(1 for ln in lines[:10] if sep in ln) for sep in sep_candidates}
        best_sep = max(scores, key=scores.get)
        if scores[best_sep] == 0:
            # no obvious separator found -> fall back to returning each non-empty line
            return [ln.strip() for ln in lines if ln.strip()]

        data_lines: List[str] = []
        found_sep_line = False
        for i, ln in enumerate(lines):
            line = ln.strip()
            if not line:
                continue
            parts = [p for p in line.split(best_sep)]  # keep empties
            # detect a separator line like "|----|-----|" or "----,----"
            if all(re.fullmatch(r'[-=:\s]*', p) for p in parts) and len(parts) > 1:
                found_sep_line = True
                continue
            # if separator line found, treat following lines as data
            if found_sep_line and len(parts) >= 1:
                data_lines.append(line)
            else:
                # heuristic: if first line is likely header (has >1 column), skip storing it as data
                if i == 0 and len(parts) > 1:
                    # treat it as header; don't store as data
                    continue
                else:
                    data_lines.append(line)

        if not data_lines:
            # fallback
            return [ln.strip() for ln in lines if ln.strip()]
        return data_lines

    # -------------------------
    # Main chunking
    # -------------------------
    def chunk_document(self, text: str, meta: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        if meta is None:
            meta = {}

        # Step 1: extract structural blocks
        blocks = extract_structured_blocks_from_text(text)
        atomic_blocks = blocks.copy()

        paras = [b.text for b in atomic_blocks]

        # Step 2: semantic similarity boundaries (embeddings if possible)
        try:
            embeddings = embed_texts(paras) if paras else None
        except Exception:
            embeddings = None

        boundaries = set()
        if embeddings is not None and cosine_similarity is not None and len(paras) > 1:
            try:
                sims = cosine_similarity(embeddings)
                for i in range(len(paras) - 1):
                    # skip table/list/code boundaries from being fused/compared
                    if atomic_blocks[i].type in ("table", "list", "code") or atomic_blocks[i + 1].type in ("table", "list", "code"):
                        continue
                    if float(sims[i, i + 1]) < self.semantic_threshold:
                        boundaries.add(i)
            except Exception:
                embeddings = None

        # fallback naive similarity if embeddings missing
        if embeddings is None:
            for i in range(len(paras) - 1):
                if atomic_blocks[i].type in ("table", "list", "code") or atomic_blocks[i + 1].type in ("table", "list", "code"):
                    continue
                if _naive_similarity(paras[i], paras[i + 1]) < 0.28:  # slightly relaxed
                    boundaries.add(i)

        chunks: List[Chunk] = []
        cur_pack: List[str] = []
        cur_meta_sources: List[str] = []

        def flush_pack():
            nonlocal cur_pack, cur_meta_sources
            if not cur_pack:
                return

            text_pack = "\n\n".join(cur_pack).strip()

            # FIXED HERE ↓↓↓
            types = list(dict.fromkeys([m["type"] for m in cur_meta_sources]))
            atomic_flags = [m.get("atomic", False) for m in cur_meta_sources]

            meta_copy = dict(meta)
            meta_copy["block_types"] = types
            meta_copy["atomic"] = any(atomic_flags)
            # FIX END ↑↑↑

            cid = _deterministic_id("chunk", source_path=meta.get("source_path"), text=text_pack)
            chunks.append(Chunk(id=cid, text=text_pack, meta=meta_copy))

            cur_pack = []
            cur_meta_sources = []
        # ----------------------------------------------------
        # chunking loop with table row atomic fix
        # ----------------------------------------------------
        for idx, b in enumerate(atomic_blocks):
            # Pre-split table blocks into rows
            current_block_texts = [b.text]
            if b.type == "table":
                current_block_texts = self._transform_universal_table(b.text)

            is_semantic_boundary = idx in boundaries

            for block_text in current_block_texts:
                block_text = block_text.strip()
                if not block_text:
                    continue

                block_tokens = _approx_tokens(block_text)

                # If block_text too big, split by sentences
                if block_tokens > self.max_tokens:
                    sents = _split_sentences(block_text)
                    pack: List[str] = []
                    for s in sents:
                        if _approx_tokens(" ".join(pack + [s])) <= self.target_tokens:
                            pack.append(s)
                        else:
                            if pack:
                                cur_pack.extend(pack)
                                cur_meta_sources.extend([{"type": b.type, "atomic": (b.type == "table")}])
                                flush_pack()
                            pack = [s]
                    if pack:
                        cur_pack.extend(pack)
                        cur_meta_sources.extend([{"type": b.type, "atomic": (b.type == "table")}])
                        flush_pack()
                    continue

                # Current pack attempt
                candidate_text = "\n\n".join(cur_pack + [block_text]).strip()

                # If candidate exceeds target or previous block was a semantic boundary -> flush
                if _approx_tokens(candidate_text) > self.target_tokens or is_semantic_boundary:
                    if not cur_pack:
                        cur_pack.append(block_text)
                        cur_meta_sources.append({"type": b.type, "atomic": (b.type == "table")})
                        flush_pack()
                    else:
                        flush_pack()
                        cur_pack.append(block_text)
                        cur_meta_sources.append({"type": b.type, "atomic": (b.type == "table")})
                        is_semantic_boundary = False
                else:
                    cur_pack.append(block_text)
                    cur_meta_sources.append({"type": b.type, "atomic": (b.type == "table")})

                # If this block originated from a table, force immediate flush to make row atomic
                if b.type == "table":
                    # Create atomic chunk for this row
                    flush_pack()
                    is_semantic_boundary = False

            # If semantic boundary triggered earlier and there's a pack, flush it
            if is_semantic_boundary and cur_pack:
                flush_pack()

        flush_pack()

        # ----------------------------------------------------
        # refinement pass: split oversize chunks and avoid merging atomic ones
        # ----------------------------------------------------
        chunks = self._refinement_pass(chunks)

        # overlap creation
        if self.overlap_tokens > 0:
            chunks = self._apply_sentence_overlaps(chunks)

        return chunks

    def _refinement_pass(self, chunks: List[Chunk]) -> List[Chunk]:
        out: List[Chunk] = []
        i = 0
        while i < len(chunks):
            c = chunks[i]
            tkn = _approx_tokens(c.text)
            # check atomic flags or block_types
            c_atomic = bool(c.meta.get("atomic") or ("table" in c.meta.get("block_types", [])) or ("code" in c.meta.get("block_types", [])))
            if tkn > self.max_tokens:
                sents = _split_sentences(c.text)
                pack: List[str] = []
                for s in sents:
                    if _approx_tokens(" ".join(pack + [s])) <= self.target_tokens:
                        pack.append(s)
                    else:
                        if pack:
                            out.append(Chunk(id=_deterministic_id("chunk", None, " ".join(pack)), text=" ".join(pack), meta=c.meta.copy()))
                        pack = [s]
                if pack:
                    out.append(Chunk(id=_deterministic_id("chunk", None, " ".join(pack)), text=" ".join(pack), meta=c.meta.copy()))
            else:
                # attempt merge only if both current and next are non-atomic
                if tkn < int(self.target_tokens * 0.25) and i + 1 < len(chunks):
                    nxt = chunks[i + 1]
                    nxt_atomic = bool(nxt.meta.get("atomic") or ("table" in nxt.meta.get("block_types", [])) or ("code" in nxt.meta.get("block_types", [])))
                    if not c_atomic and not nxt_atomic:
                        merged_text = (c.text + "\n\n" + nxt.text).strip()
                        merged_meta = {**c.meta}
                        merged_meta_types = list(dict.fromkeys(c.meta.get("block_types", []) + nxt.meta.get("block_types", [])))
                        merged_meta["block_types"] = merged_meta_types
                        out.append(Chunk(id=_deterministic_id("chunk", None, merged_text), text=merged_text, meta=merged_meta))
                        i += 1  # skip next (merged)
                    else:
                        out.append(c)
                else:
                    out.append(c)
            i += 1
        return out

    def _apply_sentence_overlaps(self, chunks: List[Chunk]) -> List[Chunk]:
        if len(chunks) <= 1:
            return chunks
        out: List[Chunk] = []
        avg_words_per_sentence = 14
        overlap_sent_count = max(1, int(self.overlap_tokens / (avg_words_per_sentence * 1.3)))

        for i, c in enumerate(chunks):
            out.append(c)
            if i < len(chunks) - 1:
                next_c = chunks[i + 1]
                # don't create overlaps for atomic chunks (table rows / code)
                c_atomic = bool(c.meta.get("atomic") or ("table" in c.meta.get("block_types", [])))
                next_atomic = bool(next_c.meta.get("atomic") or ("table" in next_c.meta.get("block_types", [])))
                if c_atomic or next_atomic:
                    continue
                tail_sents = _split_sentences(c.text)[-overlap_sent_count:]
                head_sents = _split_sentences(next_c.text)[:overlap_sent_count]
                if tail_sents and head_sents:
                    merged_text = (" ".join(tail_sents) + "\n\n" + " ".join(head_sents)).strip()
                    meta = {"overlap_of": [c.id, next_c.id], "block_types": list(dict.fromkeys(c.meta.get("block_types", []) + next_c.meta.get("block_types", [])))}
                    out.append(Chunk(id=_deterministic_id("overlap", None, merged_text), text=merged_text, meta=meta))
        return out


# ----------------------------- Helpers -----------------------------


def _naive_similarity(a: str, b: str) -> float:
    def trigrams(s: str):
        s2 = re.sub(r"\s+", " ", s.strip().lower())
        return {s2[i:i + 3] for i in range(max(0, len(s2) - 2))}
    ta = trigrams(a)
    tb = trigrams(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    denom = max(1, len(ta | tb))
    return inter / denom


# ----------------------------- File helpers -----------------------------


def chunk_file(path: str, chunker: Optional[IntelligentChunker] = None) -> List[Chunk]:
    if chunker is None:
        chunker = IntelligentChunker()

    ext = os.path.splitext(path)[1].lower()
    text = ""
    if ext == ".txt":
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    elif ext == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                pages = []
                for p in pdf.pages:
                    txt = p.extract_text() or ""
                    pages.append(txt)
                text = "\n\n".join(pages)
        except Exception:
            # fallback: try to read as bytes
            with open(path, 'rb') as f:
                data = f.read()
            try:
                text = data.decode('utf-8', errors='ignore')
            except Exception:
                text = ''
    elif ext in ('.docx', '.doc'):
        try:
            import docx
            doc = docx.Document(path)
            paras = [p.text for p in doc.paragraphs if p.text.strip()]
            text = "\n\n".join(paras)
        except Exception:
            text = ''
    else:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

    return chunker.chunk_document(text, meta={"source_path": path})


# ----------------------------- Debug helpers -----------------------------


def debug_find_token_in_chunks(chunks: List[Chunk], token: str, limit: int = 50):
    tok = token.lower()
    hits = []
    for c in chunks:
        if tok in c.text.lower():
            hits.append((c.id, c.meta.get("source_path"), c.text[:400]))
            if len(hits) >= limit:
                break
    print(f"Found {len(hits)} chunks containing '{token}':")
    for hid, src, snippet in hits:
        print(hid, src, "->", snippet.replace("\n", " ")[:200])
    return hits
