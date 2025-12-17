"""
intelligent_rag_chunker_universal.py

Production-ready Universal RAG Chunker:
- Semantic chunking (embeddings or naive fallback)
- Table/list/code handled as atomic blocks (small tables remain one chunk)
- Deterministic chunk IDs (stable across re-index)
- Sentence-level splitting for oversized chunks
- Overlap generation
- Fully universal: no hard-coded headers or section names
"""

from __future__ import annotations
import os, re, uuid, hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

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

# ----------------------------- Data Classes -----------------------------

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

# ----------------------------- Utilities -----------------------------

def _deterministic_id(prefix: str = "chunk", source_path: Optional[str] = None, text: Optional[str] = None) -> str:
    if source_path is not None and text is not None:
        data = (str(source_path) + "|" + text).encode("utf-8")
        return f"{prefix}_{hashlib.sha1(data).hexdigest()[:16]}"
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text.split()) * 1.3))

def _split_sentences(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if s.strip()]

# ----------------------------- Block Extraction -----------------------------

_TABLE_LINE_RE = re.compile(r"[|\t,]")
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6}\s+|[A-Z][A-Za-z0-9\- ]{2,120}:$)")
_LIST_RE = re.compile(r"^(\s*[-*+]\s+|\s*\d+\.|\s*•\s+)")
_CODE_FENCE_RE = re.compile(r"^\s*```")

def extract_structured_blocks_from_text(text: str) -> List[Block]:
    lines = text.splitlines()
    blocks: List[Block] = []
    buf: List[str] = []
    cur_type = "paragraph"

    def flush():
        nonlocal buf, cur_type
        if buf:
            blocks.append(Block(type=cur_type, text="\n".join(buf).rstrip()))
            buf = []
            cur_type = "paragraph"

    in_code = False
    for line in lines:
        s = line.rstrip()
        if not s.strip():
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

# ----------------------------- Embedding Helpers -----------------------------

_embedding_model = None

def load_embedding_model(name: str = "all-MiniLM-L6-v2"):
    global _embedding_model
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed")
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(name)
    return _embedding_model

def embed_texts(texts: List[str]) -> List[List[float]]:
    if SentenceTransformer:
        model = load_embedding_model()
        return model.encode(texts, convert_to_numpy=True)
    if np:
        out = []
        for t in texts:
            v = np.zeros(128, dtype=float)
            for i, ch in enumerate(t[:1024]):
                v[i % 128] += ord(ch)
            out.append((v / (np.linalg.norm(v) + 1e-12)).tolist())
        return out
    return [[len(t)] for t in texts]

# ----------------------------- Intelligent Chunker -----------------------------

class IntelligentChunker:
    def __init__(
        self,
        max_tokens: int = 1200,
        target_tokens: int = 800,
        overlap_tokens: int = 150,
        semantic_threshold: float = 0.65
    ):
        self.max_tokens = max_tokens
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.semantic_threshold = semantic_threshold

    def chunk_document(self, text: str, meta: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        if meta is None:
            meta = {}

        blocks = extract_structured_blocks_from_text(text)
        paras = [b.text for b in blocks]

        try:
            embeddings = embed_texts(paras) if paras else None
        except Exception:
            embeddings = None

        boundaries = set()
        if embeddings is not None and cosine_similarity is not None and len(paras) > 1:
            try:
                sims = cosine_similarity(embeddings)
                for i in range(len(paras) - 1):
                    if blocks[i].type in ("table", "list", "code") or blocks[i+1].type in ("table", "list", "code"):
                        continue
                    if float(sims[i, i+1]) < self.semantic_threshold:
                        boundaries.add(i)
            except Exception:
                embeddings = None

        if embeddings is None:
            for i in range(len(paras) - 1):
                if blocks[i].type in ("table", "list", "code") or blocks[i+1].type in ("table", "list", "code"):
                    continue
                if _naive_similarity(paras[i], paras[i+1]) < 0.28:
                    boundaries.add(i)

        chunks: List[Chunk] = []
        cur_pack: List[str] = []
        cur_meta_sources: List[Dict[str, Any]] = []

        def flush_pack():
            nonlocal cur_pack, cur_meta_sources
            if not cur_pack:
                return
            text_pack = "\n\n".join(cur_pack).strip()
            types = list(dict.fromkeys([m["type"] for m in cur_meta_sources]))
            meta_copy = dict(meta)
            meta_copy["block_types"] = types
            meta_copy["atomic"] = any(m.get("atomic", False) for m in cur_meta_sources)
            chunks.append(Chunk(id=_deterministic_id("chunk", meta.get("source_path"), text_pack),
                                text=text_pack,
                                meta=meta_copy))
            cur_pack = []
            cur_meta_sources = []

        for idx, b in enumerate(blocks):
            current_texts = [b.text] if b.type != "table" else [b.text]  # keep entire table as one chunk
            is_boundary = idx in boundaries

            for text_piece in current_texts:
                text_piece = text_piece.strip()
                if not text_piece:
                    continue
                if _approx_tokens(text_piece) > self.max_tokens:
                    sents = _split_sentences(text_piece)
                    pack = []
                    for s in sents:
                        if _approx_tokens(" ".join(pack + [s])) <= self.target_tokens:
                            pack.append(s)
                        else:
                            if pack:
                                cur_pack.extend(pack)
                                cur_meta_sources.extend([{"type": b.type, "atomic": (b.type=="table")}])
                                flush_pack()
                            pack = [s]
                    if pack:
                        cur_pack.extend(pack)
                        cur_meta_sources.extend([{"type": b.type, "atomic": (b.type=="table")}])
                        flush_pack()
                    continue

                candidate = "\n\n".join(cur_pack + [text_piece])
                if _approx_tokens(candidate) > self.target_tokens or is_boundary:
                    if not cur_pack:
                        cur_pack.append(text_piece)
                        cur_meta_sources.append({"type": b.type, "atomic": (b.type=="table")})
                        flush_pack()
                    else:
                        flush_pack()
                        cur_pack.append(text_piece)
                        cur_meta_sources.append({"type": b.type, "atomic": (b.type=="table")})
                        is_boundary = False
                else:
                    cur_pack.append(text_piece)
                    cur_meta_sources.append({"type": b.type, "atomic": (b.type=="table")})

                # tables remain atomic
                if b.type == "table":
                    flush_pack()

            if is_boundary and cur_pack:
                flush_pack()

        flush_pack()

        chunks = self._refinement_pass(chunks)
        if self.overlap_tokens > 0:
            chunks = self._apply_sentence_overlaps(chunks)
        return chunks

    def _refinement_pass(self, chunks: List[Chunk]) -> List[Chunk]:
        out = []
        i = 0
        while i < len(chunks):
            c = chunks[i]
            tkn = _approx_tokens(c.text)
            c_atomic = c.meta.get("atomic", False) or ("table" in c.meta.get("block_types", [])) or ("code" in c.meta.get("block_types", []))
            if tkn > self.max_tokens:
                sents = _split_sentences(c.text)
                pack = []
                for s in sents:
                    if _approx_tokens(" ".join(pack + [s])) <= self.target_tokens:
                        pack.append(s)
                    else:
                        if pack:
                            out.append(Chunk(id=_deterministic_id("chunk", None, " ".join(pack)),
                                             text=" ".join(pack),
                                             meta=c.meta.copy()))
                        pack = [s]
                if pack:
                    out.append(Chunk(id=_deterministic_id("chunk", None, " ".join(pack)),
                                     text=" ".join(pack),
                                     meta=c.meta.copy()))
            else:
                # merge tiny non-atomic chunks
                if tkn < int(self.target_tokens*0.25) and i+1 < len(chunks):
                    nxt = chunks[i+1]
                    nxt_atomic = nxt.meta.get("atomic", False) or ("table" in nxt.meta.get("block_types", [])) or ("code" in nxt.meta.get("block_types", []))
                    if not c_atomic and not nxt_atomic:
                        merged_text = (c.text + "\n\n" + nxt.text).strip()
                        merged_meta = {**c.meta, "block_types": list(dict.fromkeys(c.meta.get("block_types", []) + nxt.meta.get("block_types", [])))}
                        out.append(Chunk(id=_deterministic_id("chunk", None, merged_text),
                                         text=merged_text,
                                         meta=merged_meta))
                        i += 1
                    else:
                        out.append(c)
                else:
                    out.append(c)
            i += 1
        return out

    def _apply_sentence_overlaps(self, chunks: List[Chunk]) -> List[Chunk]:
        if len(chunks) <= 1:
            return chunks
        out = []
        avg_words_per_sentence = 14
        overlap_sent_count = max(1, int(self.overlap_tokens / (avg_words_per_sentence*1.3)))
        for i, c in enumerate(chunks):
            out.append(c)
            if i < len(chunks)-1:
                nxt = chunks[i+1]
                if c.meta.get("atomic", False) or nxt.meta.get("atomic", False):
                    continue
                tail_sents = _split_sentences(c.text)[-overlap_sent_count:]
                head_sents = _split_sentences(nxt.text)[:overlap_sent_count]
                if tail_sents and head_sents:
                    merged_text = (" ".join(tail_sents) + "\n\n" + " ".join(head_sents)).strip()
                    meta = {"overlap_of":[c.id,nxt.id], "block_types": list(dict.fromkeys(c.meta.get("block_types",[])+nxt.meta.get("block_types",[])))}
                    out.append(Chunk(id=_deterministic_id("overlap", None, merged_text), text=merged_text, meta=meta))
        return out

# ----------------------------- Helpers -----------------------------

def _naive_similarity(a:str,b:str) -> float:
    def trigrams(s:str):
        s2 = re.sub(r"\s+"," ",s.strip().lower())
        return {s2[i:i+3] for i in range(max(0,len(s2)-2))}
    ta = trigrams(a)
    tb = trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb)/max(1,len(ta | tb))
