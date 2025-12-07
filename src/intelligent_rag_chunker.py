"""
intelligent_rag_chunker_fixed.py

Corrected production-grade intelligent chunker for RAG systems (ChatGPT-style).

This file fixes metadata aggregation, overlap flattening and prevents
semantic-boundary splits across atomic blocks (tables/lists/code). It
uses sentence-level overlaps to preserve structure and provides cleaner
chunk metadata (block_types list).

Author: Regenerated for Kaiser Konok — fixes applied
"""

from __future__ import annotations

import os
import re
import json
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Optional dependencies
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


# ----------------------------- Utilities -----------------------------

def _gen_id(prefix: str = "chunk") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _approx_tokens(text: str) -> int:
    words = len(text.split())
    return int(words * 1.3)


def _split_sentences(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if s.strip()]


# ----------------------------- Data classes -----------------------------

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

_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6}\s+|[A-Z][A-Za-z0-9\- ]{2,120}:$)")
_LIST_RE = re.compile(r"^(\s*[-*+]\s+|\s*\d+\.|\s*•\s+)")
_CODE_FENCE_RE = re.compile(r"^\s*```")
_TABLE_LINE_RE = re.compile(r"\|")


def extract_structured_blocks_from_text(text: str) -> List[Block]:
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

        if _TABLE_LINE_RE.search(s) and ("|" in s):
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


# ----------------------------- Chunker -----------------------------

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

    def chunk_document(self, text: str, meta: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        if meta is None:
            meta = {}
        blocks = extract_structured_blocks_from_text(text)
        atomic_blocks = blocks.copy()

        paras = [b.text for b in atomic_blocks]

        try:
            embeddings = embed_texts(paras)
        except Exception:
            embeddings = None

        boundaries = set()
        if embeddings is not None and cosine_similarity is not None:
            try:
                sims = cosine_similarity(embeddings)
                for i in range(len(paras) - 1):
                    if atomic_blocks[i].type in ("table", "list", "code") or atomic_blocks[i + 1].type in ("table",
                                                                                                           "list",
                                                                                                           "code"):
                        continue
                    if float(sims[i, i + 1]) < self.semantic_threshold:
                        boundaries.add(i)
            except Exception:
                embeddings = None

        if embeddings is None:
            for i in range(len(paras) - 1):
                if atomic_blocks[i].type in ("table", "list", "code") or atomic_blocks[i + 1].type in ("table", "list",
                                                                                                       "code"):
                    continue
                if _naive_similarity(paras[i], paras[i + 1]) < 0.35:
                    boundaries.add(i)

        chunks: List[Chunk] = []
        cur_pack: List[str] = []
        cur_meta_sources: List[str] = []

        def flush_pack():
            nonlocal cur_pack, cur_meta_sources
            if not cur_pack:
                return
            text_pack = "\n\n".join(cur_pack).strip()
            cid = _gen_id()
            types = list(dict.fromkeys(cur_meta_sources))
            chunks.append(Chunk(id=cid, text=text_pack, meta={**meta, "block_types": types}))
            cur_pack = []
            cur_meta_sources = []

        for idx, b in enumerate(atomic_blocks):
            b_tokens = _approx_tokens(b.text)

            if b_tokens > self.max_tokens:
                sents = _split_sentences(b.text)
                pack: List[str] = []
                for s in sents:
                    if _approx_tokens(" ".join(pack + [s])) <= self.target_tokens:
                        pack.append(s)
                    else:
                        if pack:
                            cur_pack.extend(pack)
                            cur_meta_sources.extend([b.type])
                            flush_pack()
                        pack = [s]
                if pack:
                    cur_pack.extend(pack)
                    cur_meta_sources.extend([b.type])
                    flush_pack()
                continue

            cur_text = "\n\n".join(cur_pack + [b.text]).strip()
            if _approx_tokens(cur_text) > self.target_tokens or idx in boundaries:
                if not cur_pack:
                    cur_pack.append(b.text)
                    cur_meta_sources.append(b.type)
                    flush_pack()
                else:
                    flush_pack()
                    cur_pack.append(b.text)
                    cur_meta_sources.append(b.type)
            else:
                cur_pack.append(b.text)
                cur_meta_sources.append(b.type)

        flush_pack()

        chunks = self._refinement_pass(chunks)

        if self.overlap_tokens > 0:
            chunks = self._apply_sentence_overlaps(chunks)

        return chunks

    def _refinement_pass(self, chunks: List[Chunk]) -> List[Chunk]:
        out: List[Chunk] = []
        i = 0
        while i < len(chunks):
            c = chunks[i]
            tkn = _approx_tokens(c.text)
            if tkn > self.max_tokens:
                sents = _split_sentences(c.text)
                pack: List[str] = []
                for s in sents:
                    if _approx_tokens(" ".join(pack + [s])) <= self.target_tokens:
                        pack.append(s)
                    else:
                        if pack:
                            out.append(Chunk(id=_gen_id(), text=" ".join(pack), meta=c.meta.copy()))
                        pack = [s]
                if pack:
                    out.append(Chunk(id=_gen_id(), text=" ".join(pack), meta=c.meta.copy()))
            else:
                if tkn < int(self.target_tokens * 0.25) and i + 1 < len(chunks):
                    nxt = chunks[i + 1]
                    merged_text = (c.text + "\n\n" + nxt.text).strip()
                    merged_meta = {**c.meta}
                    merged_meta_types = list(
                        dict.fromkeys(c.meta.get("block_types", []) + nxt.meta.get("block_types", [])))
                    merged_meta["block_types"] = merged_meta_types
                    out.append(Chunk(id=_gen_id(), text=merged_text, meta=merged_meta))
                    i += 1
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
                tail_sents = _split_sentences(c.text)[-overlap_sent_count:]
                head_sents = _split_sentences(next_c.text)[:overlap_sent_count]
                if tail_sents and head_sents:
                    merged_text = (" ".join(tail_sents) + "\n\n" + " ".join(head_sents)).strip()
                    meta = {"overlap_of": [c.id, next_c.id], "block_types": list(
                        dict.fromkeys(c.meta.get("block_types", []) + next_c.meta.get("block_types", [])))}
                    out.append(Chunk(id=_gen_id("overlap"), text=merged_text, meta=meta))
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
    print(f'File Extension: ext')
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


# ----------------------------- Module usage example -----------------------------
if __name__ == "__main__":
    sample = """
        That's a great request! To help you test your intelligent RAG chunker, I'll generate a few distinct types of data samples. This will allow you to check its performance on different content structures, including plain text, structured lists, and technical material.
        
        ---
        
        ## 1. Simple Narrative Text (Low Structure) 📝
        
        This is a **long, flowing paragraph** where sentences are interconnected. A good chunker should recognize sentence boundaries and potentially combine related sentences into semantic units, rather than cutting mid-idea.
        
        > The discovery of penicillin by **Alexander Fleming** in 1928 marked a watershed moment in medical history. He famously observed a mold, *Penicillium notatum*, contaminating a Petri dish and noticed it inhibited the growth of *Staphylococcus* bacteria. This accidental finding laid the groundwork for the age of antibiotics, fundamentally changing how bacterial infections were treated. Before this, simple scratches or routine surgeries could often lead to deadly infections, a harsh reality of early 20th-century medicine. It took more than a decade for **Howard Florey and Ernst Chain** to further develop penicillin into a mass-produced, therapeutic drug, which became critically important during World War II, saving countless lives on the battlefield and in civilian hospitals. The subsequent rise of antibiotic resistance, however, underscores the ongoing challenge of maintaining the efficacy of these miracle drugs.
        
        ---
        
        ## 2. Structured, Multi-Topic Content (Medium Structure) 📑
        
        This section mixes a **general introduction** with **bulleted lists and defined sections**. A good chunker should be able to keep the heading and its immediate content together, and ensure bullet points are grouped into a single, cohesive chunk.
        
        ### Quantum Computing Fundamentals
        
        Quantum computing is a rapidly evolving field that leverages the principles of **quantum mechanics**—such as superposition and entanglement—to perform calculations. Unlike classical bits, which store information as 0 or 1, quantum bits, or **qubits**, can exist in a combination of both states simultaneously. This capability allows quantum computers to potentially solve certain complex problems far faster than any classical machine.
        
        * **Superposition:** The ability of a qubit to be in multiple states at once.
        * **Entanglement:** A phenomenon where two or more qubits become linked, such that they share the same fate regardless of the physical distance separating them.
        * **Decoherence:** The loss of quantum state coherence, a major technical hurdle in building stable quantum computers.
        
        **Primary Algorithms:**
        The most well-known algorithms designed for quantum computers are:
        1.  **Shor's Algorithm:** Can factor large numbers exponentially faster than classical algorithms, posing a threat to current public-key cryptography.
        2.  **Grover's Algorithm:** Offers a quadratic speedup for unstructured search problems.
        
        ---
        
        ## 3. Technical Data with Formulas (High Structure/Scientific) 🔬
        
        This content is **highly specific, includes an equation**, and requires the chunker to keep the formula close to its descriptive text to maintain context.
        
        ### The Black-Scholes Model
        
        The **Black-Scholes model** is a seminal mathematical model used for pricing European-style options. It assumes the stock price follows a geometric Brownian motion and that trading is continuous. The model's elegant formulation allows analysts to determine a theoretical price (or premium) for options contracts. A key component of the model is the calculation of the call option price, $C$.
        
        The value of a European call option is given by the formula:
        
        $$C(S_t, t) = N(d_1) S_t - N(d_2) K e^{-r(T-t)}$$
        
        Where:
        * $C$: The price of the call option.
        * $S_t$: The current stock price.
        * $K$: The strike price of the option.
        * $r$: The risk-free interest rate.
        * $T-t$: Time to maturity.
        * $N(\cdot)$: The cumulative standard normal distribution function.
        
        A sophisticated RAG chunker should treat the surrounding explanatory text and the formula as a **single, indispensable chunk** for accurate retrieval.
        
        ---
        
        ## Next Step
        
        These samples cover different structural complexities. Do you need **more examples** focusing on a specific type of data, such as code snippets, legal text, or tables, or would you like to define a specific **maximum chunk size** for me to test against?

    """

    ch = IntelligentChunker()
    res = ch.chunk_document(sample)
    print(json.dumps([{'id': c.id, 'text_preview': c.text, 'meta': c.meta} for c in res], indent=2))
