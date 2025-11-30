# src/rag_pipeline.py

import os
from typing import List, Tuple, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

# Global caches
_EMBEDDER: Optional[SentenceTransformer] = None
_CHUNKS: Optional[List[str]] = None
_CHUNK_EMBEDDINGS: Optional[np.ndarray] = None


def _get_repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _get_knowledge_base_path() -> str:
    return os.path.join(_get_repo_root(), "data", "knowledge_base.md")


def _load_knowledge_base_text() -> str:
    kb_path = _get_knowledge_base_path()
    with open(kb_path, "r", encoding="utf-8") as f:
        return f.read()


def _chunk_text(text: str, min_length: int = 100) -> List[str]:
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    buffer = ""
    for para in raw_paragraphs:
        if not buffer:
            buffer = para
        else:
            if len(buffer) < min_length:
                buffer = buffer + "\n\n" + para
            else:
                chunks.append(buffer)
                buffer = para
    if buffer:
        chunks.append(buffer)
    return chunks


def _get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMBEDDER


def build_vector_store() -> Tuple[List[str], np.ndarray]:
    global _CHUNKS, _CHUNK_EMBEDDINGS

    text = _load_knowledge_base_text()
    chunks = _chunk_text(text)
    embedder = _get_embedder()
    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)

    _CHUNKS = chunks
    _CHUNK_EMBEDDINGS = embeddings
    return _CHUNKS, _CHUNK_EMBEDDINGS


def _ensure_vector_store_built():
    if _CHUNKS is None or _CHUNK_EMBEDDINGS is None:
        build_vector_store()


def retrieve_relevant_chunks(query: str, top_k: int = 5) -> List[str]:
    _ensure_vector_store_built()
    embedder = _get_embedder()

    query_vec = embedder.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
    sims = np.dot(_CHUNK_EMBEDDINGS, query_vec) / (np.linalg.norm(_CHUNK_EMBEDDINGS, axis=1) * np.linalg.norm(query_vec) + 1e-10)
    top_k = max(1, min(top_k, len(_CHUNKS)))
    top_indices = np.argsort(sims)[-top_k:][::-1]

    return [_CHUNKS[i] for i in top_indices]
