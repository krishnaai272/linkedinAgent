# tests/test_rag_pipeline.py
from src.rag_pipeline import build_vector_store, retrieve_relevant_chunks


def test_build_vector_store_and_retrieve():
    chunks, embeddings = build_vector_store()
    assert len(chunks) > 0, "Vector store should have at least one chunk."
    assert embeddings.shape[0] == len(chunks), "Embeddings count should match chunks."

    results = retrieve_relevant_chunks("What is RAG?", top_k=3)
    assert isinstance(results, list), "retrieve_relevant_chunks should return a list."
    assert len(results) > 0, "retrieve_relevant_chunks should return non-empty results."
    assert all(isinstance(c, str) for c in results), "All returned chunks should be strings."
