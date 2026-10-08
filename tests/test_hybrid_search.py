from unittest.mock import MagicMock
from langchain_core.documents import Document
from src.retrieval.hybrid_search import HybridRetriever

def test_bm25_fit_and_retrieve():
    mock_vector_store = MagicMock()
    mock_vector_store.get_retriever.return_value.invoke.return_value = [
        Document(page_content="Semantic result 1", metadata={"source": "doc1"}),
    ]

    retriever = HybridRetriever(vector_store=mock_vector_store, cohere_api_key="mock_cohere_key")
    
    # Mock the cohere client rerank response
    mock_rerank_result = MagicMock()
    mock_item = MagicMock()
    mock_item.index = 0
    mock_item.relevance_score = 0.95
    mock_rerank_result.results = [mock_item]
    retriever.co_client.rerank = MagicMock(return_value=mock_rerank_result)

    docs = [
        Document(page_content="PostgreSQL database indexing techniques", metadata={"source": "spec.md"}),
        Document(page_content="Vector embeddings with ChromaDB and HNSW graphs", metadata={"source": "spec.md"}),
    ]
    retriever.fit_bm25(docs)

    results = retriever.retrieve(query="PostgreSQL indexing", top_k=2, rerank_top_k=1)
    assert len(results) == 1
    assert "rerank_score" in results[0].metadata
    assert results[0].metadata["rerank_score"] == 0.95
