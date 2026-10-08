import os
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
import numpy as np
from langchain_core.documents import Document
import cohere

load_dotenv()

class HybridRetriever:
    def __init__(self, vector_store, cohere_api_key=None):
        self.vector_store = vector_store
        api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError(
                "Cohere API key not found. Please provide cohere_api_key or set COHERE_API_KEY in your .env file."
            )
        self.co_client = cohere.Client(api_key)
        self.documents = []
        self.bm25 = None
        self.doc_texts = []

    def fit_bm25(self, all_documents: list[Document]):
        """Fit BM25 on the corpus."""
        self.documents = all_documents
        self.doc_texts = [doc.page_content for doc in all_documents]
        tokenized_corpus = [doc.split(" ") for doc in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 10, rerank_top_k: int = 5):
        """Perform Hybrid Search and Cross-Encoder Re-Ranking."""
        # 1. Vector Search
        vector_retriever = self.vector_store.get_retriever(k=top_k)
        vector_results = vector_retriever.invoke(query)
        
        # 2. BM25 Search (Keyword Search)
        if self.bm25:
            tokenized_query = query.split(" ")
            bm25_scores = self.bm25.get_scores(tokenized_query)
            top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
            bm25_results = [self.documents[i] for i in top_bm25_indices]
        else:
            bm25_results = []

        # Combine and deduplicate
        combined_results = {}
        for doc in vector_results + bm25_results:
            if doc.page_content not in combined_results:
                combined_results[doc.page_content] = doc
                
        unique_docs = list(combined_results.values())
        
        if not unique_docs:
            return []

        # 3. Cross-Encoder Rerank (Cohere)
        texts_to_rerank = [doc.page_content for doc in unique_docs]
        rerank_response = self.co_client.rerank(
            model='rerank-english-v3.0',
            query=query,
            documents=texts_to_rerank,
            top_n=rerank_top_k
        )
        
        reranked_docs = []
        for result in rerank_response.results:
            idx = result.index
            original_doc = unique_docs[idx]
            original_doc.metadata['rerank_score'] = result.relevance_score
            reranked_docs.append(original_doc)
            
        return reranked_docs
