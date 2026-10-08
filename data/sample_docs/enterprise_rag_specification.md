# Enterprise RAG System Specification & Architecture Guide

## 1. Document Ingestion and Chunking Strategy
The document ingestion layer supports heterogeneous formats including PDF, Markdown (.md), and plain text (.txt). Documents are parsed and split using token-aware chunking algorithms (`RecursiveCharacterTextSplitter` with `cl100k_base` tiktoken encoding). 

Each chunk is bounded between 500 and 800 tokens with a mandatory 100-token sliding overlap. This overlap guarantees contextual continuity across split boundaries, preventing truncated definitions and fragmented entities.

## 2. Storage and Vector Indexing
Extracted chunks are embedded into continuous dense vectors using OpenAI `text-embedding-3-small` (1536 dimensions) and indexed in ChromaDB. Every vector payload retains rich metadata attributes including original document source, file path, chunk index, page number, and paragraph hashes.

## 3. Hybrid Search and Cross-Encoder Re-Ranking
Production retrieval combines lexical search and semantic search:
1. Dense Semantic Search: Vector similarity in ChromaDB captures semantic intent and paraphrased queries.
2. Sparse Lexical Search: BM25 (Okapi) indexes exact domain terminology, code identifiers, and acronyms.
3. Candidate Union: The top-k candidates from both streams are merged and deduplicated.
4. Cross-Encoder Re-Ranking: Candidates are scored using the Cohere `rerank-english-v3.0` model. Cross-encoders attend across both query and document tokens simultaneously, achieving significantly higher precision than bi-encoder cosine similarities alone.

## 4. Strict Citation Enforcement & Anti-Hallucination
To satisfy regulatory and enterprise auditing constraints, the LLM prompt contains deterministic citation enforcement:
- Every affirmative answer must provide strict citations referencing the exact source document and page number.
- If the retrieved context is insufficient or ungrounded, the system is strictly instructed to respond: "I cannot answer this question based on the provided context." Guessing, extrapolation, or leveraging ungrounded prior knowledge is prohibited.

## 5. Automated CI/CD Evaluation with Ragas
The deployment pipeline incorporates continuous evaluation against a curated golden dataset of verified question-answer pairs. Ragas computes the 'faithfulness' metric to mathematically verify whether claims in the generated response are grounded in the retrieved context. A pull request is automatically rejected if the mean faithfulness score falls below the 0.80 threshold.
