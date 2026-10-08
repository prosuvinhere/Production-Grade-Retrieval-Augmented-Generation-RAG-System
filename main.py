#!/usr/bin/env python3
"""
Main CLI Entrypoint for Production-Grade RAG System.

Commands:
    ingest    Ingest documents into ChromaDB vector store
    query     Ask a question with citation enforcement and optional hybrid search
    evaluate  Run Ragas offline evaluation against golden dataset
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.ingestion.document_processor import DocumentProcessor
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_search import HybridRetriever
from src.generation.llm_chain import RAGChain
from src.evaluation.evaluate import run_evaluation


def handle_ingest(args):
    print(f"\n📂 Starting document ingestion from: {args.dir}")
    processor = DocumentProcessor(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunks = processor.process_directory(args.dir)
    
    if not chunks:
        print(f"⚠️ No documents or chunks found in {args.dir}.")
        return

    print(f"✂️ Processed {len(chunks)} chunks with chunk size {args.chunk_size} and overlap {args.chunk_overlap}.")
    
    vs = VectorStore(persist_directory=args.persist_dir)
    print(f"💾 Writing chunks to ChromaDB at: {args.persist_dir}...")
    vs.add_documents(chunks)
    print("✅ Ingestion complete! Chunks successfully stored in vector database.\n")


def handle_query(args):
    query = args.query
    print(f"\n🔎 Question: {query}\n" + "-" * 50)
    
    vs = VectorStore(persist_directory=args.persist_dir)
    vs.create_or_load()
    chain = RAGChain()

    if args.hybrid:
        print("⚡ Executing Hybrid Search (BM25 + ChromaDB) with Cohere Reranking...")
        cohere_key = os.getenv("COHERE_API_KEY")
        if not cohere_key:
            print("❌ Error: COHERE_API_KEY environment variable is required for hybrid search.")
            sys.exit(1)
            
        hybrid_retriever = HybridRetriever(vector_store=vs, cohere_api_key=cohere_key)
        
        # In practice, fit on raw chunks or load from store
        processor = DocumentProcessor()
        chunks = processor.process_directory(args.doc_dir) if os.path.exists(args.doc_dir) else []
        if chunks:
            hybrid_retriever.fit_bm25(chunks)
            
        retrieved_docs = hybrid_retriever.retrieve(query, top_k=args.top_k, rerank_top_k=args.rerank_top_k)
        retriever_func = lambda q: retrieved_docs
    else:
        print(f"⚡ Executing Dense Semantic Retrieval (top_k={args.top_k})...")
        base_retriever = vs.get_retriever(k=args.top_k)
        retriever_func = lambda q: base_retriever.invoke(q)

    # Generate response
    response = chain.invoke(retriever_func, query)
    print("\n💡 Generated Answer:")
    print(response)
    print("-" * 50 + "\n")


def handle_evaluate(args):
    print(f"\n🧪 Running Ragas Evaluation with threshold: {args.threshold}...")
    run_evaluation(threshold=args.threshold)


def main():
    parser = argparse.ArgumentParser(
        description="Production-Grade Retrieval-Augmented Generation (RAG) CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available subcommands")

    # Ingest subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into vector database")
    ingest_parser.add_argument("--dir", default="data/sample_docs", help="Directory containing source documents")
    ingest_parser.add_argument("--persist-dir", default=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"), help="ChromaDB persistence path")
    ingest_parser.add_argument("--chunk-size", type=int, default=800, help="Maximum token chunk size (500-800)")
    ingest_parser.add_argument("--chunk-overlap", type=int, default=100, help="Token overlap size (default: 100)")
    ingest_parser.set_defaults(func=handle_ingest)

    # Query subcommand
    query_parser = subparsers.add_parser("query", help="Query the RAG pipeline")
    query_parser.add_argument("query", type=str, help="User question / prompt")
    query_parser.add_argument("--hybrid", action="store_true", help="Enable Hybrid Search + Cohere Cross-Encoder Reranking")
    query_parser.add_argument("--top-k", type=int, default=10, help="Candidate chunks to retrieve")
    query_parser.add_argument("--rerank-top-k", type=int, default=5, help="Top chunks to keep after re-ranking")
    query_parser.add_argument("--doc-dir", default="data/sample_docs", help="Directory for BM25 corpus indexing")
    query_parser.add_argument("--persist-dir", default=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"), help="ChromaDB persistence path")
    query_parser.set_defaults(func=handle_query)

    # Evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Run offline Ragas evaluation against golden dataset")
    eval_parser.add_argument("--threshold", type=float, default=float(os.getenv("RAGAS_FAITHFULNESS_THRESHOLD", 0.80)), help="Minimum faithfulness threshold")
    eval_parser.set_defaults(func=handle_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
