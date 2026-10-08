import os
import tempfile
import pytest
from src.ingestion.document_processor import DocumentProcessor

def test_document_processor_initialization():
    processor = DocumentProcessor(chunk_size=600, chunk_overlap=100)
    assert processor.text_splitter is not None

def test_markdown_processing():
    processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)
    sample_content = (
        "# Architectural Guidelines\n\n"
        + "Production RAG systems require continuous monitoring, evaluation, and strict latency SLAs. " * 20
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        md_path = os.path.join(tmpdir, "test_doc.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(sample_content)
            
        chunks = processor.process_directory(tmpdir)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.page_content
            assert len(chunk.page_content) > 0
