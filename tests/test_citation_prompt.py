import os
import yaml
from pathlib import Path
from langchain_core.documents import Document
from src.generation.llm_chain import RAGChain

def test_prompts_yaml_structure():
    prompt_path = Path("src/config/prompts.yaml")
    assert prompt_path.exists(), "prompts.yaml must exist"
    
    with open(prompt_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    assert "qa_prompt" in config
    assert "{context}" in config["qa_prompt"]
    assert "{question}" in config["qa_prompt"]
    assert "citations" in config["qa_prompt"].lower()

def test_format_docs_with_citations():
    # Pass mock OPENAI_API_KEY if not in environment for testing
    os.environ.setdefault("OPENAI_API_KEY", "test-key-mock")
    chain = RAGChain()
    
    docs = [
        Document(page_content="Content paragraph 1", metadata={"source": "manual.pdf", "page": 3}),
        Document(page_content="Content paragraph 2", metadata={"source": "faq.md", "page": 1}),
    ]
    
    formatted = chain.format_docs(docs)
    assert "[Source: manual.pdf, Page: 3]" in formatted
    assert "[Source: faq.md, Page: 1]" in formatted
    assert "Content paragraph 1" in formatted
