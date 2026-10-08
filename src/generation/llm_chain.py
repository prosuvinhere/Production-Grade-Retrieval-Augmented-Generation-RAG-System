import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class RAGChain:
    def __init__(self, prompt_config_path=None, model_name=None, temperature=0.0):
        if prompt_config_path is None:
            # Resolve relative to project root or config directory
            current_dir = Path(__file__).resolve().parent
            prompt_config_path = current_dir.parent / "config" / "prompts.yaml"
            
        with open(prompt_config_path, 'r', encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        self.prompt_template = PromptTemplate.from_template(config['qa_prompt'])
        model = model_name or os.getenv("OPENAI_MODEL", "gpt-4o")
        self.llm = ChatOpenAI(model=model, temperature=temperature)

    def format_docs(self, docs):
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown Source')
            page = doc.metadata.get('page', 'Unknown Page')
            formatted.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}")
        return "\n\n".join(formatted)

    def get_chain(self, retriever_function):
        chain = (
            {"context": retriever_function | self.format_docs, "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
        return chain
        
    def invoke(self, retriever_function, query):
        """Invoke the RAG pipeline end-to-end."""
        chain = self.get_chain(retriever_function)
        return chain.invoke(query)
