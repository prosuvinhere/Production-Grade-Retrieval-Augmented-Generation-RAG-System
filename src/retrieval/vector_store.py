import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

class VectorStore:
    def __init__(self, persist_directory=None):
        self.persist_directory = persist_directory or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.embeddings = OpenAIEmbeddings(model=embedding_model) 
        self.db = None

    def create_or_load(self):
        self.db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        return self.db

    def add_documents(self, chunks):
        if not self.db:
            self.create_or_load()
        self.db.add_documents(chunks)
        
    def get_retriever(self, k=5):
        if not self.db:
            self.create_or_load()
        return self.db.as_retriever(search_kwargs={"k": k})
