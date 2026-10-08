import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, chunk_size=800, chunk_overlap=100):
        # Chunking based on tokens
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_document(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.md':
            loader = UnstructuredMarkdownLoader(file_path)
        elif ext == '.txt':
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        return loader.load()

    def process_directory(self, directory_path):
        documents = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    docs = self.load_document(file_path)
                    documents.extend(docs)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        chunks = self.text_splitter.split_documents(documents)
        return chunks

if __name__ == "__main__":
    processor = DocumentProcessor()
    # Example usage:
    # chunks = processor.process_directory("../../data/raw")
    # print(f"Processed into {len(chunks)} chunks.")
