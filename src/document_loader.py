import os
from typing import List, Dict, Any

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentLoader:
    def __init__(self, data_dir: str = "data"):
        """Initialize the document loader.
        
        Args:
            data_dir: Directory containing documents to load
        """
        self.data_dir = data_dir
        
    def load_documents(self) -> List[Dict[str, Any]]:
        """Load all documents from the data directory using LangChain.
        
        Returns:
            A list of document dictionaries with 'content' and 'metadata'
        """
        documents = []
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(self.data_dir, filename)
                
                try:
                    loader = TextLoader(file_path, encoding='utf-8')
                    langchain_docs = loader.load()
                    
                    for doc in langchain_docs:
                        documents.append({
                            'content': doc.page_content,
                            'metadata': {
                                'source': filename,
                                'filename': filename,
                                **doc.metadata
                            }
                        })
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
                
        return documents
    
    def chunk_documents(self, documents: List[Dict[str, Any]], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """Chunk documents into smaller pieces using LangChain's text splitter.
        
        Args:
            documents: List of document dictionaries
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            A list of chunk dictionaries with 'content' and 'metadata'
        """
        langchain_docs = [
            Document(
                page_content=doc['content'],
                metadata=doc['metadata']
            ) for doc in documents
        ]
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        split_docs = text_splitter.split_documents(langchain_docs)
        
        chunks = []
        for i, doc in enumerate(split_docs):
            chunks.append({
                'content': doc.page_content,
                'metadata': {
                    **doc.metadata,
                    'chunk_id': i
                }
            })
        
        return chunks