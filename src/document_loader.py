import os
from typing import Any

from langchain_community.document_loaders import TextLoader, CSVLoader, JSONLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentLoader:
    def __init__(self, data_dir: str = f"{os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')}"):
        """Initialize the document loader.
        
        Args:
            (optional) data_dir: Directory containing documents to load. Default is the "data" directory that's listed with "src".
        """
        
        self.data_dir = data_dir

    def load_documents(self) -> list[dict[str, Any]]:
        """Load all documents from the data directory.
        
        Supports multiple file types:
        - .txt: Plain text files
        - .csv: CSV files
        - .json: JSON files
        - .pdf: PDF files
        
        Returns:
            A list of document dictionaries with 'content' and 'metadata'.
        """
        documents = []
        
        for filename in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, filename)
            file_extension = os.path.splitext(filename)[1].lower()
            
            print(f"Loading {filename}...")
            
            try:
                # Select appropriate loader based on file extension
                if file_extension == '.txt':
                    loader = TextLoader(file_path, encoding = 'utf-8')
                elif file_extension == '.csv':
                    loader = CSVLoader(file_path)
                elif file_extension == '.json':
                    loader = JSONLoader(
                        file_path = file_path,
                        jq_schema = '.',
                        text_content = False
                    )
                elif file_extension == '.pdf':
                    loader = PyPDFLoader(file_path)
                else:
                    print(f"Unsupported file extension: {file_extension} for {filename}")
                    continue
                
                # Load document with appropriate loader
                langchain_docs = loader.load()
                
                # Process loaded documents
                for doc in langchain_docs:
                    documents.append({
                        'content': doc.page_content,
                        'metadata': {
                            'source': filename,
                            'filename': filename,
                            'file_type': file_extension,
                            **doc.metadata
                        }
                    })
                    
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
        
        return documents
    
    def chunk_documents(self, documents: list[dict[str, Any]], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[dict[str, Any]]:
        """Chunk documents into smaller pieces using LangChain's text splitter.
        
        Args:
            documents: List of document dictionaries.
            (optional) chunk_size: Maximum size of each chunk. Default is 1000 characters.
            (optional) chunk_overlap: Overlap between chunks. Default is 200 characters.
            
        Returns:
            A list of chunk dictionaries with 'content' and 'metadata'.
        """
        langchain_docs = [
            Document(
                page_content = doc['content'],
                metadata = doc['metadata']
            ) for doc in documents
        ]
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            length_function = len,
            separators = ["\n\n", "\n", ". ", " ", ""]
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