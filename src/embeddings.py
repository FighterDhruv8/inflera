import os
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions

class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", persist_directory: str = "data/chroma_db"):
        """Initialize the vector store with the specified embedding model.
        
        Args:
            model_name: Name of the sentence transformer model to use
            persist_directory: Directory to persist the vector database
        """
        #self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self._get_collection()
        
    def _get_collection(self, collection_name: str = "document_chunks"):
        """Get a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            The ChromaDB collection
        """
        try:
            return self.client.get_or_create_collection(
                name=collection_name,
                #embedding_function=self.embedding_function
            )
        except Exception as e:
            print(f"Error while fetching data to be stored in database:\n{e}")
    
    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Add document chunks to the vector store.
        
        Args:
            chunks: List of chunk dictionaries with 'content' and 'metadata'
        """
        if not chunks:
            print("No chunks provided to add to the vector store")
            return
        
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        documents = [chunk['content'] for chunk in chunks]
        metadatas = [chunk.get('metadata', {}) for chunk in chunks]
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Added {len(chunks)} chunks to the vector store")
    
    def search(self, query: str, n_results: int = 3):
        """Search for relevant chunks.
        
        Args:
            query: The query string
            n_results: Number of results to return
            
        Returns:
            List of results with content and metadata
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        formatted_results = []
        if results and results['documents'] and results['metadatas']:
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results.get('distances', [[]])[0]
            
            for i, (doc, meta) in enumerate(zip(documents, metadatas)):
                result = {
                    'content': doc,
                    'metadata': meta
                }
                if distances:
                    result['distance'] = distances[i]
                formatted_results.append(result)
                
        return formatted_results
    
    def save(self):
        """Save the vector index to disk.
        
        Note: ChromaDB automatically persists data so this is mostly a no-op
        """
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)
            
        print(f"Vector store is already persisted at {self.persist_directory}")
    
    def load(self):
        """Load the vector index from disk.
        
        Note: ChromaDB automatically loads data so this just verifies the collection exists
        
        Returns:
            True if the collection exists, False otherwise
        """
        try:
            count = self.collection.count()
            if count > 0:
                print(f"Loaded vector store with {count} chunks")
                return True
            else:
                print("Vector store exists but contains no chunks")
                return False
        except Exception as e:
            print(f"Failed to load vector store: {e}")