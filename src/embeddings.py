from typing import Any
import chromadb

class VectorStore:
    def __init__(self):
        """Initialize the vector store with the specified embedding model."""
        self.client = chromadb.EphemeralClient()
        self.collection = self._get_collection()
    
    def _get_collection(self, collection_name: str = "document_chunks") -> chromadb.Collection:
        """Get a ChromaDB collection.
        
        Args:
            (optional) collection_name: Name of the collection. Default is "document_chunks".
        Returns:
            The ChromaDB collection.
        """
        try:
            return self.client.get_or_create_collection(name = collection_name)
        except Exception as e:
            print(f"Error while accessing database:\n{e}")
    
    def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Add document chunks to the vector store.
        
        Args:
            chunks: List of chunk dictionaries with 'content' and 'metadata'.
            
        Returns:
            None.
        """
        if not chunks:
            print("No chunks provided to add to the vector store")
            return
        
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        documents = [chunk['content'] for chunk in chunks]
        metadatas = [chunk.get('metadata', {}) for chunk in chunks]
        
        self.collection.add(
            documents = documents,
            metadatas = metadatas,
            ids = ids
        )
    
    def search(self, query: str, n_results: int = 3) -> list[Any]:
        """Search for relevant chunks.
        
        Args:
            query: The query string.
            (optional) n_results: Number of top relevant results to return. Default is 3.
            
        Returns:
            List of results with content and metadata.
        """
        results = self.collection.query(
            query_texts = [query],
            n_results = n_results
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