from typing import List, Dict, Any

class Retriever:
    def __init__(self, vector_store):
        """Initialize the retriever with a vector store.
        
        Args:
            vector_store: VectorStore instance
        """
        self.vector_store = vector_store
        
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve the top_k most relevant chunks for the query.
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve
            
        Returns:
            List of the top_k most relevant chunks with their metadata
        """
        if self.vector_store.collection is None or self.vector_store.collection.count() == 0:
            raise ValueError("Vector store is empty. Please add documents first.")
        
        results = self.vector_store.search(query, top_k)
        
        retrieved_chunks = []
        for result in results:
            retrieved_chunks.append({
                'content': result['content'],
                'metadata': result['metadata'],
                'relevance_score': float(1.0 / (1.0 + result.get('distance', 0)))
            })
        
        return retrieved_chunks