from typing import Any
from retrieval import Retriever
from llm import LLMService

class Agent:
    def __init__(self, retriever: Retriever, llm_service: LLMService):
        """Initialize the agent with necessary components.
        
        Args:
            retriever: Retriever instance for retrieving relevant chunks.
            llm_service: LLMService instance for generating responses.
        """
        self.retriever = retriever
        self.llm_service = llm_service
    
    def process_query(self, query: str) -> dict[str, Any]:
        """Process a user query and return a response.
        
        Args:
            query: User query string.
            
        Returns:
            Dictionary with response and process information.
        """
        
        response = {
            "query": query,
            "tool_used": None,
            "log": [],
            "reason": None
        }
        
        response["log"].append("Retrieving relevant chunks...")
        chunks = self.retriever.retrieve(query)
        
        response["retrieved_chunks"] = [
            {
                "content": chunk["content"],
                "source": chunk["metadata"]["source"],
                "relevance_score": chunk["relevance_score"]
            }
            for chunk in chunks if chunk["relevance_score"] > 0.4
        ]
        
        if response["retrieved_chunks"] == []:
            response["log"].append("No relevant chunks found.")
            response["log"].append("Agent detected tool: none")
            response["tool_used"] = "none"
        else:
            response["log"].append(f"Retrieved {len(chunks)} chunks")
            response["log"].append("Agent detected tool: rag")
            response["tool_used"] = "rag"
        
        response["log"].append("Generating response with LLM...")
        llm_response, response["reason"] = self.llm_service.generate_response(query, (None if response["retrieved_chunks"] == [] else response["retrieved_chunks"]))
        
        if(llm_response == "Invalid model."):
            response["result"] = "Invalid model."
            response["log"].append("Invalid model.")
        elif(llm_response == "Error while accessing LLM service. Please ensure the Ollama server is running by running 'ollama ps'.\n(Maybe the model is listening on a different port?)"):
            response["result"] = llm_response
            response["log"].append("Error occurred while accessing LLM service.")
        else:
            response["result"] = llm_response
            response["log"].append("LLM response generated")
            
        return response