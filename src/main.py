import argparse
from typing import Dict, Any
from pprint import pprint

from document_loader import DocumentLoader
from embeddings import VectorStore
from retrieval import Retriever
from llm import LLMService
from agent import Agent

def setup_system(rebuild_index: bool = False) -> Dict[str, Any]:
    """Set up the RAG agent system.
    
    Args:
        rebuild_index: Whether to rebuild the vector index
        
    Returns:
        Dictionary with system components
    """
    print("Setting up RAG agent system...")
    
    loader = DocumentLoader()
    
    vector_store = VectorStore()
    
    index_exists = vector_store.load()
    
    if rebuild_index or not index_exists:
        print("Building document index...")
        documents = loader.load_documents()
        chunks = loader.chunk_documents(documents)
        vector_store.add_chunks(chunks)
        vector_store.save()
    
    retriever = Retriever(vector_store)
    
    llm_service = LLMService()
    
    agent = Agent(retriever, llm_service)
    
    return {
        "loader": loader,
        "vector_store": vector_store,
        "retriever": retriever,
        "llm_service": llm_service,
        "agent": agent
    }

def cli_interface():
    """Run a simple CLI interface for the RAG agent."""
    parser = argparse.ArgumentParser(description="RAG Agent System")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild document index")
    args = parser.parse_args()
    
    system = setup_system(rebuild_index=args.rebuild)
    agent = system["agent"]
    
    print("\nRAG Agent System ready. Type 'exit' to quit.")
    print("Example queries:")
    print("  - What is your flagship product?")
    print("  - Calculate 25 * 16 - 42")
    print("  - Define artificial intelligence")
    print()
    
    while True:
        query = input("Hi! How can I help you?\n")
        
        if query.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        
        response = agent.process_query(query)
        
        print("\n" + "="*50)
        print(f"QUERY: {response['query']}")
        print(f"TOOL: {response['tool_used']}")
        print("-"*50)
        
        if response['tool_used'] == 'rag':
            print("RETRIEVED CHUNKS:")
            for i, chunk in enumerate(response['retrieved_chunks']):
                print(f"\nChunk {i+1} (Source: {chunk['source']}, Score: {chunk['relevance_score']:.2f}):")
                print(f"{chunk['content'][:200]}...")
            print("-"*50)
            
        print("LOGS:")
        for log in response['log']:
            print(f"- {log}")
        print("-"*50)
        
        if response['tool_used'] == 'rag':
            print("RESULT:")
            print(response['result'].content)
            print("\nUSAGE INFO:")
            pprint(response['result'].usage_metadata)
            print("\nMETADATA:")
            pprint(response['result'].response_metadata)
            print("\nID:")
            print(response['result'].id)
            print("="*50 + "\n")
        else:
            print("RESULT:")
            print(response['result'])
            print("="*50 + "\n")

def main():
    """Main entry point for the application."""
    cli_interface()

if __name__ == "__main__":
    main()