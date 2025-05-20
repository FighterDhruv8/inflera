import subprocess
import sys
import os
print("\nManaging dependencies...\nThis might take a few seconds...\n")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", f"{os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'requirements.txt')}"], stdout = subprocess.DEVNULL)
from pprint import pprint
from argparse import ArgumentParser, Namespace

from document_loader import DocumentLoader
from embeddings import VectorStore
from retrieval import Retriever
from llm import LLMService
from agent import Agent

class main:
    def __init__(self, args: Namespace):
        """Set up the RAG agent system.
        
        Args:
            args: Command line arguments.
        """
        
        print("Setting up RAG agent system...")
        
        self.loader = DocumentLoader()
        self.vector_store = VectorStore()
        
        self.documents = self.loader.load_documents()
        self.chunks = self.loader.chunk_documents(self.documents)
        self.vector_store.add_chunks(self.chunks)
        
        if args.model and not args.model_url:
            self.llm_service = LLMService(model_name = args.model)
        elif args.model_url and not args.model:
            self.llm_service = LLMService(base_url = args.model_url)
        elif args.model and args.model_url:
            self.llm_service = LLMService(model_name = args.model, base_url = args.model_url)
        else:
            self.llm_service = LLMService()
        
        self.retriever = Retriever(self.vector_store)
        
        self.agent = Agent(self.retriever, self.llm_service)
        
        self.prev_response_info = None
        self.logs = None

    def cli_interface(self) -> None:
        """Run a simple CLI interface for the RAG agent.
        
        Returns:
            None
        """
        
        print("\n\nRAG Agent System ready.\nType 'exit' to quit, 'info' for additional information about the last query made, or 'logs' to see the logs of the last query.")
        print("Example queries:")
        print("  - What is your flagship product?")
        
        while True:
            
            query = input("Hi! How can I help you?\n")
            
            if query.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                break
            
            if query.lower() in ["info", "information"]:
                if self.prev_response_info is None:
                    print("Invalid request. There was no query to the LLM made previously.\n")
                else:
                    print("-"*50)
                    print("\nUSAGE INFO:")
                    pprint(self.prev_response_info['Usage info'])
                    print("\nMETADATA:")
                    pprint(self.prev_response_info['Metadata'])
                    print("\nID:")
                    print(self.prev_response_info['ID'])
                    print("-"*50)
                continue
            
            if query.lower() in ["logs", "log"]:
                if self.logs is None:
                    print("Invalid request. There was no query made previously.\n")
                else:
                    print("-"*50)
                    print("LOGS:")
                    for log in self.logs:
                        print(f"- {log}")
                    print("-"*50)
                continue
            
            self.response = self.agent.process_query(query)
            
            if self.response['result'] == "Invalid model.":
                break
            if self.response['result'] == "Error while accessing LLM service. Please ensure the Ollama server is running by running 'ollama ps'.\n(Maybe the model is listening on a different port?)":
                break
            
            self.logs = self.response['log']
            
            print("\n" + "="*50)
            print(f"QUERY: {self.response['query']}")
            print(f"TOOL: {self.response['tool_used']}")
            print("-"*50)
            
            if self.response['tool_used'] == 'rag':
                print("RETRIEVED CHUNKS:")
                for i, chunk in enumerate(self.response['retrieved_chunks']):
                    print(f"\nChunk {i+1} (Source: {chunk['source']}, Score: {chunk['relevance_score']:.2f}):")
                    print(f"{chunk['content'][:200]}...")
                print("-"*50)
            
            if self.response['tool_used'] == 'rag' or self.response['tool_used'] == 'none':
                self.prev_response_info = {
                    "Usage info": self.response['result'].usage_metadata,
                    "Metadata": self.response['result'].response_metadata,
                    "ID": self.response['result'].id
                    }
                
                if self.response['reason']:
                    print("REASONING:")
                    [print(f"{_}") for _ in self.response['reason']]
                    print("-"*50)
                
                print("RESULT:")
                print(self.response['result'].content)
                print("="*50 + "\n")
            else:
                print("RESULT:")
                print(self.response['result'])
                print("="*50 + "\n")

if __name__ == "__main__":
    parser = ArgumentParser(description = "RAG Agent System. Available tools: Calulator, Dictionary, RAG.")
    parser.add_argument("--model", help = "Select the ollama model to use for the LLM service.")
    parser.add_argument("--model_url", help = "Select the url the ollama model exists at.")
    args = parser.parse_args()
    obj = main(args)
    obj.cli_interface()