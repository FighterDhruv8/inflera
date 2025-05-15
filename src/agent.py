import re
from typing import Any
from retrieval import Retriever
from llm import LLMService
from tools import Calculator, Dictionary

class Agent:
    def __init__(self, retriever: Retriever, llm_service: LLMService):
        """Initialize the agent with necessary components.
        
        Args:
            retriever: Retriever instance for retrieving relevant chunks.
            llm_service: LLMService instance for generating responses.
        """
        self.retriever = retriever
        self.llm_service = llm_service
        self.calculator = Calculator()
        self.dictionary = Dictionary()
        
        self.calculator_keywords = ['calculate', 'computation', 'compute', 'solve', 'what is', 'equals', 'result of', 'evaluate']
        self.dictionary_keywords = ['define', 'definition', 'meaning', 'what does', 'mean', 'what is', 'explain', 'describe']
        
    def _detect_tool(self, query: str) -> tuple[str, dict[str, Any]]:
        """Detect which tool to use based on the query.
        
        Args:
            query: User query string.
            
        Returns:
            Tuple of (tool_name, tool_result).
        """
        if any(kw in query.lower() for kw in self.calculator_keywords) and re.search(r'\d', query):
            calc_result = self.calculator.calculate(query)
            if calc_result["success"]:
                return "calculator", calc_result
        
        if any(kw in query.lower() for kw in self.dictionary_keywords):
            dict_result = self.dictionary.define(query)
            if dict_result["success"]:
                return "dictionary", dict_result
                
        return "other", {}
    
    def process_query(self, query: str) -> dict[str, Any]:
        """Process a user query and return a response.
        
        Args:
            query: User query string.
            
        Returns:
            Dictionary with response and process information.
        """
        tool_name, tool_result = self._detect_tool(query)
        
        response = {
            "query": query,
            "tool_used": tool_name,
            "log": [f"Agent detected tool: {tool_name}"],
            "reason": None
        }
        
        if tool_name == "calculator":
            response["result"] = tool_result["explanation"]
            response["log"].append(f"Calculator processed: {tool_result['expression']} = {tool_result['result']}")
            
        elif tool_name == "dictionary":
            definitions = []
            for i, def_entry in enumerate(tool_result["definitions"]):
                definitions.append(f"{i+1}. ({def_entry['part_of_speech']}) {def_entry['definition']}")
                if def_entry['example']:
                    definitions.append(f"   Example: {def_entry['example']}")
                    
            response["result"] = f"Definition of '{tool_result['word']}':\n" + "\n".join(definitions)
            response["log"].append(f"Dictionary looked up: {tool_result['word']}")
            
        else:
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