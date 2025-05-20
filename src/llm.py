from typing import Any
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from ollama import _types
import shutil
import re

class LLMService:
    def __init__(self, model_name: str = "gemma3:1b", base_url: str = "http://localhost:11434"):
        """Initialize the LLM service using LangChain and Ollama.
        
        Args:
            (optional) model_name: Name of the Ollama model to use. Default is "gemma3:1b".
            (optional) base_url: Base URL for the Ollama API. Default is "http://localhost:11434".
        """
        self.model_name = model_name
        self.base_url = base_url
        
        if shutil.which("ollama") is None:
            print("Could not detect Ollama. Please install it from https://ollama.com/download or add it to PATH.")
            print("(I tried to find it using the name 'ollama'. Please rename the PATH variable to 'ollama' if it has a different name.)")
            exit(1)
        
        self.llm = ChatOllama(
            model = model_name,
            base_url = base_url,
            temperature = 0.2,
            num_predict = 500
        )
        
        self.qa_with_context_template = PromptTemplate(
            input_variables = ["context", "question"],
            template = (
                '''You are an AI assistant with access to the following information.
                Use this information to answer the user's question.
                If the information doesn't contain the answer, say so. Do not make up information.
                If the question seems nonsensical, say so.
                CONTEXT INFORMATION:\n{context}\n
                USER QUESTION: {question}'''
            )
        )
        
        self.qa_without_context_template = PromptTemplate(
            input_variables = ["question"],
            template = (
                '''You are a helpful AI assistant. Answer the user's question based on your knowledge. If the question seems nonsensical, say so.
                USER QUESTION: {question}'''
            )
        )
    
    def _remove_reasoning_tags(self, message: AIMessage) -> tuple[AIMessage, list[str]]:
        """Remove reasoning tags from the AI message.
        
        Args:
            message: The LLM output to process.
            
        Returns:
            Tuple containing the LLM output with reasoning removed and the reasoning text.
        """
        
        reason = re.findall(r'<think>(.*?)</think>', message.content, flags = re.DOTALL)
        message.content = re.sub(r'<think>.*?</think>', '', message.content, flags = re.DOTALL)
        return message, reason
    
    def generate_response(self, query: str, context_chunks: list[dict[str, Any]] = None) -> str:
        """Generate a response using the Ollama model via LangChain.
        
        Args:
            query: The user query.
            (optional) context_chunks: List of context chunks for the query. Default is None.
            
        Returns:
            Generated LLM response.
        """
        
        llm = self.llm
        print(f"Loading response using the {self.model_name} model...\n")
        try:
            if context_chunks:
                context_text = "\n\n".join([chunk['content'] for chunk in context_chunks])
                prompt = self.qa_with_context_template
                chain = prompt | llm
                response = chain.invoke(input = {"context": context_text, "question": query})
            else:
                prompt = self.qa_without_context_template
                chain = prompt | llm
                response = chain.invoke(query)
            return self._remove_reasoning_tags(response)
        except _types.ResponseError as e:
            s = "Invalid model."
            print(s)
            return s, None
        except Exception as e:
            s = "Error while accessing LLM service. Please ensure the Ollama server is running by running 'ollama ps'.\n(Maybe the model is listening on a different port?)"
            print(s)
            return s, None