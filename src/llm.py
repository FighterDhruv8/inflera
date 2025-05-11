from typing import List, Dict, Any
from langchain_ollama import ChatOllama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
import os

class LLMService:
    def __init__(self, model_name: str = "gemma3:1b", base_url: str = "http://localhost:11434"):
        """Initialize the LLM service using LangChain and Ollama.
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for the Ollama API
        """
        self.model_name = model_name
        self.base_url = base_url
        
        print("\n\nPulling the LLM from Ollama...\n\n")
        os.system('ollama pull gemma3:1b')
        
        self.llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=0.5,
            num_predict=500
        )
        
        self.qa_with_context_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are an AI assistant with access to the following information. "
                "Use this information to answer the user's question. "
                "If the information doesn't contain the answer, say so. Do not make up information.\n\n"
                "CONTEXT INFORMATION:\n{context}\n\n"
                "USER QUESTION: {question}"
            )
        )
        
        self.qa_without_context_template = PromptTemplate(
            input_variables=["question"],
            template=(
                "You are a helpful AI assistant. Answer the user's question based on your knowledge.\n\n"
                "USER QUESTION: {question}"
            )
        )
        
    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]] = None) -> str:
        """Generate a response using the Ollama model via LangChain.
        
        Args:
            query: User query
            context_chunks: List of context chunks for the query
            stream: Whether to stream the response (prints to stdout)
            
        Returns:
            Generated response
        """
        
        llm = self.llm
        
        try:
            if context_chunks:
                context_text = "\n\n".join([chunk['content'] for chunk in context_chunks])
                prompt=self.qa_with_context_template
                chain = prompt | llm
                response = chain.invoke(input = {"context": context_text, "question": query})
            else:
                prompt=self.qa_without_context_template
                chain = prompt | llm
                response = chain.invoke(query)
            return response
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            return f"I encountered an error while processing your request. {error_msg}"
    
    def rag_pipeline(self, query: str, retriever: Any, stream: bool = False) -> str:
        """Run a complete RAG (Retrieval-Augmented Generation) pipeline.
        
        Args:
            query: User query
            retriever: A LangChain retriever component
            stream: Whether to stream the response
            
        Returns:
            Generated response
        """
        try:
            retrieved_docs = retriever.get_relevant_documents(query)
            
            context_chunks = [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in retrieved_docs
            ]
            
            return self.generate_response(
                query=query,
                context_chunks=context_chunks,
                stream=stream
            )
            
        except Exception as e:
            error_msg = f"Error in RAG pipeline: {str(e)}"
            print(error_msg)
            return f"I encountered an error in the retrieval-augmented generation process. {error_msg}"