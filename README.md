## How to Run

1. Clone or download this repo to your local system.
2. Make sure you have python installed.
3. _(optional)_ Create a virtual environment using ```python -m venv your_venv_name_here```. If you do this, make sure to use the python interpreter inside the venv!
4. Run the [main.py](src/main.py) file inside the [src](src/) directory **from the command line**.

## How the System Works

1. **Data Ingestion**:
   - The system loads sample documents from the [data](data/) directory.
   - Documents are chunked into smaller pieces with overlap for better retrieval.

2. **Vector Store & Retrieval**:
   - Documents are embedded and stored in a ChromaDB database.
   - The retriever finds the top-k most relevant chunks for each query.

3. **LLM Integration**:
   - Ollama's Gemma3:1b is used as the default LLM.
   - The system formats prompts with retrieved context for better responses if needed.

4. **Agentic Workflow**:
   - The agent routes queries to appropriate tools:
     - Calculator for mathematical operations.
     - Dictionary for word definitions.
     - RAG for knowledge-based questions.
   - Defaults to just the LLM if no tools are used.
   - All decision steps are logged for transparency

5. **User Interface**:
   - The CLI can be used, no webapp was made due to lack of time.
   - Results show the tool used, retrieved context, and process logs.

## Next Steps

You could extend this system by:
1. Adding a webapp UI.
2. Adding more specialized tools.
3. Supporting document upload through the UI.
4. Implementing a chat history database.
5. Adding more advanced agent frameworks like ReAct or MRKL.
6. Supporting multiple LLM providers.