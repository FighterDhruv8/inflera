# How to Run

### Via Streamlit on Your Local System
1. Clone or download this repo to your local system.
2. Make sure you have python installed.
3. _(optional)_ Create a virtual environment using ```python.exe -m venv your_venv_name_here```. If you do this, make sure to use the python interpreter inside the venv going ahead!
4. Run ```python.exe -m pip install streamlit``` via CLI and wait for the installation to finish.
5. Run ```streamlit run app.py``` via CLI.

### Via Command Line Interface
1. Clone or download this repo to your local system.
2. Make sure you have python installed.
3. _(optional)_ Create a virtual environment using ```python.exe -m venv your_venv_name_here```. If you do this, make sure to use the python interpreter inside the venv going ahead!
4. Run the [main.py](src/main.py) file inside the [src](src/) directory **from the command line**.

# How the System Works

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
   - The agent uses RAG tool to answer queries.
   - Defaults to just the LLM if no tool is used.
   - All decision steps are logged for transparency

5. **User Interface**:
   - UI was made using streamlit. The CLI can also be used.
   - Results show the final ouput, reasoning (if any), retrieved context (if any), tool used, process logs, additional information.