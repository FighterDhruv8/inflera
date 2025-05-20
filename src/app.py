import os
import streamlit as st
from argparse import Namespace
from pprint import pformat
from main import main
from time import sleep
from keyboard import press_and_release
import psutil

def load_agent(model_name, model_url):
    """Load the RAG agent with the specified model name and URL.
    
    Args:
        model_name (str): Name of the model to use.
        model_url (str): URL of the model to use.
    """
    args = Namespace(
        model = model_name if model_name.strip() else None,
        model_url = model_url if model_url.strip() else None,
    )
    return main(args)

def exit_app(time: float = 1.5) -> None:
    """Close the app and terminate the process.
    
    Args:
        (optional) time (float): Time (in seconds) to wait before closing the app. Default is 1.5 seconds.
        
    Returns:
        None.
    """
    with col1:
        st.info("Goodbye! Closing the app...")
    sleep(time)
    press_and_release('ctrl+w')
    pid = os.getpid()
    p = psutil.Process(pid)
    p.terminate()

st.set_page_config(layout = "wide")

st.sidebar.header("Model Configuration")

model_name = st.sidebar.text_input("Model Name (optional)", placeholder = "e.g., gemma3:1b", )
model_url = st.sidebar.text_input("Model URL (optional)", placeholder = "e.g., http://localhost:11434")

col1, col2 = st.columns([0.85, 0.15])

with col2:
    exit_app_flag = st.button(label = "Close App", help = "Close the app and terminate the process.", key = "exit_app", icon = ":material/close:")
    if exit_app_flag:
        exit_app()

with col1:
    st.title("RAG Agent Streamlit Interface")

    if "agent_initialized" not in st.session_state:
        st.session_state.agent_initialized = False

    if "rag_agent" not in st.session_state:
        st.session_state.rag_agent = None

    if st.sidebar.button(label = "Initialize Agent", help = "Initialize the RAG agent with the specified model name and URL.", key = "initialize_agent", icon = ":material/robot_2:"):
        with st.spinner("Initializing RAG agent..."):
            st.session_state.rag_agent = load_agent(model_name, model_url)
            st.session_state.agent_initialized = True
            st.session_state.prev_response_info = None
            st.session_state.logs = None
        st.sidebar.success("Agent initialized!")

    if "prev_response_info" not in st.session_state:
        st.session_state.prev_response_info = None

    if "logs" not in st.session_state:
        st.session_state.logs = None

    if not st.session_state.agent_initialized:
        st.info("Please initialize the agent from the sidebar before submitting queries.")
        st.stop()

    query = st.text_input("Enter your query:", key = "query_input", placeholder = "e.g., What is your flagship product?")
    q = query.strip()

    if q:
        if q.lower() in ["exit", "quit", "bye"]:
            exit_app(2)
        elif q.lower() in ["info", "information"]:
            if st.session_state.prev_response_info is None:
                st.warning("Invalid request. There was no query to the LLM made previously.")
            else:
                info = st.session_state.prev_response_info
                st.subheader("USAGE INFO:")
                st.text(pformat(info.get("Usage info")))
                st.subheader("METADATA:")
                st.text(pformat(info.get("Metadata")))
                st.subheader("ID:")
                st.text(str(info.get("ID")))
        elif q.lower() in ["logs", "log"]:
            if st.session_state.logs is None:
                st.warning("Invalid request. There was no query made previously.")
            else:
                st.subheader("LOGS:")
                for log_entry in st.session_state.logs:
                    st.text(log_entry)
        else:
            with st.spinner("Processing your query..."):
                response = st.session_state.rag_agent.agent.process_query(q)
            
            if response['result'] == "Invalid model.":
                st.error("Invalid model. Please check the model name or URL.")
            elif response['result'] == "Error while accessing LLM service. Please ensure the Ollama server is running by running 'ollama ps'.\n(Maybe the model is listening on a different port?)":
                st.error("Error while accessing LLM service. Please ensure the Ollama server is running by running 'ollama ps' in a command prompt on your system.\n(Maybe the model is listening on a different port?)")
            else:

                st.session_state.logs = response.get('log', None)
                result = response.get('result', None)

                if response.get('tool_used') in ['rag', 'none']:
                    usage_info = getattr(result, 'usage_metadata', None)
                    metadata = getattr(result, 'response_metadata', None)
                    res_id = getattr(result, 'id', None)

                    st.session_state.prev_response_info = {
                        "Usage info": usage_info,
                        "Metadata": metadata,
                        "ID": res_id,
                    }

                    content = getattr(result, 'content', str(result))
                else:
                    content = str(result)
                
                st.subheader("Result")
                st.info(content)
                
                if response.get('reason'):
                    st.subheader("Reasoning")
                    for _ in response.get('reason'):
                        st.markdown(_)

                st.subheader("Query Result")
                st.markdown(f"**Query:** {response.get('query', q)}")
                st.markdown(f"**Tool Used:** {response.get('tool_used', 'unknown')}")

                if response.get('tool_used') == 'rag':
                    with st.expander("Show Retrieved Chunks"):
                        st.subheader("Retrieved Chunks")
                        for i, chunk in enumerate(response.get('retrieved_chunks', [])):
                            st.markdown(
                                f"**Chunk {i+1} (Source: {chunk.get('source', 'N/A')}, Score: {chunk.get('relevance_score', 0):.2f})**"
                            )
                            content_snippet = chunk.get('content', '')
                            st.text(content_snippet)
    else:
        st.info("Enter a query above and click enter to start.")