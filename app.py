import tempfile
from pathlib import Path

import streamlit as st
from streamlit.logger import get_logger

from RAG import DocumentProcessor, RAGQueryProcessor
from utils import copy_files_to_dir, empty_dir

st.set_page_config(layout="wide")
st.title("Local RAG - Query Your Documents Using ChatGPT")

st.markdown(
    """
    <style>
        [data-testid=stSidebar] {
            width: 500px !important;
            background-color: #99b5f0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

logger = get_logger(__name__)


TMP_DIR = Path(tempfile.gettempdir()) / "local_rag_tmp"
Path.mkdir(TMP_DIR, exist_ok=True)


def process_documents():
    copy_files_to_dir(TMP_DIR, st.session_state.source_docs)

    document_handler = DocumentProcessor(st.session_state.openai_api_key)
    documents = document_handler.load_documents(TMP_DIR)
    texts = document_handler.split_documents(documents)
    empty_dir(TMP_DIR)

    st.session_state.retriever = document_handler.get_retriever(texts)


def sidebar():
    st.header("Upload Documents")
    try:
        if st.secrets and "openai_api_key" in st.secrets:
            st.session_state.openai_api_key = st.secrets.openai_api_key
    except Exception as e:
        logger.warning("OpenAI API Key was not found in the secret file.")
        st.session_state.openai_api_key = st.text_input(
            "OpenAI API key", type="password"
        )

    st.markdown("Load your files and let LMM help you query your files")
    st.markdown("---")

    st.session_state.source_docs = st.file_uploader(
        "Upload your files (pdf, txt, docx)",
        type=("txt", "pdf", "docx"),
        accept_multiple_files=True,
    )

    st.button("Load documents", on_click=process_documents)


def chat_page():
    st.header("Query your documents")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.stop()

    for message in st.session_state.messages:
        st.chat_message(message["role"]).write(message["content"])

    if query := st.chat_input("what is your question?"):
        if "local_rag_chain" not in st.session_state:
            st.session_state.local_rag_chain = RAGQueryProcessor(
                st.session_state.openai_api_key, st.session_state.retriever
            )

        st.chat_message("human").write(query)
        response = st.session_state.local_rag_chain.query_LLM(
            query, st.session_state.messages
        )
        st.chat_message("ai").write(response)
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.messages.append({"role": "ai", "content": response})
        # st.session_state.messages.append((query, response))


if __name__ == "__main__":
    with st.sidebar:
        sidebar()

    chat_page()
