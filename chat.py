# streamlit_rag_demo_final.py
import sys
from pathlib import Path

# ----------------------
# Add src/ to Python path
# ----------------------
sys.path.append(str(Path(__file__).parent / "src"))

# ----------------------
# Imports
# ----------------------
import streamlit as st
import os
import docx
import pdfplumber
from src.rag_engine import CollegeRAG  # Your RAG system

# ----------------------
# Configuration
# ----------------------
DATA_DIR = '/home/kaiserkonok/computer_programming/K_RAG/test_data/'
st.set_page_config(page_title="CollegeRAG Demo", layout="wide")

# ----------------------
# Load RAG system (cached)
# ----------------------
@st.cache_resource(show_spinner=True)
def load_rag():
    return CollegeRAG(DATA_DIR)

rag = load_rag()

# ----------------------
# Sidebar Navigation
# ----------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Chat", "File Explorer"])

# ----------------------
# Chat Page
# ----------------------
if page == "Chat":
    st.title("CollegeRAG Chat 💬")

    # Initialize session state for chat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    chat_container = st.container()

    def submit_chat():
        query = st.session_state.input_box.strip()
        if not query:
            return
        st.session_state.input_box = ""  # Clear input
        # Save user message immediately
        st.session_state.chat_history.append(("You", query))
        response = ""
        placeholder = chat_container.empty()

        # Streaming response
        for chunk in rag.llm.stream(rag.ask(query, chat_history=st.session_state.chat_history, stream=True)):
            response += chunk
            with placeholder:
                # Display all previous messages
                for speaker, message in st.session_state.chat_history:
                    if speaker == "You":
                        st.markdown(f"<div style='text-align:right;color:blue;'><b>{speaker}:</b> {message}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align:left;color:green;'><b>{speaker}:</b> {message}</div>", unsafe_allow_html=True)
                # Display current streaming response
                st.markdown(f"<div style='text-align:left;color:green;'><b>AI:</b> {response}</div>", unsafe_allow_html=True)

        # Save AI message to history
        st.session_state.chat_history.append(("AI", response))

    # Fixed input box at bottom (Enter triggers submit_chat)
    st.text_input("Type a message and press Enter", key="input_box", on_change=submit_chat)

# ----------------------
# File Explorer Page
# ----------------------
elif page == "File Explorer":
    st.title("CollegeRAG File Explorer 📁")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Files and Folders")

        def list_files(base_path):
            entries = []
            for f in sorted(os.listdir(base_path)):
                entries.append(os.path.join(base_path, f))
            return entries

        folder_tree = list_files(DATA_DIR)
        selected_file = None

        # Expandable folders and clickable files
        for entry in folder_tree:
            if os.path.isdir(entry):
                with st.expander(os.path.basename(entry)):
                    sub_files = list_files(entry)
                    for f in sub_files:
                        if st.button(f"[File] {os.path.basename(f)}", key=f):
                            selected_file = f
            else:
                if st.button(f"[File] {os.path.basename(entry)}", key=entry):
                    selected_file = entry

        # Upload new files
        uploaded_file = st.file_uploader("Upload document", type=["txt", "pdf", "docx"])
        if uploaded_file:
            save_path = os.path.join(DATA_DIR, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Uploaded: {uploaded_file.name}")
            rag.ingest([save_path])
            st.experimental_rerun()  # Refresh explorer

    with col2:
        if selected_file:
            st.subheader(f"Preview: {os.path.basename(selected_file)}")
            ext = Path(selected_file).suffix.lower()
            text = ""
            try:
                if ext == ".txt":
                    with open(selected_file, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                elif ext == ".docx":
                    doc = docx.Document(selected_file)
                    text = "\n".join([p.text for p in doc.paragraphs])
                elif ext == ".pdf":
                    with pdfplumber.open(selected_file) as pdf:
                        pages = [p.extract_text() for p in pdf.pages if p.extract_text()]
                        text = "\n".join(pages)

                st.text_area("File Content (first 5000 chars)", text[:5000], height=300)

                # Show chunks preview
                chunks = rag.chunker.chunk_document(text)
                st.subheader("Chunks Preview")
                for i, c in enumerate(chunks[:20]):  # Limit to first 20 chunks
                    st.markdown(f"**Chunk {i+1} (ID: {c.id})**")
                    st.write(c.text)

            except Exception as e:
                st.error(f"Failed to load file: {e}")
