import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template, expander_css

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Configure page
st.set_page_config(layout="wide", page_title="Interactive Reader", page_icon="📚")
st.write(css, unsafe_allow_html=True)

# Initialize session variables
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "N" not in st.session_state:
    st.session_state.N = 0

# Layout: two columns
st.session_state.col1, st.session_state.col2 = st.columns([1, 1])

# Left column: header, input, chat
st.session_state.col1.header("📘 Interactive Reader")
user_question = st.session_state.col1.text_input("Ask a question on the contents of the uploaded PDF:")
st.session_state.col1.markdown(expander_css, unsafe_allow_html=True)
st.session_state.expander1 = st.session_state.col1.expander("Your Chat", expanded=True)

# Right column: PDF rendering placeholder (will integrate actual PDF view later)
st.session_state.col2.info("📄 PDF preview will appear here after upload.")
