import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

# Load API Key
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Apply CSS
st.set_page_config(page_title="Chat with PDF", layout="wide")
st.markdown(css, unsafe_allow_html=True)

st.header("📘 Chat with your PDF")

# Upload PDF
pdf = st.file_uploader("Upload your PDF", type="pdf")

if pdf:
    reader = PdfReader(pdf)
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text()

    # Split text
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(raw_text)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    # Initialize LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_key)

    # Conversational Chain
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=vectorstore.as_retriever())
    
    # Chat memory
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User input
    user_question = st.text_input("Ask a question about your PDF:")
    if user_question:
        result = qa({"question": user_question, "chat_history": st.session_state.chat_history})
        st.session_state.chat_history.append((user_question, result["answer"]))

    # Display chat
    for i, (q, a) in enumerate(st.session_state.chat_history):
        st.markdown(user_template.format(q), unsafe_allow_html=True)
        st.markdown(bot_template.format(a), unsafe_allow_html=True)
