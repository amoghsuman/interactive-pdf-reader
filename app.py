import os
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from tempfile import NamedTemporaryFile
import base64
from htmlTemplates import expander_css, css, bot_template, user_template
from datetime import datetime

def process_file(docs):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_documents(docs, embeddings)
    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0.3, openai_api_key=os.getenv("OPENAI_API_KEY")),
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True
    )
    return chain

def handle_userinput(query):
    response = st.session_state.conversation(
        {"question": query, 'chat_history': [(q, a) for q, a, _ in st.session_state.chat_history]},
        return_only_outputs=True
    )
    answer = response["answer"].strip()
    timestamp = datetime.now().strftime("%b %d, %I:%M %p")
    st.session_state.chat_history.append((query, answer, timestamp))

    for q, a, ts in reversed(st.session_state.chat_history):
        st.session_state.expander1.markdown(f"<p style='text-align:right; font-size: 12px; color: gray;'>{ts}</p>", unsafe_allow_html=True)
        st.session_state.expander1.write(user_template.replace("{{MSG}}", q), unsafe_allow_html=True)
        st.session_state.expander1.write(bot_template.replace("{{MSG}}", a), unsafe_allow_html=True)
        st.session_state.expander1.markdown("<hr style='margin: 5px 0; border: none; border-top: 1px solid #ccc;' />", unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(layout="wide", page_title="Interactive PDF Reader", page_icon="📘")
    st.write(css, unsafe_allow_html=True)

    for key in ["conversation", "chat_history", "pdf_base64"]:
        if key not in st.session_state:
            st.session_state[key] = [] if key == "chat_history" else None

    col1, col2 = st.columns([1, 1])
    col1.header("📘 Ask Questions on PDF")

    with col1.form("ask_form", clear_on_submit=True):
        user_question = st.text_input("Ask something from the PDF:")
        submitted = st.form_submit_button("Ask")

    st.session_state.expander1 = col1.expander("Your Chat", expanded=True)
    col1.markdown(expander_css, unsafe_allow_html=True)

    col1.subheader("Upload Your PDF")
    uploaded_file = col1.file_uploader("Upload and click Process")

    if col1.button("Process") and uploaded_file:
        with st.spinner("Processing..."):
            file_bytes = uploaded_file.read()
            st.session_state.pdf_base64 = base64.b64encode(file_bytes).decode("utf-8")

            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_bytes)
                loader = PyMuPDFLoader(tmp.name)
                docs = loader.load()
                st.session_state.conversation = process_file(docs)

            col1.success("✅ Processed. You can ask questions now.")

    if submitted and user_question:
        if st.session_state.conversation:
            handle_userinput(user_question)
        else:
            col1.warning("⚠️ Please process a PDF first.")

    if st.session_state.pdf_base64:
        col2.markdown("#### 📄 Full PDF Viewer", unsafe_allow_html=True)
        pdf_display = f"""
        <iframe src="data:application/pdf;base64,{st.session_state.pdf_base64}"
                width="100%" height="900px"
                style="border: none;">
        </iframe>
        """
        col2.markdown(pdf_display, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
