import os
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from PyPDF2 import PdfReader, PdfWriter
from tempfile import NamedTemporaryFile
import base64
from htmlTemplates import expander_css, css, bot_template, user_template
from datetime import datetime

# Step 1: Process the PDF and create retrieval chain
def process_file(doc):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    vectorstore = FAISS.from_documents(doc, embeddings)

    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0.3, openai_api_key=os.getenv("OPENAI_API_KEY")),
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True
    )
    return chain

# Step 2: Handle user input and update chat + page
def handle_userinput(query):
    response = st.session_state.conversation(
        {"question": query, 'chat_history': [(q, a) for q, a, _ in st.session_state.chat_history]},
        return_only_outputs=True
    )

    answer = response['answer'].strip()
    source_doc = response['source_documents'][0]
    page_num = source_doc.metadata.get("page", 0)

    timestamp = datetime.now().strftime("%b %d, %I:%M %p")
    st.session_state.chat_history.append((query, answer, timestamp))
    st.session_state.scroll_to_page = page_num

    for message in reversed(st.session_state.chat_history):
        user_msg, bot_msg, ts = message
        st.session_state.expander1.markdown(f"<p style='text-align:right; font-size: 12px; color: gray;'>{ts}</p>", unsafe_allow_html=True)
        st.session_state.expander1.write(user_template.replace("{{MSG}}", user_msg), unsafe_allow_html=True)
        st.session_state.expander1.markdown(f"<p style='text-align:right; font-size: 12px; color: gray;'>{ts}</p>", unsafe_allow_html=True)
        st.session_state.expander1.write(bot_template.replace("{{MSG}}", bot_msg), unsafe_allow_html=True)
        st.session_state.expander1.markdown("<hr style='margin: 5px 0; border: none; border-top: 1px solid #ccc;' />", unsafe_allow_html=True)

# Step 3: Main App
def main():
    load_dotenv()
    st.set_page_config(layout="wide", page_title="Interactive Reader", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    for key in ["conversation", "chat_history", "scroll_to_page", "base64_pdf"]:
        if key not in st.session_state:
            st.session_state[key] = [] if key == "chat_history" else None

    st.session_state.col1, st.session_state.col2 = st.columns([1, 1])
    st.session_state.col1.header("Interactive Reader :books:")
    user_question = st.session_state.col1.text_input("Ask a question on the contents of the uploaded PDF:")
    st.session_state.expander1 = st.session_state.col1.expander('Your Chat', expanded=True)
    st.session_state.col1.markdown(expander_css, unsafe_allow_html=True)

    # Upload and process PDF
    st.session_state.col1.subheader("Your documents")
    st.session_state.pdf_doc = st.session_state.col1.file_uploader("Upload your PDF here and click on 'Process'")

    if st.session_state.col1.button("Process", key='a'):
        with st.spinner("Processing"):
            if st.session_state.pdf_doc is not None:
                with NamedTemporaryFile(suffix="pdf", delete=False) as temp:
                    temp.write(st.session_state.pdf_doc.getvalue())
                    temp.flush()
                    loader = PyMuPDFLoader(temp.name)
                    pdf = loader.load()
                    st.session_state.conversation = process_file(pdf)
                    st.session_state.col1.markdown("✅ Done processing. You may now ask a question.")

                # Save base64 version for iframe
                with open(temp.name, "rb") as f:
                    st.session_state.base64_pdf = base64.b64encode(f.read()).decode("utf-8")

    # Handle Q&A
    if user_question:
        if st.session_state.conversation is not None:
            handle_userinput(user_question)
        else:
            st.session_state.col1.warning("⚠️ Please upload and process a PDF before asking a question.")

    # Generate base64 version only once
    if "base64_pdf" not in st.session_state and st.session_state.pdf_doc is not None:
        with NamedTemporaryFile(suffix="pdf", delete=False) as full_temp:
            full_temp.write(st.session_state.pdf_doc.getvalue())
            full_temp.flush()
            with open(full_temp.name, "rb") as f:
                st.session_state.base64_pdf = base64.b64encode(f.read()).decode("utf-8")

    # Always display the PDF in an iframe — scrollable + page jump
    if st.session_state.base64_pdf:
        scroll_page = (st.session_state.scroll_to_page or 0) + 1  # Page offset

        iframe_code = f"""
        <iframe
            src="data:application/pdf;base64,{st.session_state.base64_pdf}#page={scroll_page}"
            width="100%" height="900" type="application/pdf"
            style="border: none;">
        </iframe>
        """

        st.session_state.col2.markdown(iframe_code, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
