import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from PyPDF2 import PdfReader
from tempfile import NamedTemporaryFile
import base64
from htmlTemplates import expander_css, css, bot_template, user_template
from datetime import datetime
import fitz  # PyMuPDF

# Task 4: Process the Input PDF
def process_file(doc):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    pdfsearch = FAISS.from_documents(doc, embeddings)

    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0.3, openai_api_key=os.getenv("OPENAI_API_KEY")),
        retriever=pdfsearch.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True
    )
    return chain

# Task 6: Method for Handling User Input
def handle_userinput(query):
    response = st.session_state.conversation(
        {"question": query, 'chat_history': [(q, a) for q, a, _ in st.session_state.chat_history]},
        return_only_outputs=True
    )

    source_doc = response['source_documents'][0]
    chunk = source_doc.page_content.strip()
    page_num = source_doc.metadata.get("page", 0)

    # Extract actual matching line from the page
    with NamedTemporaryFile(suffix="pdf", delete=False) as temp:
        temp.write(st.session_state.pdf_doc.getvalue())
        temp.flush()
        doc = fitz.open(temp.name)
        page = doc.load_page(page_num)
        page_text = page.get_text("text")
        doc.close()

    matching_text = ""
    for line in chunk.splitlines():
        line = line.strip()
        if line in page_text and len(line) > len(matching_text):
            matching_text = line

    if not matching_text:
        matching_text = chunk[:100]

    # Store response & highlight source
    timestamp = datetime.now().strftime("%b %d, %I:%M %p")
    st.session_state.chat_history.append((query, response['answer'], timestamp))
    st.session_state.source_info = {"text": matching_text, "page": page_num}

    # Show chat
    for message in reversed(st.session_state.chat_history):
        user_msg, bot_msg, ts = message
        st.session_state.expander1.markdown(f"<p style='text-align:right; font-size: 12px; color: gray;'>{ts}</p>", unsafe_allow_html=True)
        st.session_state.expander1.write(user_template.replace("{{MSG}}", user_msg), unsafe_allow_html=True)
        st.session_state.expander1.markdown(f"<p style='text-align:right; font-size: 12px; color: gray;'>{ts}</p>", unsafe_allow_html=True)
        st.session_state.expander1.write(bot_template.replace("{{MSG}}", bot_msg), unsafe_allow_html=True)
        st.session_state.expander1.markdown("<hr style='margin: 5px 0; border: none; border-top: 1px solid #ccc;' />", unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(layout="wide", page_title="Interactive Reader", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "source_info" not in st.session_state:
        st.session_state.source_info = None

    st.session_state.col1, st.session_state.col2 = st.columns([1, 1])
    st.session_state.col1.header("Interactive Reader :books:")
    user_question = st.session_state.col1.text_input("Ask a question on the contents of the uploaded PDF:")
    st.session_state.expander1 = st.session_state.col1.expander('Your Chat', expanded=True)
    st.session_state.col1.markdown(expander_css, unsafe_allow_html=True)

    # Upload & process PDF
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

    # Handle user query + highlight
    if user_question:
        if st.session_state.conversation is not None:
            handle_userinput(user_question)

            with NamedTemporaryFile(suffix="pdf", delete=False) as temp:
                temp.write(st.session_state.pdf_doc.getvalue())
                temp.flush()

                doc = fitz.open(temp.name)
                highlight_text = st.session_state.source_info["text"]
                page_num = st.session_state.source_info["page"]

                try:
                    page = doc.load_page(page_num)
                    matches = page.search_for(highlight_text)
                    if matches:
                        for m in matches:
                            page.add_highlight_annot(m)
                    else:
                        print("❗ No exact match found for highlighting.")
                except Exception as e:
                    print("Highlighting error:", e)

                with NamedTemporaryFile(suffix="pdf", delete=False) as highlighted_temp_pdf:
                    doc.save(highlighted_temp_pdf.name)
                    doc.close()

                    with open(highlighted_temp_pdf.name, "rb") as f:
                        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

                    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}#page={page_num + 1}" ' \
                                  f'width="100%" height="900" type="application/pdf" frameborder="0"></iframe>'
                    st.session_state.col2.markdown(pdf_display, unsafe_allow_html=True)
        else:
            st.session_state.col1.warning("⚠️ Please upload and process a PDF before asking a question.")

if __name__ == '__main__':
    main()
