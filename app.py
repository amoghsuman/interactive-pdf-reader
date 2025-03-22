import os
import re
import string
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
from fuzzywuzzy import fuzz

# Step 1: Process PDF
def process_file(doc):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    vectorstore = FAISS.from_documents(doc, embeddings)

    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0.3, openai_api_key=os.getenv("OPENAI_API_KEY")),
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True
    )
    return chain

# Step 2: Handle User Input
def handle_userinput(query):
    response = st.session_state.conversation(
        {"question": query, 'chat_history': [(q, a) for q, a, _ in st.session_state.chat_history]},
        return_only_outputs=True
    )

    answer = response['answer'].strip()
    source_doc = response['source_documents'][0]
    chunk = source_doc.page_content.strip()
    page_num = source_doc.metadata.get("page", 0)

    highlight_text = find_best_match_using_fuzzy_window(answer, page_num)

    timestamp = datetime.now().strftime("%b %d, %I:%M %p")
    st.session_state.chat_history.append((query, answer, timestamp))
    st.session_state.source_info = {"text": highlight_text, "page": page_num}

    for message in reversed(st.session_state.chat_history):
        user_msg, bot_msg, ts = message
        st.session_state.expander1.markdown(f"<p style='text-align:right; font-size: 12px; color: gray;'>{ts}</p>", unsafe_allow_html=True)
        st.session_state.expander1.write(user_template.replace("{{MSG}}", user_msg), unsafe_allow_html=True)
        st.session_state.expander1.markdown(f"<p style='text-align:right; font-size: 12px; color: gray;'>{ts}</p>", unsafe_allow_html=True)
        st.session_state.expander1.write(bot_template.replace("{{MSG}}", bot_msg), unsafe_allow_html=True)
        st.session_state.expander1.markdown("<hr style='margin: 5px 0; border: none; border-top: 1px solid #ccc;' />", unsafe_allow_html=True)

# Step 3: Fuzzy sliding window match
def find_best_match_using_fuzzy_window(answer, page_num, window_size=6):
    with NamedTemporaryFile(suffix="pdf", delete=False) as temp:
        temp.write(st.session_state.pdf_doc.getvalue())
        temp.flush()
        doc = fitz.open(temp.name)
        page = doc.load_page(page_num)
        words = page.get_text("words", delimiters=",.")  # (x0, y0, x1, y1, "word")
        doc.close()

    answer_clean = answer.translate(str.maketrans('', '', string.punctuation)).lower()
    answer_tokens = answer_clean.split()

    best_score = 0
    best_indices = (None, None)
    best_phrase = ""

    for i in range(len(words) - 1):
        for w in range(2, window_size + 1):
            if i + w > len(words): break
            window_tokens = [words[j][4].strip(string.punctuation).lower() for j in range(i, i + w)]
            window_phrase = " ".join(window_tokens)
            score = fuzz.partial_ratio(" ".join(answer_tokens), window_phrase)
            if score > best_score:
                best_score = score
                best_indices = (i, i + w - 1)
                best_phrase = window_phrase

    if best_score > 70:
        st.session_state.highlight_indices = (page_num, words, best_indices)
        return best_phrase
    else:
        st.session_state.highlight_indices = None
        return answer[:80]

# Step 4: Streamlit Frontend
def main():
    load_dotenv()
    st.set_page_config(layout="wide", page_title="Interactive Reader", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    for key in ["conversation", "chat_history", "source_info", "highlight_indices"]:
        if key not in st.session_state:
            st.session_state[key] = None if key != "chat_history" else []

    st.session_state.col1, st.session_state.col2 = st.columns([1, 1])
    st.session_state.col1.header("Interactive Reader :books:")
    user_question = st.session_state.col1.text_input("Ask a question on the contents of the uploaded PDF:")
    st.session_state.expander1 = st.session_state.col1.expander('Your Chat', expanded=True)
    st.session_state.col1.markdown(expander_css, unsafe_allow_html=True)

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

    if user_question:
        if st.session_state.conversation is not None:
            handle_userinput(user_question)

            with NamedTemporaryFile(suffix="pdf", delete=False) as temp:
                temp.write(st.session_state.pdf_doc.getvalue())
                temp.flush()
                doc = fitz.open(temp.name)
                page_num = st.session_state.source_info["page"]
                highlight_text = st.session_state.source_info["text"]
                page = doc.load_page(page_num)

                if st.session_state.highlight_indices:
                    _, words, (start, end) = st.session_state.highlight_indices
                    rects = [fitz.Rect(words[i][:4]) for i in range(start, end + 1)]
                    page.add_highlight_annot(rects)
                else:
                    matches = page.search_for(highlight_text)
                    for m in matches:
                        page.add_highlight_annot(m)

                with NamedTemporaryFile(suffix="pdf", delete=False) as highlighted_temp_pdf:
                    doc.save(highlighted_temp_pdf.name)
                    doc.close()
                    with open(highlighted_temp_pdf.name, "rb") as f:
                        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

                st.session_state.col2.markdown(
                    f'<iframe src="data:application/pdf;base64,{base64_pdf}#page={page_num + 1}" '
                    f'width="100%" height="900" type="application/pdf" frameborder="0"></iframe>',
                    unsafe_allow_html=True
                )
        else:
            st.session_state.col1.warning("⚠️ Please upload and process a PDF before asking a question.")

if __name__ == '__main__':
    main()
