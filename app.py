import os
import streamlit as st
import streamlit.components.v1 as components  # add this at the top
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

    # Show latest chat at top
    for user_msg, bot_msg, ts in reversed(st.session_state.chat_history):
        st.session_state.expander1.markdown(f"<p style='text-align:right; font-size: 12px; color: gray;'>{ts}</p>", unsafe_allow_html=True)
        st.session_state.expander1.write(user_template.replace("{{MSG}}", user_msg), unsafe_allow_html=True)
        st.session_state.expander1.write(bot_template.replace("{{MSG}}", bot_msg), unsafe_allow_html=True)
        st.session_state.expander1.markdown("<hr style='margin: 5px 0; border: none; border-top: 1px solid #ccc;' />", unsafe_allow_html=True)

# Step 3: Main app
def main():
    load_dotenv()
    st.set_page_config(layout="wide", page_title="Interactive Reader", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize session state
    for key in ["conversation", "chat_history", "scroll_to_page", "base64_pdf"]:
        if key not in st.session_state:
            st.session_state[key] = [] if key == "chat_history" else None

    st.session_state.col1, st.session_state.col2 = st.columns([1, 1])
    st.session_state.col1.header("Interactive Reader :books:")
    with st.session_state.col1.form("user_input_form", clear_on_submit=True):
        user_question = st.text_input("Ask a question on the contents of the uploaded PDF:")
        submitted = st.form_submit_button("Ask")

    st.session_state.expander1 = st.session_state.col1.expander("Your Chat", expanded=True)
    st.session_state.col1.markdown(expander_css, unsafe_allow_html=True)

    # Upload and process PDF
    st.session_state.col1.subheader("Your documents")
    uploaded_pdf = st.session_state.col1.file_uploader("Upload your PDF here and click on 'Process'")

    if st.session_state.col1.button("Process", key="process_btn"):
        if uploaded_pdf is not None:
            with st.spinner("Processing..."):
                # Save uploaded file for future reference
                with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_pdf.getvalue())
                    temp_file.flush()
                    st.session_state.pdf_path = temp_file.name

                # Generate embeddings and setup chain
                loader = PyMuPDFLoader(st.session_state.pdf_path)
                docs = loader.load()
                st.session_state.conversation = process_file(docs)

                # Cache the base64-encoded PDF once
                with open(st.session_state.pdf_path, "rb") as f:
                    st.session_state.base64_pdf = base64.b64encode(f.read()).decode("utf-8")

                st.session_state.col1.success("✅ Done processing. You may now ask a question.")

    # Process question
    if submitted and user_question:
        if st.session_state.conversation is not None:
            handle_userinput(user_question)
        else:
            st.session_state.col1.warning("⚠️ Please upload and process a PDF before asking a question.")

    # Display the full PDF in a scrollable iframe, jump to page if set
    # Show the full PDF with correct iframe embedding (no new tab)
    # Show the full PDF in a scrollable iframe, staying embedded
    if st.session_state.base64_pdf:
        page = (st.session_state.scroll_to_page or 0) + 1
        iframe_html = f"""
            <iframe
                src="data:application/pdf;base64,{st.session_state.base64_pdf}#page={page}"
                width="100%"
                height="900"
                type="application/pdf"
                style="border: none;"
            ></iframe>
        """
        st.session_state.col2.markdown("#### 📄 PDF Viewer", unsafe_allow_html=True)
        st.session_state.col2.markdown(iframe_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
