import streamlit as st
from dotenv import load_dotenv
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

# Load env variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 PDF RAG Chatbot")
st.write("Upload a PDF and chat with your document")

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type="pdf"
)

# Session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Process PDF
if uploaded_file is not None:

    with st.spinner("Processing PDF..."):

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:

            tmp_file.write(uploaded_file.read())
            temp_pdf_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(temp_pdf_path)
        docs = loader.load()

        # Split text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        # Embedding model
        embedding_model = MistralAIEmbeddings()

        # Create vector DB
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory="chroma_db"
        )

        st.session_state.vectorstore = vectorstore

        st.success("PDF processed successfully!")

# Chat section
if st.session_state.vectorstore:

    retriever = st.session_state.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    llm = ChatMistralAI(
        model="mistral-small-2506"
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a helpful AI assistant.

Use only the provided context to answer the question.

If the answer is not present in the context,
say:
"I could not find the answer in the document."
"""
        ),
        (
            "human",
            """
Context:
{context}

Question:
{question}
"""
        )
    ])

    # Show previous chats
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    query = st.chat_input("Ask question from PDF...")

    if query:

        # Show user message
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("user"):
            st.markdown(query)

        # Retrieve docs
        docs = retriever.invoke(query)

        # Create context
        context = "\n\n".join([
            doc.page_content for doc in docs
        ])

        # Final prompt
        final_prompt = prompt.invoke({
            "context": context,
            "question": query
        })

        # Generate response
        response = llm.invoke(final_prompt)

        ai_response = response.content

        # Save AI response
        st.session_state.messages.append({
            "role": "assistant",
            "content": ai_response
        })

        # Display AI response
        with st.chat_message("assistant"):
            st.markdown(ai_response)