import streamlit as st
from dotenv import load_dotenv
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_mistralai import (
    MistralAIEmbeddings,
    ChatMistralAI
)

from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
    page_title="Advanced PDF RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Advanced PDF RAG Chatbot")
st.markdown("Upload PDF files and chat with your documents")

with st.sidebar:

    st.header("⚙️ Settings")

    mistral_api_key = st.text_input(
        "Enter Mistral API Key",
        type="password"
    )

    st.markdown(
        "[Create Free Mistral API Key](https://console.mistral.ai/api-keys)"
    )

    chunk_size = st.slider(
        "Chunk Size",
        500,
        2000,
        1000
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        0,
        500,
        200
    )

    k_value = st.slider(
        "Retrieved Chunks",
        1,
        10,
        4
    )

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []
        st.session_state.chat_history = []

        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY")

if not api_key:

    st.warning(
        "Please add your Mistral API key from the sidebar."
    )

    st.stop()

@st.cache_resource
def load_embedding_model(api_key):

    return MistralAIEmbeddings(
        api_key=api_key
    )

try:

    embedding_model = load_embedding_model(
        api_key
    )

except Exception as e:

    st.error(
        "Invalid Mistral API Key."
    )

    st.stop()

uploaded_files = st.file_uploader(
    "Upload PDF Files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    with st.spinner("Processing PDFs..."):

        all_docs = []

        for uploaded_file in uploaded_files:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp_file:

                tmp_file.write(
                    uploaded_file.read()
                )

                temp_pdf_path = tmp_file.name

            loader = PyPDFLoader(
                temp_pdf_path
            )

            docs = loader.load()

            for doc in docs:

                doc.metadata["source"] = (
                    uploaded_file.name
                )

            all_docs.extend(docs)

            os.remove(temp_pdf_path)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        chunks = splitter.split_documents(
            all_docs
        )

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory="chroma_db"
        )

        st.session_state.vectorstore = (
            vectorstore
        )

        st.success(
            "PDFs processed successfully!"
        )

if st.session_state.vectorstore:

    retriever = (
        st.session_state.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k_value,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )
    )

    try:

        llm = ChatMistralAI(
            model="mistral-small-2506",
            temperature=0.3,
            api_key=api_key
        )

    except Exception:

        st.error(
            "Unable to connect to Mistral AI."
        )

        st.stop()

    prompt = ChatPromptTemplate.from_messages([

        (
            "system",

            """
You are an intelligent AI PDF assistant.

Answer ONLY from the provided context.

Rules:
1. Give accurate answers.
2. Use previous conversation memory.
3. Mention source file name and page number.
4. If answer is missing, say:
'I could not find the answer in the document.'
5. Do not hallucinate.
"""
        ),

        (
            "human",

            """
Chat History:
{chat_history}

Context:
{context}

Question:
{question}

Answer:
"""
        )
    ])

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    query = st.chat_input(
        "Ask anything from PDFs..."
    )

    if query:

        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        st.session_state.chat_history.append(
            f"User: {query}"
        )

        with st.chat_message("user"):

            st.markdown(query)

        docs = retriever.invoke(query)

        context = ""

        sources = []

        for doc in docs:

            source = doc.metadata.get(
                "source",
                "Unknown File"
            )

            page = doc.metadata.get(
                "page",
                "Unknown"
            )

            context += f"""

Source File: {source}

Page Number: {page}

Content:
{doc.page_content}

"""

            sources.append(
                f"{source} - Page {page}"
            )

        final_prompt = prompt.invoke({

            "chat_history": "\n".join(
                st.session_state.chat_history[-6:]
            ),

            "context": context,

            "question": query
        })

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                response = llm.invoke(
                    final_prompt
                )

                ai_response = (
                    response.content
                )

                ai_response += (
                    "\n\n---\n\n📚 Sources:\n"
                )

                for src in list(set(sources)):

                    ai_response += (
                        f"- {src}\n"
                    )

                st.markdown(ai_response)

        st.session_state.messages.append({

            "role": "assistant",
            "content": ai_response
        })

        st.session_state.chat_history.append(
            f"Assistant: {response.content}"
        )