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
st.markdown(
    "Upload PDF files and chat with your documents"
)

with st.sidebar:

    st.header("⚙️ Settings")

    mistral_api_key = st.text_input(
        "Enter Mistral API Key",
        type="password"
    )

    st.markdown(
        """
[Create Free Mistral API Key](https://console.mistral.ai/api-keys)
"""
    )

    exam_mode = st.toggle(
        "🎓 RGPV Exam Mode"
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

api_key = mistral_api_key or os.getenv(
    "MISTRAL_API_KEY"
)

if not api_key:

    st.warning(
        "Please enter your Mistral API key from sidebar."
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

except Exception:

    st.error(
        "Invalid Mistral API Key"
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
            embedding=embedding_model
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
            "Unable to connect with Mistral AI"
        )

        st.stop()

    memory_prompt = ChatPromptTemplate.from_messages([

        (
            "system",

            """
You are a query rewriting AI.

Convert follow-up questions into
standalone questions using chat history.

Example:

History:
User: What is waterfall model?

Follow Up:
Explain its types

Standalone Question:
Explain types of waterfall model

Only return the standalone question.
"""
        ),

        (
            "human",

            """
Chat History:
{chat_history}

Follow Up Question:
{question}

Standalone Question:
"""
        )
    ])

    prompt = ChatPromptTemplate.from_messages([

        (
            "system",

            """
You are an advanced AI PDF assistant.

Answer ONLY from provided context.

Rules:
1. Understand follow-up questions.
2. Use chat history properly.
3. Give detailed and accurate answers.
4. Give exam-oriented answers if needed.
5. Use bullet points where useful.
6. Mention source file and page number.
7. If answer does not exist say:
"I could not find the answer in the document."
8. Never hallucinate.
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
        "Ask anything from your PDFs..."
    )

    if query:

        if exam_mode:

            query += """
            
Explain for RGPV exam with:
- definition
- important points
- advantages/disadvantages if possible
- conclusion
"""

        st.session_state.messages.append({

            "role": "user",
            "content": query
        })

        st.session_state.chat_history.append(
            f"User: {query}"
        )

        with st.chat_message("user"):

            st.markdown(query)

        memory_chain = memory_prompt | llm

        standalone_question = (
            memory_chain.invoke({

                "chat_history": "\n".join(
                    st.session_state.chat_history[-6:]
                ),

                "question": query
            })
        )

        final_query = (
            standalone_question.content
        )

        docs = retriever.invoke(
            final_query
        )

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

            content = doc.page_content

            context += f"""

Source File: {source}

Page Number: {page}

Content:
{content}

"""

            sources.append(
                f"{source} - Page {page}"
            )

        final_prompt = prompt.invoke({

            "chat_history": "\n".join(
                st.session_state.chat_history[-6:]
            ),

            "context": context,

            "question": final_query
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

                st.markdown(
                    ai_response
                )

        st.session_state.messages.append({

            "role": "assistant",
            "content": ai_response
        })

        st.session_state.chat_history.append(
            f"Assistant: {response.content}"
        )