from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
docs = [
    Document(
        page_content="Python is widely used in Artificial Intelligence.",
        metadata={"source": "AI Notes"}
    ),

    Document(
        page_content="Pandas is used for data analysis in Python.",
        metadata={"source": "Data Science Book"}
    ),

    Document(
        page_content="Neural networks are used in deep learning.",
        metadata={"source": "Deep Learning Notes"}
    )
]

embeddings =  MistralAIEmbeddings()

vectorstore = Chroma.from_documents(
    docs,
    embeddings,
    persist_directory="./db"
)


result = vectorstore.similarity_search(" what is used for data analysis?",k=1)

for r in result:
    print(r.page_content)
    print(r.metadata)


retriver =  vectorstore.as_retriever(
    
)

docs = retriver.invoke("Explain deep learning")

for d in docs:
    print(d.page_content)