#  Load the pdf 
#  Splite in the chunks 
# Create the embeddings
# Store embeddings in data base 

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()

loader =  PyPDFLoader("./documentloaders/dataScienceNotes.pdf")
docs =  loader.load()

splitter =RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunk = splitter.split_documents(docs)

embedding_model =  MistralAIEmbeddings()

vectorstore =  Chroma.from_documents(
    documents=chunk,
    embedding=embedding_model,
    persist_directory="chroma_db"
)
