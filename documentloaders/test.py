from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import   TokenTextSplitter
from langchain_text_splitters import  CharacterTextSplitter
# splitter =  CharacterTextSplitter(
#     chunk_size =10,
#     chunk_overlap=1

# )

# splitter = TokenTextSplitter(
#     chunk_size = 1000,
#     chunk_overlap=100
# )


loader = PyPDFLoader("documentloaders/dataScienceNotes.pdf")

docs = loader.load()
# print(docs,len(docs))

# chunk = splitter.split_documents(docs)

# print(len(chunk),chunk[0])


# chunk = splitter.split_documents(docs)

# for i  in chunk:
#     print(i.page_content)
#     print()
#     print()
#     print()
#     print()

# print(docs[0].page_content)