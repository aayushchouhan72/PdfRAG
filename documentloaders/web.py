from langchain_community.document_loaders import WebBaseLoader

url = "https://medium.com/@raja.gupta20/generative-ai-for-beginners-part-1-introduction-to-ai-eadb5a71f07d"

data =  WebBaseLoader(url)

docs =  data.load()

print(docs[0].page_content)