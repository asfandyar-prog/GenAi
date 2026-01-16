from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

loader = TextLoader("docs.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever()

query = "What are the key takeaways from the documents?"
retrieved_docs = retriever.get_relevant_documents(query)
retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

llm = ChatGroq(model="mixtral-8x7b-32768")

prompt = f"Based on the following text answer the question: {query}\n\n{retrieved_text}"
answer = llm.invoke(prompt)

print("Answer:", answer)
