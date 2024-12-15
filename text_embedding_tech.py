"""
This section will teach us how to use the embedding techniques to convert the
text into vectors
"""
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from llm import LLMModel
from constants import TEXT_FILEPATH

azurekey = LLMModel()
azure_embeddings = azurekey.create_embeddings()

# OPENAI EMBEDDING TECH
# 1step is to load the documents and than split them into chunks
text_loader = TextLoader(TEXT_FILEPATH, encoding="utf-8")
text_documents = text_loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
final_docs = text_splitter.split_documents(text_documents)

# second step is to create the embedings of them
embeddings = [
    {
        "text":doc.page_content,
        "embedding":azure_embeddings.embed_query(doc.page_content)
    }
    for doc in final_docs
]

# Creating the database with the documents and the embeddings and then
# querying it to search in it
# chroma = Chroma.from_documents(final_docs, azure_embeddings, persist_directory="./chroma")
# to laod the data base
# db2 = Chroma(persist_directory="./chroma", embedding_function=azure_embeddings)
# query = "vipin kumar"
# retrieved_results = chroma.similarity_search(query)

"""
FAISS => Facebook AI Similarity Search

(1) In this we have a "Retriever" => which on the basis of the query gets the relevant docs
or you can say information. This is a part of RAG "Retrieval Augmented Generation", and
this is used in Q&A, Chatboats, etc
How does it work?
With the help of embedding models we convert the query and the documents to vector
embeddings, and then with the help of faiss which used the ANN, Brute force and other
methods to perform the similarity search.

(2) In this we also have search_with_score method and lower the score the better the result
is. And this is done using the: db.similarity_search_with_score(query).
"""
db = FAISS.from_documents(final_docs, azure_embeddings)
# To save the database locally
# and to load: loaded_db = FAISS.load_local("faiss_index", azure_embeddings)
db.save_local("faiss")
query = "What is the work experience of Vipin kumar and which all projects he has done?"
retriever = db.as_retriever()
data_ = retriever.invoke(query)
# for dat in data_:
    # print(dat.page_content)