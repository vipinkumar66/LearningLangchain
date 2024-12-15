import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm import LLMModel
from langchain.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import streamlit as st

llmmodel = LLMModel()
azure_model = llmmodel.azure_llm_model()
azure_embeddings = llmmodel.create_embeddings()

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
        {context}
    </context>
    Question: {input}
    """
)

def create_vector_embeddings():
    if "vector" not in st.session_state:
        st.session_state.embeddings = azure_embeddings
        st.session_state.loader = PyPDFDirectoryLoader("resources")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                                        chunk_overlap=200)
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs,
                                                        st.session_state.embeddings)

st.title("CHATBOT USING RAG (Retrieval Augmented Generation)")
if st.button("Create Document Embeddings"):
    create_vector_embeddings()
    st.write("Vector DB is ready")

user_prompt = st.text_input("Ask your question to chatboat")
if user_prompt:
    document_chain = create_stuff_documents_chain(azure_model,prompt)
    vector_retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(vector_retriever, document_chain)
    response = retrieval_chain.invoke({"input":user_prompt})
    if response:
        st.write(response["answer"])
        with st.expander("Similar Documents"):
            for i,docs in enumerate(response["context"]):
                st.write(docs.page_content)
                st.write("--------------------------------------------------")
