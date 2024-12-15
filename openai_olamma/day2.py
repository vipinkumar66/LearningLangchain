"""
    Here we are building a DOCUMENT Q/A APP
    The main stuff to take from this code is about the document chain
    and create_stuff_documents_chain. The document chain is nothing but
    the sequence of operations that are applied on the documents to get the
    specific results.
    The create_stuff_documents_chain is nothing but it takes documents and
    then stuff them together into a single document that is used with the
    specified llm model to do the specified tasks

    Retriever => It is nothing but a pathway to get the data from the vectorstoredb,
    it is a kind of interface
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm import LLMModel
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from bs4 import BeautifulSoup
import streamlit as st

llm_model = LLMModel()
azure_openai_api = llm_model.azure_llm_model()
azure_openai_embeddings = llm_model.create_embeddings()

def load_data_and_process():
    # This step will load the website and do the scraping, convert it into docs
    websitedata = WebBaseLoader("https://blog.bytebytego.com/p/cloudflares-trillion-message-kafka")
    websitedata_docs = websitedata.load()

    # second step is to basically convert this to chunks as our models has context limits
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    websitedata_docs_spliited = text_splitter.split_documents(websitedata_docs)

    # third step is to convert this to vectors and then store them in a vector db
    vectorstoredb = FAISS.from_documents(websitedata_docs_spliited, azure_openai_embeddings)
    cwd = os.getcwd()
    os.chdir(cwd)
    faisspath = os.path.join(cwd, "faiss")
    vectorstoredb.save_local(faisspath)

def read_from_db(input_text):
    cwd = os.getcwd()
    os.chdir(cwd)
    loaded_db = FAISS.load_local("faiss", azure_openai_embeddings,
                                 allow_dangerous_deserialization=True)
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the following question based on the provided context
        <context>
            {context}
        </context>
        """
    )
    document_chain = create_stuff_documents_chain(azure_openai_api,prompt)
    vectordb_retriever = loaded_db.as_retriever()
    vectordb_retriever_chain = create_retrieval_chain(vectordb_retriever, document_chain)
    response = vectordb_retriever_chain.invoke({"input":input_text})
    return response["answer"]

def load_streamlit():
    st.title("Q/A app with the AZURE OPEN AI")
    input_text = st.text_input("Ask you question related to this website !!")
    if input_text:
        model_output = read_from_db(input_text)
        st.write(model_output)


if __name__ == "__main__":
    cwd = os.getcwd()
    os.chdir(cwd)
    if os.path.exists("faiss"):
        # if the vector db file already exists then we will read it and do the further process
        load_streamlit()
    else:
        # we will start everything from begining
        load_data_and_process()
        load_streamlit()



