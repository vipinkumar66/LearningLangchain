import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm import LLMModel
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st


llmmodel = LLMModel()
azure_model = llmmodel.azure_llm_model()
azure_embeddings = llmmodel.create_embeddings()

st.title("Conversational PDF chatboat with chat history")
st.write("Upload your pdf and ask question here")

session_id = st.text_input("Session id", value="default_session")
if "store" not in st.session_state:
    st.session_state.store = {}

uploaded_file = st.file_uploader(label="Choose a pdf file", type="pdf",
                                 accept_multiple_files=False)

documents = []
if uploaded_file:
    temppdf = "./temp.pdf"
    with open (temppdf, "wb") as file:
        file.write(uploaded_file.getvalue())
        filename = uploaded_file.name

    loader = PyPDFLoader(temppdf)
    pdf_docs = loader.load()
    documents.extend(pdf_docs)

if documents:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splitted_docs = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(splitted_docs, azure_embeddings)
    vector_store_retriever = vector_store.as_retriever()

    contextualize_q_system_prompt = (
        "Given a chat history and latest user questions "
        "which might reference questions in the chat history "
        "formulate a standalone question that can be understood "
        "without the chat history. Do not answer question just "
        "reformulate it if needed otherwise return it as it is."
    )

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    history_awre_retriever = create_history_aware_retriever(azure_model,
                                                            vector_store_retriever, contextualize_prompt)
    system_prompt = (
        "You are an assistant for the question answering tasks. Use"
        " the following pieces of retrieved context to answer the "
        "question, if you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    doc_chain = create_stuff_documents_chain(azure_model, qa_prompt)
    retrieval_chain = create_retrieval_chain(history_awre_retriever, doc_chain)

    def get_session_history(session:str)->BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    conversational_rag = RunnableWithMessageHistory(
        retrieval_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    user_input = st.text_input("Enter your question")
    if user_input:
        session_histiry = get_session_history(session_id)
        response = conversational_rag.invoke(
            {"input":user_input},
            config={
                "configurable":{"session_id":session_id}
            }
        )
        st.write(st.session_state.store)
        st.write("Assistant: ", response["answer"])
        st.write("chat history: ", session_histiry.messages)