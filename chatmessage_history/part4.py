import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm import LLMModel
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage


llmmodel = LLMModel()
azure_model = llmmodel.azure_llm_model()
azure_embeddings = llmmodel.create_embeddings()

# def load_page_and_create_embeddings():
if os.path.exists("faiss"):
    vectordb = FAISS.load_local("faiss", azure_embeddings,
                                allow_dangerous_deserialization=True)
else:
    print("Faiss db not in local")
    web_docs = WebBaseLoader(
        "https://blog.cloudflare.com/using-apache-kafka-to-process-1-trillion-messages/"
        ).load()
    chunks_of_doc = RecursiveCharacterTextSplitter(chunk_size=500,
                                                    chunk_overlap=30).split_documents(web_docs)
    vectordb = FAISS.from_documents(chunks_of_doc,azure_embeddings)
    vectordb.save_local("faiss")


system_prompt = (
    "Given a chat history and latest questions which might "
    "reference context in the chat history, formulate a standalone "
    "question which can be understood without the chat history. "
    "Do not answer question, just reformulate it if needed otherwise "
    "return it as it is."
)
prompt = ChatPromptTemplate.from_messages([
    ("system",system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human","{input}")
])
vector_retriever = vectordb.as_retriever()
# But now we need a retriever that remebers the history
history_aware_retriever = create_history_aware_retriever(azure_model, vector_retriever,prompt)

# system_prompt = ("You are an AI assistant for question-answering task"
#                  " Use the following piece of retrieved context to answer "
#                  "the question. If you don't know the answer, say that you "
#                  "don't know. Use three sentence maximum and keep the answer"
#                  "concise"
#                  "\n\n"
#                  "{context}"
#                 )
"""
These steps are for normal chat app without history
"""
# question_answer_chain = create_stuff_documents_chain(azure_model, prompt)
# rag_chain = create_retrieval_chain(vector_retriever, question_answer_chain)
# response1 = rag_chain.invoke({"input":"What is the role of connector in handling the 1 trillion of messages?"})
# print(response1["answer"])

"""
Let's discuss the chat app with history version
"""
qa_prompt = ChatPromptTemplate.from_messages([
    ("system",system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "Context: {context}\n\n{input}")
])

question_answer_chain = create_stuff_documents_chain(azure_model,qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
chat_history = []
response1 = rag_chain.invoke({"input":"What is the role of connector in handling the 1 trillion of messages tell me in 30 words?",
                              "chat_history":chat_history})
chat_history.extend(
    [
        HumanMessage(content="What is the role of connector in handling the 1 trillion of messages?"),
        AIMessage(content=response1["answer"])
    ]
)
question2 = "Tell me in breif about it"
response2 = rag_chain.invoke({"input":question2, "chat_history":chat_history})
print(response2["answer"])