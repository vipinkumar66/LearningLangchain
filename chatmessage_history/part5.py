import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
from llm import LLMModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

llmodel = LLMModel()
azure_model = llmodel.azure_llm_model()
azure_embeddings = llmodel.create_embeddings()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant, please answer to user questions"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user","Question: {question}")
])
chat_history = ChatMessageHistory()
output_parser = StrOutputParser()
chain = prompt|azure_model|output_parser

def generate_response(question):
    chat_history.add_message({"role": "user", "content": question})
    chain_with_message_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: chat_history,
        input_messages_key="question",
        history_messages_key="chat_history"
    )
    # chain = prompt|azure_model|output_parser
    answer = chain_with_message_history.invoke(
        {"question":question},
        {"configurable":{"session_id":"chat1"}}
        )
    chat_history.add_message({"role": "assistant", "content": answer})
    return answer

def start_streamlit():
    st.write("Go ahead and ask your question")
    user_input = st.text_input("You: ")
    if user_input:
        answer = generate_response(user_input)
        st.write(answer)
    else:
        st.write("Ask some question")

start_streamlit()


