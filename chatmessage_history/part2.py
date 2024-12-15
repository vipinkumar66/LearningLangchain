"""
    In this i am using chat prompt templates,
    Creating a chain
    Using multiple inputs in thet prompt template and also
    telling the runnables how to get the correct key for the
    session id to store the history of messages
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm import LLMModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


llmmodel = LLMModel()
azure_model = llmmodel.azure_llm_model()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are an AI engineer so ask all questions as per your best knowledge in {language}"),
        MessagesPlaceholder(variable_name="messages")
    ]
)

chain = prompt|azure_model
store = {}
def get_session_history(session_id:str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

with_history_runnables = RunnableWithMessageHistory(
    chain, get_session_history, input_messages_key="messages"
    )
config1 = {"configurable":{"session_id":"chat1"}}
response1 = with_history_runnables.invoke(
    {"messages":[HumanMessage(content="Hey my name is Vipin")], "language":"spanish"},
    config=config1
)
print(response1.content)