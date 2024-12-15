"""
We really need to manage the chat history properly,
if left unmanaged then most probably the history is going to
get bundled and are context is going to be full
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from llm import LLMModel

llmmodel = LLMModel()
azure_model = llmmodel.azure_llm_model()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are an AI assistant answer questions according to best of your knowledge in {language}"),
        MessagesPlaceholder(variable_name="messages")
    ]
)
trimmer = trim_messages(
    max_tokens = 30,
    token_counter=azure_model,
    strategy="last",
    allow_partial = False,
    include_system=True,
    start_on="human"
)

messages = [
    HumanMessage(content="Hey my name is vipin kumar"),
    AIMessage(content="Nice to meet you vipin"),
    HumanMessage(content="I am a web developer"),
    AIMessage(content="Nice to hear that may i know which new technologies you are working?"),
    HumanMessage(content="I like chocolate ice cream"),
    AIMessage(content="That is nice"),
]

chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages")|trimmer)
    | prompt
    | azure_model
)

store = {}
def get_message_history(session_id:str)-> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(
    chain,
    get_message_history,
    input_messages_key="messages"
)

config1 = {"configurable":{"session_id":"chat6"}}
response1 = with_message_history.invoke({
    "messages":messages+[HumanMessage(content="What is my fav icecream?")],
    "language":"english"
}, config=config1)
print(response1.content)