"""
    In this we will learn how the message history works with
    respect to sessions, and the main component behind this
    is message history

    MESSAGE HISTORY CLASS, and we use this to wrap it around
    our model which will make it stateless and that can be used
    further to pass the context to the chain
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm import LLMModel
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
"""
The main difference in the ChatMessageHistory and BaseChatMessageHistory
lies in their implementations. The Base part allows us to cutomize how we
are going to, add, delete or retrieve the messages but the normal provides
an in memory implementation and stores the messages in list or some other
python data stucture and we can't add much modifications to it
"""

"""
Runnables => You can think of runnables as a small piece of code or task that
can be combined later to produce chains that can perform a task, we can pass
the input of one runnable to another runnable with the help of pipeline and this
is called chain.
"""

azure_api = LLMModel().azure_llm_model()
messages = [
    HumanMessage(content="Hey my name is Vipin and i am web developer"),
    AIMessage(content="Hello Vipin! It's great to meet you. As a web developer, you must work on a variety of interesting projects. Are there any particular technologies or frameworks you're currently focusing on, or is there anything specific you'd like to discuss or get help with?"),
    HumanMessage(content="Hey what is my name and what i do")
]

store = {}

def get_Session_id(session_id:str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

config = {"configurable":{"session_id":"chat1"}}
message_history_runnables = RunnableWithMessageHistory(azure_api, get_Session_id)

message_history_runnables.invoke(
    [HumanMessage(content="Hey my name is Vipin and i am web developer ")],
    config=config
)

response1 = message_history_runnables.invoke(
    [HumanMessage(content="Hey what is my name and what do i do? ")],
    config=config
)
"""
Your name is Vipin, and you are a web developer.
"""
config2  = {"configurable":{"session_id":"chat2"}}
response2 = message_history_runnables.invoke(
    [HumanMessage(content="Hey what is my name and what do i do? ")],
    config=config2
)
"""
    I'm sorry, but I don't have access to personal data about individuals unless it's shared with
    me in conversation. I can't know your name or what you do unless you tell me. How can I
    assist you today?
"""

