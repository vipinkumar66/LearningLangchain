import sys
import os
from langchain_core.prompts import ChatPromptTemplate
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm import LLMModel

azurechatopenai = LLMModel().azure_llm_model()
prompt = ChatPromptTemplate.from_messages(
    [
        ('system','You are an AI engineer, answers all the questions which are asked to you'),
        ('user','{input}')
    ]
)

chain = prompt|azurechatopenai
response= chain.invoke({"input":"What is langsmith?"})
print(response.content)

