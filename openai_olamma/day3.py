"""
    LCEL => Langchain Expression Language
    This is basically used to chain components together, so in this
    instead of telling how to do we tell what to do allowing langchain
    to optimize run time executions of chain

    In this we are making use of Groq-API
    Why this is different from others is that: Its used LPU which is faster then
    GPU and CPU in terms of.
    LPU (Learning processing unit), they are the hardware that are used for the
    machine learning and deep learning,
    => They are part of GPU and TPU, but they are made for inferencing and learning

"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def create_load_model():
    groq_api = os.getenv("GROQ_API_KEY")
    model = ChatGroq(model="Gemma2-9b-It", api_key=groq_api)
    parser = StrOutputParser()
    messages = [
        SystemMessage(content="Give the answer of the question asked by the user."),
        HumanMessage(content="What is the purpose of life according to Bhagwad Gita?")
    ]
    generic_template = "Translate the following into {language}"
    prompt = ChatPromptTemplate.from_template(
        """
        Translate the following into {language}
        <context>
            {text}
        </context>
        """
    )
    # prompt = ChatPromptTemplate.from_messages([
    #     ("system",generic_template),
    #     ("user","{text}")
    # ])
    chain = prompt|model|parser
    result = chain.invoke({"language":"Marathi", "text":"Hello how are you?"})
    print(result)

create_load_model()

