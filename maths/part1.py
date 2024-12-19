import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
import streamlit as st
from langchain.chains import LLMChain, LLMMathChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

groq_api_key = os.getenv("GROQ_API_KEY")
language_model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# SETUP OF THE WIKIPEDIA TOOL
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="wikipedia tool",
    func=wikipedia_wrapper.run,
    description="A tool to search internet(wikipedia) about the given topic"
)

# SETUP OF MATHS CHAIN
math_chain = LLMMathChain.from_llm(llm=language_model)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool to handle the mathematical problems"
)

prompt = """
    You are a AI agent that answers mathematical problems logically.
    And give the answers of the problems in the points.
    Question: {question}
    Answer:
"""
prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)
chain = LLMChain(llm=language_model, prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning Tool",
    func=chain.run,
    description="Tool to solve the logical mathematical problems"
)

assistant_agent = initialize_agent(
    tools = [wikipedia_tool, calculator, reasoning_tool],
    llm=language_model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_erros=True
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role":"assistant", "content":"Hi i am a AI Agent how can i help you"}
    ]

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

question = st.text_input("Enter your question here")
if st.button("Solve Question"):
    if question:
        with st.spinner("Generating the response"):
            st.session_state.messages.append({"role":"user", "content":question})
            st.chat_message("user").write(question)
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = assistant_agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({"role":"assistant", "content":response})
            st.write("#####Response")
            st.success(response)

    else:
        st.error("Please enter your question first")
