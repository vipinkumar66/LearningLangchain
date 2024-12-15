import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm import LLMModel
import streamlit as st
from langchain.agents import AgentType, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper


llm = LLMModel()

azuremodel = llm.azure_llm_model()
azurechain = llm.create_embeddings()

# Creating the tools to use for the chatboat
arxiv_wraper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki_wraper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wraper)
wiki = WikipediaQueryRun(api_wrapper=wiki_wraper)
search = DuckDuckGoSearchRun(name="search")


# setting up the streamlit
st.title("Search APP with langchain")

# To maintain the chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role":"assistant", "content":"Hi I am a chatboat who can search the web. How can i help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt:=st.chat_input(placeholder="What is machine Learning?"):
    st.session_state.messages.append({"role":"user", "content":prompt})
    st.chat_message("user").write(prompt)

    tools = [search,arxiv, wiki]
    search_agent = initialize_agent(tools=tools, llm=azuremodel,
                                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                    handle_parsing_errors=True)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        print(response)
        st.session_state.messages.append({"role":"assistant", "content":response})
        st.write(response)




