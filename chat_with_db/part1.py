import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sqlite3
from llm import LLMModel
from langchain.utilities import SQLDatabase
from langchain.agents import initialize_agent, AgentType, create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st
from sqlalchemy import create_engine
from pathlib import Path
from dotenv import load_dotenv
from urllib import parse

load_dotenv()

# BASIC MODEL AND EMBEDDINGS PART
llmmodel = LLMModel()
azure_model = llmmodel.azure_llm_model()
azure_embeddings = llmmodel.create_embeddings()

# STREAMLIT BASIC DESIGNING
st.set_page_config(page_title="Langchain: Chat with SQL Database")
st.title("Chat with SQL Database")

LOCAL_DB = "Use Local SQL lite Database"
MYSQL = "Use MYSQL"

radio_options = ["Use SQL lite 3 database", "Use Local MYSQL Database"]
selected_opt = st.sidebar.radio(label="Choose the db with which you want to chat", options=radio_options)

if radio_options.index(selected_opt)==1:
    db_uri = MYSQL
else:
    db_uri = LOCAL_DB

if not db_uri:
    st.info("Please select the database first")

# DATABASE CONNECTION PART
@st.cache_resource(ttl="2h")
def configure_db(db_uri):
    if db_uri == LOCAL_DB:
        dbpath = (Path(__file__).parent/"student.db").absolute()
        """
            The creator param in the create_engine, allows us to provide
            a custom connect behaviour, so as you can see we are opening
            our database in read only mode.
            Using a lambda allows the connection logic to be defined
            dynamically and executed only when a connection is actually
            needed, rather than at the time the engine is created.
        """
        creator = lambda: sqlite3.connect(f"file:{dbpath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    elif db_uri == MYSQL:
        password_ = parse.quote_plus(os.environ.get("MYSQL_PASSWORD"))
        password = password_
        user = os.getenv('MYSQL_USER')
        host = os.getenv('MYSQL_HOST')
        database = os.getenv('MYSQL_DATABASE')
        return SQLDatabase(create_engine(f'mysql+pymysql://{user}:{password}@{host}/{database}'))

db = configure_db(db_uri)


# CREATING THE TOOLKIT
toolkit = SQLDatabaseToolkit(
    db=db,
    llm=azure_model
)
agent = create_sql_agent(
    llm=azure_model,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

if "messages" not in st.session_state or st.sidebar.button("Clear Message History"):
    st.session_state.messages = [{"role":"assistant", "content":"Hi how can i help you"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_quey = st.text_input("Enter your question that you want to ask from database")
if user_quey:
    st.session_state.messages.append({"role":"user", "content":user_quey})
    st.chat_message("user").write(user_quey)

    with st.chat_message("assistant"):
        streamlitcallback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = agent.run(user_quey, callbacks=[streamlitcallback])
        st.session_state.messages.append({"role":"assistant", "content":response})
        st.write(response)
