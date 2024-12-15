import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm import LLMModel

llmmodel = LLMModel()
azure_model = llmmodel.azure_llm_model()
azure_embeddings = llmmodel.create_embeddings()

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain import hub

wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=200)
arxiv_wrapper = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=200)
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

prompt = hub.pull("hwchase17/openai-tools-agent")

tools = [wikipedia_tool,arxiv_tool]
agent = create_openai_tools_agent(azure_model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
response = agent_executor.invoke({"input":"What are the transformers in machine learning?"
                       " And give me the best research paper number to learn about that"})
print(response)
