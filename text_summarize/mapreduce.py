import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from llm import LLMModel

llmmodel = LLMModel().azure_llm_model()

filepath = r"text_summarize\speech.pdf"
loader = PyPDFLoader(filepath)
docs = loader.load()
finaldocs = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200).split_documents(docs)

chunks_prompt = """
    Write a short and consice summary for the given speech.
    Speech : '{text}'
    Summary:
"""
final_prompt = """
    Hey write a consice summary using the given points.
    Give this summary a good title and start the summary with an introduction, also include
    the given points in summary as number points.
    Speech: {text}
"""

map_prompt_template = PromptTemplate(
    input_variables=["text"],
    template=chunks_prompt
)

final_prompt_template = PromptTemplate(
    input_variables=["text"],
    template=final_prompt
)

chain = load_summarize_chain(
    llm=llmmodel,
    chain_type="map_reduce",
    map_prompt= map_prompt_template,
    combine_prompt = final_prompt_template
)
output_summary = chain.run(finaldocs)
print(output_summary)


