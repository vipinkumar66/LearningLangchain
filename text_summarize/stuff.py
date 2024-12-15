import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from llm import LLMModel

llmmodel = LLMModel().azure_llm_model()

filepath = r"text_summarize\speech.pdf"
loader = PyPDFLoader(filepath)
docs = loader.load_and_split()

template = """
    Write a short and concise summary of the given speech.
    Speech : {text}
"""
prompt = PromptTemplate(
    input_variables=["text"],
    template=template
)

chain = load_summarize_chain(llm=llmmodel, chain_type="stuff", prompt=prompt, verbose=True)
output_summary = chain.run(docs)
print(output_summary)




