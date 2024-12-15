import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from llm import LLMModel
import streamlit as st
import validators
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from fake_useragent import UserAgent

useragent = UserAgent().random
llmmodel = LLMModel().azure_llm_model()

prompt_template = """
    Provide a summary of the given content with in 300 words.
    {text}
"""
prompt = PromptTemplate(input_variables=["text"], template=prompt_template)

st.set_page_config(page_title="Summarize URL")
st.title("Summarize your URL")
st.subheader("Enter URL")

url = st.text_input("URL", label_visibility="collapsed")
if st.button("Summarize the content from the Yt or Website"):
    if not url.strip():
        st.error('Please provide the url to start the app')
    elif not validators.url(url):
        st.error("Please provide a valid URL")
    else:
        try:
            with st.spinner("Waiting..."):
                if "youtube.com" in url:
                    loader = YoutubeLoader.from_youtube_url(url)
                else:
                    loader = UnstructuredURLLoader(urls=[url], ssl_verify=False, headers={"User-Agent":useragent})
                docs = loader.load()
                chain = load_summarize_chain(llm=llmmodel, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)
                st.success(output_summary)
        except Exception as e:
            st.exception(e)


