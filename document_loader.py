from langchain_community.document_loaders import (TextLoader, PyPDFLoader,
                                                  WebBaseLoader, ArxivLoader)
from constants import TEXT_FILEPATH, PDF_FILEPATH
import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter

# loading the files as documents for the text
text_loader = TextLoader(TEXT_FILEPATH, encoding="utf-8")
text_loader_docs = text_loader.load()


# LOADING THE PDF FILE AS DOCUMENTS
pdf_loader = PyPDFLoader(PDF_FILEPATH)
pdf_loader_docs = pdf_loader.load()


# LOADING THE WEB BASED PAGES AND SOME SPECIFIC CONTENT AS DOCUMENTS
web_based_loader = WebBaseLoader(web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
                                 bs_kwargs=dict(parse_only = bs4.SoupStrainer(
                                     class_=("post-title","post-header","post-content")
                                 )))
web_based_loader_docs = web_based_loader.load()


# THERE IS A WEBSITE WITH THE RESEARCH PAPERS AND WE DIRECTLY WANT TO
# WORK WITH IT THE LANGCHAIN HAS A LOADER FOR IT
arxiv_loader= ArxivLoader(
    query="1706.03762", load_max_docs=2
).load()

# TEXT SPLITTING TECHNIQUES
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
final_docs = text_splitter.split_documents(pdf_loader_docs)