from langchain_text_splitters import (HTMLHeaderTextSplitter,
                                      RecursiveJsonSplitter)
import requests

"""
Here we will be talking about the html header splitter that is used to split
on the basis of the headers or you can say classes of the html content
"""
headers_to_split_on = [
    ("h1", "Main Heading"),
    ("h2", "Sub Heading")
]
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on)
html_header_splits = html_splitter.split_text_from_url("https://lilianweng.github.io/posts/2023-06-23-agent/")

json_response = requests.get("https://jsonplaceholder.typicode.com/posts").json()
json_splitter = RecursiveJsonSplitter(max_chunk_size=300)
json_chunks = json_splitter.split_json(json_response)
print(json_chunks)
