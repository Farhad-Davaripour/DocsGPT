# docsGpt.py - Contains the docsGpt functions and classes for document parsing
# Author: Armin Norouzi, Farhad Davaripour 
# Contact: https://github.com/Farhad-Davaripour/DocsGPT
# Date created: April 14, 2023
# Last modified: April 14, 2023
# License: MIT License

# Import required modules
import sys
import subprocess

# List of library names to import
library_names = ['langchain', 'openai', 'PyPDF2', 'tiktoken']

# Dynamically import libraries from list
for name in library_names:
    try:
        __import__(name)
    except ImportError:
        print(f"{name} not found. Installing {name}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', name])
        __import__(name)


from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS





from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()
chain = load_qa_chain(OpenAI(), chain_type="stuff")


def upload_doc(reader):

    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    # retreival we don't hit the token size limits. 

    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
    )

    texts = text_splitter.split_text(raw_text)


    docsearch = FAISS.from_texts(texts, embeddings)
    return docsearch


def run_query(query, file):

    docsearch = upload_doc(file)
    docs = docsearch.similarity_search(query)
    return chain.run(input_documents=docs, question=query)
