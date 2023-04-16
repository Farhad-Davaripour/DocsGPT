# docsGpt.py - Contains the docsGpt functions and classes for document parsing
# Author: Armin Norouzi, Farhad Davaripour 
# Contact: https://github.com/Farhad-Davaripour/DocsGPT
# Date created: April 14, 2023
# Last modified: April 14, 2023
# License: MIT License

# Import required modules
import sys
import subprocess
from google.colab import files
import os
import shutil
import time


# List of library names to import
library_names = ['langchain', 'openai', 'PyPDF2', 'tiktoken', 'faiss-cpu']

# Dynamically import libraries from list
for name in library_names:
    try:
        __import__(name)
    except ImportError:
        print(f"{name} not found. Installing {name}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', name])


# Import required modules
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from getpass import getpass
import os

# adding token
# print("You need OpenAI token: Here is the link to get 
# the keys: https://platform.openai.com/account/billing/overview")

token = getpass("Enter your OpenAI token: ()")
os.environ["OPENAI_API_KEY"] = str(token)


# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()
chain = load_qa_chain(OpenAI(), chain_type="stuff")


def upload_doc(reader):
    """
    Extracts text from a PDFReader object and splits it into 
    chunks using the CharacterTextSplitter class. 
    Computes embeddings for each chunk using an embeddings 
    model and indexes them using the FAISS library. 

    Args:
    - reader: A PDFReader object containing the PDF file to 
    be processed.

    Returns:
    - A FAISS index object containing the embeddings of the 
    text chunks.
    """
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
    """
    Runs a query on a PDF file using the docsearch and chain 
    libraries. 

    Args:
    - query: A string representing the query to be run.
    - file: A PDFReader object containing the PDF file to be 
    searched.

    Returns:
    - A string containing the output of the chain library run 
    on the documents returned by the docsearch similarity search.
    """
        
    docsearch = upload_doc(file)
    docs = docsearch.similarity_search(query)
    return chain.run(input_documents=docs, question=query)


def upload_file(folder_path):
    """
    Uploads a file from the local file system and saves it to 
    a folder path. 

    Args:
    - folder_path: A string representing the folder path where 
    the file will be saved.

    Returns:
    - A string representing the path of the uploaded file.
    """
    
    uploaded = files.upload()

    for filename, data in uploaded.items():
        with open(filename, 'wb') as f:
            f.write(data)
        shutil.copy(filename, folder_path + "/")
        root_file = folder_path + "/" + filename
        os.remove(filename)

    return root_file


def run_conversation(folder_path):
    """
    Initiates a conversation with the user by repeatedly asking for 
    input queries and running them on a PDF file. 

    Args:
    - folder_path: A string representing the folder path where the 
    PDF file is located.

    Returns:
    - Run conversation based on PDF
    """
      
    # location of the pdf file/files.
    reader = PdfReader(str(upload_file(folder_path)))

    count = 0
    while True:
        print("Question ", count + 1)
        start_time = time.time()
        query = input(" Ask your question or if you have no further question type stop:\n ")
        if query != "stop":
            print("Answer:\n", run_query(query, reader))
            count += 1
        else:
            print("### Thanks for using the app! ###")
            break

        if (time.time() - start_time) > 300:  # check if 5 minutes have passed
            print("### You have not asked a question in 5 minutes. Exiting the program. ###")
            break
