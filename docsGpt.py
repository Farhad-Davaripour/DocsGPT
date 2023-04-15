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
library_names = ['langchain', 'openai', 'PyPDF2', 'tiktoken', 'faiss-cpu']

# Dynamically import libraries from list
for name in library_names:
    try:
        __import__(name)
    except ImportError:
        print(f"{name} not found. Installing {name}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', name])



from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# print("You need OpenAI token: Here is the link to get the keys: https://platform.openai.com/account/billing/overview")

from getpass import getpass

token = getpass("Enter your OpenAI token: ()")

import os
os.environ["OPENAI_API_KEY"] = str(token)


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


from google.colab import files
import os
import shutil



def upload_file():
  uploaded = files.upload()

  drive_path = '/content/drive/MyDrive'
  folder_name = 'git_repo'
  folder_path = os.path.join(drive_path, folder_name)

  for filename, data in uploaded.items():
      with open(filename, 'wb') as f:
          f.write(data)
      shutil.copy(filename, folder_path + "/")
      root_file = folder_path + "/" + filename
      os.remove(filename)

  return root_file



def run_conversation():
  # location of the pdf file/files.
  reader = PdfReader(str(upload_file()))

  count = 0
  while True:
      query = input("Question ", count + 1, " Ask your question: ")
      print(run_query(query, reader))
      count += 1
      if count == 10:
          print("### You asked 10 questions, run cell to ask more questions! ###")
          break
