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
uploaded = files.upload()


import os
import shutil
for filename, data in uploaded.items():
    with open(filename, 'wb') as f:
        f.write(data)
    shutil.copy(filename, folder_path + "/")
    root_file = folder_path + "/" + filename
    os.remove(filename)


# location of the pdf file/files.
reader = PdfReader(str(root_file))


# define a function to get user input and validate their choice
def get_user_input(choices, prompt):
    """
    Get user input and validate their choice.
    Args:
        choices (list): A list of possible choices.
        prompt (str): The prompt to display to the user.
    Returns:
        str: The user's choice.
    """
    while True:
        user_input = input(prompt)
        if user_input in choices:
            return user_input
        else:
            print("Invalid choice. Please try again.")

# define the available choices
choices = ['Yes', 'No']

# define the prompt for the user
prompt = "Do you stil have queestion from PDF?: "

# get user input
user_choice = get_user_input(choices, prompt)

# display the user's choice
print("You chose option", user_choice)




count = 0
while True:
    query = input("Ask your question: ")
    print(run_query(query, reader))
    count += 1
    if count == 10:
        print("You asked 10 questions, run cell to ask more questions")
        break
