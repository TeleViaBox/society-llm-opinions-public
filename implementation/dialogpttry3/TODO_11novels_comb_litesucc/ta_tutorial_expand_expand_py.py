#!/usr/bin/env python
# coding: utf-8
# ! pip install openai
# ! npm install openai@^4.0.0

# get_ipython().system('pip install transformers accelerate langchain==0.0.263 tiktoken huggingface_hub sentence_transformers chromadb==0.4.5 openai==0.27.8')

import os
import requests
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
     
try:
    with open("./hf_token.txt", "r") as file:
        hf_token = file.readline().strip()
    success = True
except FileNotFoundError:
    hf_token = ""
    success = False

print(success)


os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
loader = TextLoader("./novelsascii/1Adventures.txt")
data = loader.load()
data[0].page_content


r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separators=["\n\n", "\n", "(?<=\. )", " "]
)
     
paras = r_splitter.split_documents(data)
len(paras)

from pprint import pprint
# for para in paras:
#     pprint(para.page_content)
#     print("-"*100)
    

embeddings_model = HuggingFaceEmbeddings()


persistent_directory = './chroma_db'


from pprint import pprint
vectordb = Chroma.from_documents(
    documents = paras,
    embedding=embeddings_model,
    persist_directory=persistent_directory,
)


query = "Who Stole the Tarts"
docs = vectordb.similarity_search(query, k=3)
for rank, doc in enumerate(docs):
    print(f"Rank {rank+1}:")
    pprint(doc.page_content)
    print("\n")

try:
    with open("./open-ai-key.txt", "r") as file:
        open_ai_key = file.readline().strip()
    success = True
except FileNotFoundError:
    open_ai_key = ""
    success = False
print(success)


llm = OpenAI(openai_api_key=open_ai_key)

def get_answer(query):
    new_line = '\n'
    template = f"Use the following pieces of context to answer truthfully.{new_line}If the context does not provide the truthful answer, make the answer as truthful as possible.{new_line}Use 15 words maximum. Keep the response as concise as possible.{new_line}{{context}}{new_line}Question: {{question}}{new_line}Response: "
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

    from langchain.chains import RetrievalQA
    question = query
    qa_chain = RetrievalQA.from_chain_type(llm,
                                          retriever=vectordb.as_retriever(),
                                          return_source_documents=True,
                                          chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})


    result = qa_chain({"query": question})
    return result["result"]

query = "What caused Alice to start shrinking in size?"
result = get_answer(query)
print("Question:", query)
print("Answer:", result.strip())

# query = "Who Stole the Tarts"
# result = get_answer(query)
# print("Question:", query)
# print("Answer:", result.strip())


# query = "How does the Silent use her skills in battle?"
# result = get_answer(query)
# print("Question:", query)
# print("Answer:",result.strip())

# query = "What are the types of orbs used by Defect and their functions?"
# result = get_answer(query)
# print("Question:", query)
# print("Answer:",result.strip())

