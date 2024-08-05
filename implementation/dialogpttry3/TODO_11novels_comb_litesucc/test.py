import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# 设置环境变量和文件路径
hf_token_path = "./hf_token.txt"
open_ai_key_path = "./open-ai-key.txt"

# 读取 Hugging Face token 和 OpenAI key
with open(hf_token_path, "r") as file:
    hf_token = file.readline().strip()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

with open(open_ai_key_path, "r") as file:
    open_ai_key = file.readline().strip()

# 初始化 LLM 和 Chroma



embeddings_model = HuggingFaceEmbeddings()
persistent_directory = './chroma_db'

vectordb = Chroma.from_documents(
    documents=all_paras,
    embedding=embeddings_model,
    persist_directory=persistent_directory,
)

llm = OpenAI(openai_api_key=open_ai_key)
embeddings_model = HuggingFaceEmbeddings()
vectordb = Chroma(persist_directory='./chroma_db')

# def get_answer(query):
#     new_line = '\n'
#     template = f"Use the following pieces of context to answer truthfully.{new_line}If the context does not provide the truthful answer, make the answer as truthful as possible.{new_line}Use 15 words maximum. Keep the response as concise as possible.{new_line}{{context}}{new_line}Question: {{question}}{new_line}Response: "
#     QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
#     qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever(), return_source_documents=True, chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
#     result = qa_chain({"query": query})
#     return result["result"]

# # 终端交互循环
# while True:
#     input_text = input("Please enter your question (type 'exit' to quit): ")
#     if input_text.lower() == 'exit':
#         break
#     response = get_answer(input_text)
#     print("Answer:", response)





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


# llm = OpenAI(openai_api_key=open_ai_key)

# def get_answer(query):
#     new_line = '\n'
#     template = f"Use the following pieces of context to answer truthfully.{new_line}If the context does not provide the truthful answer, make the answer as truthful as possible.{new_line}Use 15 words maximum. Keep the response as concise as possible.{new_line}{{context}}{new_line}Question: {{question}}{new_line}Response: "
#     QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

#     from langchain.chains import RetrievalQA
#     question = query
#     qa_chain = RetrievalQA.from_chain_type(llm,
#                                           retriever=vectordb.as_retriever(),
#                                           return_source_documents=True,
#                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})


#     result = qa_chain({"query": question})
#     return result["result"]

