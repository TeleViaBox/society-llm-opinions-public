import os
import requests
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from pprint import pprint

# 設定環境變量和檔案路徑
novel_folder_path = './novelsascii'  # 設定資料夾路徑
hf_token_path = "./hf_token.txt"
open_ai_key_path = "./open-ai-key.txt"

try:
    with open(hf_token_path, "r") as file:
        hf_token = file.readline().strip()
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
    success = True
except FileNotFoundError:
    hf_token = ""
    success = False
print("Hugging Face token loaded:", success)

try:
    with open(open_ai_key_path, "r") as file:
        open_ai_key = file.readline().strip()
    success = True
except FileNotFoundError:
    open_ai_key = ""
    success = False
print("OpenAI key loaded:", success)

novel_files = [f for f in os.listdir(novel_folder_path) if f.endswith('.txt')]

all_paras = []
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separators=["\n\n", "\n", "(?<=\. )", " "]
)

for file_name in novel_files:
    file_path = os.path.join(novel_folder_path, file_name)
    loader = TextLoader(file_path)
    data = loader.load()
    paras = r_splitter.split_documents(data)
    all_paras.extend(paras)

print(f"Total paragraphs processed: {len(all_paras)}")

# 嵌入模型和向量資料庫初始化
embeddings_model = HuggingFaceEmbeddings()
persistent_directory = './chroma_db'

vectordb = Chroma.from_documents(
    documents=all_paras,
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

