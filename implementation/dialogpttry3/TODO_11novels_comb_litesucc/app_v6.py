# 第一步是提取兩個檔案中共有的部分，並進行合併。這包括環境變量設置、模型和資料庫初始化。

# 共用的導入語句和變數
common_imports = """
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
"""

# 共用的環境變量和檔案路徑設定
common_setup = """
novel_folder_path = './novelsascii'
hf_token_path = "./hf_token.txt"
open_ai_key_path = "./open-ai-key.txt"

with open(hf_token_path, "r") as file:
    hf_token = file.readline().strip()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

with open(open_ai_key_path, "r") as file:
    open_ai_key = file.readline().strip()
"""

# 生成合併後的共用部分
merged_common = common_imports + common_setup

# 顯示合併後的共用部分
merged_common



# 第二步是整合第一個檔案中的文本處理和問答功能到Flask應用中。

# 從第一個檔案中提取的文本處理和向量資料庫初始化代碼
text_processing_and_db_init = """
# 文本處理和向量資料庫初始化
all_paras = []
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separators=["\n\n", "\n", "(?<=\\. )", " "]
)

novel_files = [f for f in os.listdir(novel_folder_path) if f.endswith('.txt')]
for file_name in novel_files:
    file_path = os.path.join(novel_folder_path, file_name)
    loader = TextLoader(file_path)
    data = loader.load()
    paras = r_splitter.split_documents(data)
    all_paras.extend(paras)

vectordb = Chroma.from_documents(
    documents=all_paras,
    embedding=HuggingFaceEmbeddings(),
    persist_directory='./chroma_db',
)
"""

# 從第一個檔案中提取的問答函數
qa_function = """
def get_answer(query):
    new_line = '\\n'
    template = f"Use the following pieces of context to answer truthfully.{new_line}If the context does not provide the truthful answer, make the answer as truthful as possible.{new_line}Use 15 words maximum. Keep the response as concise as possible.{new_line}{{context}}{new_line}Question: {{question}}{new_line}Response: "
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template)

    llm = OpenAI(openai_api_key=open_ai_key)
    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=vectordb.as_retriever(),
                                           return_source_documents=True,
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

    result = qa_chain({"query": query})
    return result["result"]
"""

# 將這些代碼整合到 Flask 應用中
flask_app_integration = """
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    input_text = data['message']
    response = get_answer(input_text)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
"""

# 生成合併後的 Flask 應用代碼
merged_flask_app = merged_common + text_processing_and_db_init + qa_function + flask_app_integration

# 顯示合併後的 Flask 應用代碼
print(merged_flask_app)

