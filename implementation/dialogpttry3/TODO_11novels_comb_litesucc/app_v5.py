from flask import Flask, request, jsonify, render_template
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

app = Flask(__name__)

# 設定環境變量和檔案路徑
novel_folder_path = './novelsascii'
hf_token_path = "./hf_token.txt"
open_ai_key_path = "./open-ai-key.txt"

# 讀取 Hugging Face token 和 OpenAI key
with open(hf_token_path, "r") as file:
    hf_token = file.readline().strip()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

with open(open_ai_key_path, "r") as file:
    open_ai_key = file.readline().strip()

# 初始化 LLM 和 Chroma
llm = OpenAI(openai_api_key=open_ai_key)
embeddings_model = HuggingFaceEmbeddings()
# persistent_directory = './chroma_db'
# vectordb = Chroma(embedding=embeddings_model, persist_directory=persistent_directory)
vectordb = Chroma(persist_directory='./chroma_db')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    input_text = data['message']

    # 使用 Langchain LLM 進行問答
    def get_answer(query):
        new_line = '\n'
        template = f"Use the following pieces of context to answer truthfully.{new_line}If the context does not provide the truthful answer, make the answer as truthful as possible.{new_line}Use 15 words maximum. Keep the response as concise as possible.{new_line}{{context}}{new_line}Question: {{question}}{new_line}Response: "
        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever(), return_source_documents=True, chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
        result = qa_chain({"query": query})
        return result["result"]

    response = get_answer(input_text)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
