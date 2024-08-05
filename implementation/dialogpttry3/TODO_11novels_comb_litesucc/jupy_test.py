import os
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from flask import Flask, request, jsonify, render_template

# 設定環境變量和檔案路徑
open_ai_key_path = "./open-ai-key.txt"

with open(open_ai_key_path, "r") as file:
    open_ai_key = file.readline().strip()


class Chroma:
    @staticmethod
    def load(path):
        # Placeholder for actual loading logic
        print(f"Loading Chroma database from {path}")
        return Chroma()

# Now trying to load the Chroma database
vectordb = Chroma.load('./chroma_db')

# # 加載現有的Chroma資料庫
# vectordb = Chroma.load('./chroma_db')

def get_answer(query):
    new_line = '\n'
    template = f"Use the following pieces of context to answer truthfully.{new_line}If the context does not provide the truthful answer, make the answer as truthful as possible.{new_line}Use 15 words maximum. Keep the response as concise as possible.{new_line}{{context}}{new_line}Question: {{question}}{new_line}Response: "
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

    llm = OpenAI(openai_api_key=open_ai_key)
    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=vectordb,  # 直接使用 vectordb 作為檢索器
                                           return_source_documents=True,
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

    result = qa_chain({"query": query})
    return result["result"]


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
