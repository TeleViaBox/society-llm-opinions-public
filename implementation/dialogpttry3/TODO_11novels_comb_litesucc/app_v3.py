from flask import Flask, request, jsonify, render_template
import os
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

app = Flask(__name__)

# Read OpenAI key and Hugging Face token
with open("./TODO_11novels_comb/open-ai-key.txt", "r") as file:
    open_ai_key = file.readline().strip()
os.environ["OPENAI_API_KEY"] = open_ai_key

with open("./TODO_11novels_comb/hf_token.txt", "r") as file:
    hf_token = file.readline().strip()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

# Initialize OpenAI and Chroma
llm = OpenAI(openai_api_key=open_ai_key)

# Specify the Chroma database path
persistent_directory = './TODO_11novels_comb/chroma_db'
vectordb = Chroma(persist_directory=persistent_directory)

# QA Template and Chain Setup
def get_answer(query):
    new_line = '\n'
    template = f"Use the following pieces of context to answer truthfully.{new_line}If the context does not provide the truthful answer, make the answer as truthful as possible.{new_line}Use 15 words maximum. Keep the response as concise as possible.{new_line}{{context}}{new_line}Question: {{question}}{new_line}Response: "
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=vectordb.as_retriever(),
                                           return_source_documents=True,
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    result = qa_chain({"query": query})
    return result["result"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    input_text = data['message']

    # Use LangChain and OpenAI for Q&A
    response = get_answer(input_text)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
