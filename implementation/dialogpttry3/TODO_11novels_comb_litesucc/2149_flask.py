import os
from langchain.vectorstores import Chroma as LangChainChroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from flask import Flask, request, jsonify, render_template

# 設定環境變量和檔案路徑
open_ai_key_path = "./open-ai-key.txt"
with open(open_ai_key_path, "r") as file:
    open_ai_key = file.readline().strip()

# Define the embedding function
embedding_function = OpenAIEmbeddings(openai_api_key=open_ai_key, model="text-embedding-ada-002")  # Ensure consistent model

# Load the Chroma database from the specified path with consistent embedding function
def load_chroma_db(path):
    return LangChainChroma(persist_directory=path, embedding_function=embedding_function)

# Load the Chroma database
vectordb = load_chroma_db('./chroma_db')

app = Flask(__name__)

@app.route('/')
def index():
    # Load the index HTML file
    return render_template('index.html')

def get_answer(query):
    new_line = '\n'
    template = f"Use the following pieces of context to answer truthfully.{new_line}If the context does not provide the truthful answer, make the answer as truthful as possible.{new_line}Use 15 words maximum. Keep the response as concise as possible.{new_line}{{context}}{new_line}Question: {{question}}{new_line}Response: "
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

    llm = OpenAI(openai_api_key=open_ai_key)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever(), return_source_documents=True, chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

    # Debug: Check the retrieved documents
    retrieved_docs = vectordb.similarity_search(query, k=3)
    print("Retrieved Documents:", retrieved_docs)

    result = qa_chain({"query": query})
    return result["result"]

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    query = data['query']
    answer = get_answer(query)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)
