import os

from src.prompt import *
from src.helper import *

from flask import Flask, render_template, jsonify, request

from langchain.llms import CTransformers
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain import PromptTemplate
from langchain.chains import RetrievalQA

from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()

API_KEY = os.getenv("PINECONE_API_KEY")

embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=API_KEY)

index_name = "medical-chatbot-llama2"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

vector_store = PineconeVectorStore(
    index=pc.Index(index_name),
    embedding=embeddings,
)

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

llm = CTransformers(
    model="../model/llama-2-7b-chat.ggmlv3.q2_K.bin",
    model_type="llama",
    config={"max_new_tokens": 1024, "temperature": 0.8},
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=False,
)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
