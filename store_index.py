import os

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from src.helpers import *
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=API_KEY)

index_name = "medical-chatbot-llama2"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

embeddings = download_hugging_face_embeddings()

vector_store = PineconeVectorStore(
    index=pc.Index(index_name),
    embedding=embeddings,
)
