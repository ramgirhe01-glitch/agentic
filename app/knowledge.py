from functools import lru_cache

from langchain_ollama import OllamaEmbeddings
from pinecone import Pinecone,ServerlessSpec
import json
from app.config import settings

@lru_cache()
def get_embedding_and_pinecone():
    pinecone = Pinecone(api_key=settings.pinecone_api_key)
    embeddings = OllamaEmbeddings(model="nomic-embed-text",base_url="http://129.103.33.38:11434")

    return pinecone, embeddings


def get_index():
    pinecone, embeddings = get_embedding_and_pinecone()
    if(not pinecone.has_index(settings.pinecone_index_name)):
        print(f"Index {settings.pinecone_index_name} already does not exist.")
        pinecone.create_index(
            name=settings.pinecone_index_name,
            dimension=768,
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pinecone.Index(settings.pinecone_index_name)
    return index
                                  
def get_embeddings():
    _, embeddings = get_embedding_and_pinecone()
    return embeddings

def embed_documents(texts: list[str]):
    embeddings = get_embeddings()
    vectors = embeddings.embed_documents(texts)
    return vectors

def embed_query(query: str):
    embeddings = get_embeddings()
    single_vector = embeddings.embed_query(query)
    return single_vector

def search_index(query: str, top_k: int = 5):
    index = get_index()
    embeddings = get_embeddings()
    single_vector = embeddings.embed_query(query)
    result = index.query(
        vector=single_vector, top_k=top_k, include_metadata=True
    )
    return result

print(settings.pinecone_index_name)

# # pinecone.create_index(
# #     name=settings.pinecone_index_name,
# #     dimension=768,
# #     spec=ServerlessSpec(cloud="aws", region="us-east-1")
# # )
# index = pinecone.Index(settings.pinecone_index_name)

# embeddings = 
# single_vector = embeddings.embed_query("Hi i am ram")
# print(single_vector)  # Show the first 100 characters of the vector
# # index.upsert(vectors=[("2", single_vector, {"text": "Hi i am ram"})])
# result = index.query(
#     vector=single_vector,top_k=5, include_metadata=True)

# print(result)  # Show the first 100 characters of the result

# # pinecone.delete_index(name=settings.pinecone_index_name)