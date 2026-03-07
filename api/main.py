from fastapi import FastAPI
from pydantic import BaseModel

import numpy as np
import faiss
import joblib

from src.embedding_model import EmbeddingModel
from src.semantic_cache import SemanticCache
from src.data_loader import load_dataset

# Initialize FastAPI
app = FastAPI()


# Load embedding model
print("Loading embedding model...")
embedding_model = EmbeddingModel()


# Load FAISS vector database
print("Loading vector database...")
index = faiss.read_index("models/faiss_index.bin")


# Load clustering model
print("Loading clustering model...")
cluster_model = joblib.load("models/gmm_model.pkl")

print("Loading documents...")
documents, labels = load_dataset()

print("All models loaded successfully!")


# Initialize semantic cache
cache = SemanticCache()


# Request format
class QueryRequest(BaseModel):
    query: str


# -----------------------------
# QUERY API
# -----------------------------
@app.post("/query")
def query_endpoint(request: QueryRequest):

    query = request.query

    # Convert query to embedding
    embedding = embedding_model.encode_query(query)

    # Check semantic cache
    hit, match, score = cache.lookup(embedding)

    if hit:

        return {
            "query": query,
            "cache_hit": True,
            "matched_query": match["query"],
            "similarity_score": float(score),
            "result": match["result"],
            "dominant_cluster": match["cluster"]
        }

    # If cache miss → perform vector search
    query_vec = np.array([embedding]).astype("float32")

    distances, indices = index.search(query_vec, 5)

    top_index = indices[0][0]
    result = documents[top_index][:500]

    # Determine cluster
    cluster_probs = cluster_model.predict_proba([embedding])[0]

    dominant_cluster = int(np.argmax(cluster_probs))

    # Store in cache
    cache.add(query, embedding, result, dominant_cluster)

    return {
        "query": query,
        "cache_hit": False,
        "result": result,
        "dominant_cluster": dominant_cluster
    }


# -----------------------------
# CACHE STATS API
# -----------------------------
@app.get("/cache/stats")
def cache_stats():

    return cache.stats()


# -----------------------------
# CLEAR CACHE API
# -----------------------------
@app.delete("/cache")
def clear_cache():

    cache.clear()

    return {"message": "Cache cleared successfully"}