# Semantic Search System with Fuzzy Clustering and Semantic Cache

## Project Overview

This project implements a lightweight **Semantic Search System** built using the **20 Newsgroups dataset**. The system combines **vector embeddings, fuzzy clustering, semantic caching, and a FastAPI service** to enable efficient natural language search.

The goal is to demonstrate how semantic similarity can be used to retrieve relevant documents and avoid redundant computation through a custom semantic cache.

---

## Key Features

• **Sentence Embeddings**

* Uses `sentence-transformers/all-MiniLM-L6-v2`
* Converts documents and queries into dense vector representations

• **Vector Database**

* FAISS is used for efficient similarity search over document embeddings

• **Fuzzy Clustering**

* Implemented using **Gaussian Mixture Model (GMM)**
* Each document belongs to multiple clusters with probability distribution

• **Semantic Cache**

* Custom cache implementation (no Redis or external caching)
* Detects semantically similar queries using cosine similarity
* Avoids recomputing results for similar queries

• **FastAPI Service**

* Provides REST API endpoints for querying the system
* Tracks cache performance statistics

---

## System Architecture

User Query
↓
Embedding Model (Sentence Transformers)
↓
Semantic Cache Lookup
↓
Cache Hit → Return Cached Result

Cache Miss →
↓
FAISS Vector Search
↓
Fuzzy Cluster Detection
↓
Return Top Document
↓
Store Result in Cache

---

## API Endpoints

### 1. Query Endpoint

POST `/query`

Request:

```
{
 "query": "space shuttle launch"
}
```

Response:

```
{
 "query": "...",
 "cache_hit": true,
 "matched_query": "...",
 "similarity_score": 0.91,
 "result": "...",
 "dominant_cluster": 3
}
```

---

### 2. Cache Statistics

GET `/cache/stats`

Example Response:

```
{
 "total_entries": 42,
 "hit_count": 17,
 "miss_count": 25,
 "hit_rate": 0.405
}
```

---

### 3. Clear Cache

DELETE `/cache`

Resets cache and statistics.

---

## Installation & Setup

### 1. Clone the repository

```
git clone https://github.com/YOUR_USERNAME/trademarkia-semantic-search.git
cd trademarkia-semantic-search
```

### 2. Create virtual environment

```
python -m venv venv
```

Activate environment

Windows

```
venv\Scripts\activate
```

Linux/Mac

```
source venv/bin/activate
```

---

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

### 4. Build the vector database and clustering model

```
python -m src.setup_pipeline
```

This step:

• Loads dataset
• Generates embeddings
• Builds FAISS index
• Trains fuzzy clustering model
• Saves models to `models/`

---

### 5. Run the API server

```
uvicorn api.main:app --reload
```

Open in browser:

```
http://127.0.0.1:8000/docs
```

---

## Technologies Used

* Python
* FastAPI
* Sentence Transformers
* FAISS
* Scikit-learn
* NumPy
* Uvicorn

---

## Project Structure

```
trademarkia-semantic-search
│
├── api
│   └── main.py
│
├── src
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── embedding_model.py
│   ├── vector_store.py
│   ├── clustering.py
│   ├── semantic_cache.py
│   ├── search_engine.py
│   └── setup_pipeline.py
│
├── models
│   ├── faiss_index.bin
│   ├── embeddings.npy
│   └── gmm_model.pkl
│
├── requirements.txt
└── README.md
```

---

## Future Improvements

• Cluster visualization using UMAP
• Distributed vector search
• Cache optimization for large-scale deployments
• Docker containerization

---

## Author

AI/ML Engineering Assignment Submission for **Trademarkia**
