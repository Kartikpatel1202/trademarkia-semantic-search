from src.clustering import FuzzyCluster
from src.data_loader import load_dataset
from src.embedding_model import EmbeddingModel
from src.vector_store import VectorStore


print("Loading dataset...")
docs, labels = load_dataset()

print("Generating embeddings...")
model = EmbeddingModel()

embeddings = model.encode_documents(docs)

dimension = embeddings.shape[1]

print("Building vector database...")
vector_store = VectorStore(dimension)

vector_store.add_embeddings(embeddings)

print("Vector database ready!")
print("Training fuzzy clustering model...")

cluster_model = FuzzyCluster(n_clusters=15)

cluster_model.train(embeddings)

print("Clustering model trained successfully!")

import joblib
import faiss
import numpy as np

print("Saving models...")

faiss.write_index(vector_store.index, "models/faiss_index.bin")

np.save("models/embeddings.npy", embeddings)

joblib.dump(cluster_model.model, "models/gmm_model.pkl")

print("Models saved successfully!")