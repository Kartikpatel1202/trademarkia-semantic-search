from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:

    def __init__(self):

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def encode_documents(self, docs):

        embeddings = self.model.encode(
            docs,
            show_progress_bar=True
        )

        return np.array(embeddings)

    def encode_query(self, query):

        return self.model.encode([query])[0]