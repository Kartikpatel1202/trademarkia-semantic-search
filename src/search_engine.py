class SearchEngine:

    def __init__(self, vector_store, documents):

        self.vector_store = vector_store
        self.documents = documents

    def search(self, query_embedding):

        distances, indices = self.vector_store.search(
            query_embedding,
            k=5
        )

        results = []

        for idx in indices:

            results.append(self.documents[idx])

        return results