import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SemanticCache:

    def __init__(self, threshold=0.85):

        self.cache = []
        self.threshold = threshold

        self.hit_count = 0
        self.miss_count = 0

    def lookup(self, query_embedding):

        best_score = 0
        best_match = None

        for item in self.cache:

            score = cosine_similarity(
                [query_embedding],
                [item["embedding"]]
            )[0][0]

            if score > best_score:

                best_score = score
                best_match = item

        if best_score >= self.threshold:

            self.hit_count += 1

            return True, best_match, best_score

        self.miss_count += 1

        return False, None, best_score

    def add(self, query, embedding, result, cluster):

        self.cache.append({
            "query": query,
            "embedding": embedding,
            "result": result,
            "cluster": cluster
        })

    def stats(self):

        total = len(self.cache)

        hit_rate = self.hit_count / \
            (self.hit_count + self.miss_count + 1e-9)

        return {
            "total_entries": total,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate
        }

    def clear(self):

        self.cache = []
        self.hit_count = 0
        self.miss_count = 0