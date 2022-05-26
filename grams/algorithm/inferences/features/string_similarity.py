from rltk.similarity import hybrid_jaccard_similarity


class StringSimilarity:
    @staticmethod
    def hybrid_jaccard_similarity(s1: str, s2: str) -> float:
        return hybrid_jaccard_similarity(set(s1.split(" ")), set(s2.split(" ")))
