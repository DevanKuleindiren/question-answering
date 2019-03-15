from math import sqrt


class NgramAggregator:

    def __init__(self):
        self.idf = {}
        self.ngram_to_docs = {}
        self.document_magnitudes = {}

    def aggregate(self, n_grams):
        s = {}
        for n_gram in n_grams:
            key = hash(n_gram)
            s[key] = s.get(key, 0) + 1
            if s[key] == 1:
                self.idf[key] = self.idf.get(key, 0) + 1
        return s

    def add_to_representation(self, doc_id, s):
        magnitude = 0
        for (hashedNgram, tf) in s.items():
            ls = self.ngram_to_docs.get(hashedNgram, [])
            ls.append((doc_id, tf))
            self.ngram_to_docs[hashedNgram] = ls
            magnitude += tf * tf
        self.document_magnitudes[doc_id] = sqrt(magnitude)

    def get_representation(self):
        return self.ngram_to_docs

    def get_magnitude(self, doc_id):
        if doc_id in self.document_magnitudes:
            return self.document_magnitudes[doc_id]
        return 0
