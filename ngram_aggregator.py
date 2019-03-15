from math import sqrt


class NgramAggregator:

    def __init__(self):
        self.ngram_to_docs = {}
        self.document_magnitudes = {}
        self.doc_frequency = {}

    def aggregate(self, n_grams, shouldAggregate=True):
        s = {}
        for n_gram in n_grams:
            key = hash(n_gram)
            s[key] = s.get(key, 0) + 1
            if s[key] == 1 and shouldAggregate:
                self.doc_frequency[key] = 1 + self.doc_frequency.get(key, 0)
        return s

    def add_to_representation(self, doc_id, s):
        magnitude = 0
        for (hashedNgram, tf) in s.items():
            ls = self.ngram_to_docs.get(hashedNgram, [])
            ls.append((doc_id, tf))
            self.ngram_to_docs[hashedNgram] = ls
            tf_idf = tf / self.doc_frequency[hashedNgram]
            magnitude += tf_idf * tf_idf
        self.document_magnitudes[doc_id] = sqrt(magnitude)

    def get_representation(self):
        return self.ngram_to_docs

    def get_magnitude(self, doc_id):
        if doc_id in self.document_magnitudes:
            return self.document_magnitudes[doc_id]
        return 0
