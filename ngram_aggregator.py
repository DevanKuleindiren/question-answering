from math import sqrt


class NgramAggregator:

    def __init__(self):
        self.ngram_to_docs = {}
        self.document_magnitudes = {}
        self.doc_frequency = {}

    def aggregate(self, n_grams, aggregate_doc_frequency=True):
        s = {}
        for n_gram in n_grams:
            key = hash(n_gram)
            s[key] = s.get(key, 0) + 1
            if s[key] == 1 and aggregate_doc_frequency:
                self.doc_frequency[key] = 1 + self.doc_frequency.get(key, 0)
        return s

    def add_to_representation(self, doc_id, s):
        magnitude = 0
        for (hashedNgram, tf) in s.items():
            ls = self.ngram_to_docs.get(hashedNgram, [])
            ls.append((doc_id, tf))
            self.ngram_to_docs[hashedNgram] = ls
            # Add 1 to the document frequency to avoid division by 0.
            tf_idf = tf / (self.doc_frequency[hashedNgram] + 1)
            magnitude += tf_idf * tf_idf
        self.document_magnitudes[doc_id] = sqrt(magnitude)

    def get_representation(self):
        return self.ngram_to_docs

    def get_magnitude(self, doc_id):
        if doc_id in self.document_magnitudes:
            return self.document_magnitudes[doc_id]
        return 0

    def merge_results(self, query_ngram_to_docs, query_magnitude):
        scores = {}
        for (query_ngram, docs) in query_ngram_to_docs:
            query_ngram_df = self.doc_frequency[query_ngram]
            for (doc, tf) in docs:
                # Add 1 to the document frequency to avoid division by 0.
                scores[doc] = scores.get(doc, 0) + tf / (query_ngram_df + 1)
        for doc in scores:
            scores[doc] /= self.document_magnitudes[doc] * query_magnitude
        return scores

    def query(self, query_ngrams):
        # Filter out duplicates
        query_ngrams = self.aggregate(query_ngrams, aggregate_doc_frequency=False).keys()
        query_magnitude = sqrt(len(query_ngrams))
        query_ngram_to_docs = []
        for query_ngram in query_ngrams:
            if query_ngram in self.ngram_to_docs:
                query_ngram_to_docs.append((query_ngram, self.ngram_to_docs[query_ngram]))
        return self.merge_results(query_ngram_to_docs, query_magnitude)
