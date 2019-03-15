class NgramAggregator:

    def __init__(self):
        self.idf = {}
        self.indexer = {}

    def aggregate(self, n_grams):
        s = {}
        for n_gram in n_grams:
            key = hash(n_gram)
            s[key] = s.get(key, 0) + 1
            if s[key] == 1:
                self.idf[key] = self.idf.get(key, 0) + 1
        return s

    def add_to_representation(self, doc_id, s):
        for (hashedNgram, tf) in s.items():
            ls = self.indexer.get(hashedNgram, [])
            ls.append((doc_id, tf))
            self.indexer[hashedNgram] = ls

    def get_representation(self):
        return self.indexer
