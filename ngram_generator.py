class NgramGenerator:

    def __init__(self, n):
        self.n = n

    def generate_ngrams(self, doc_content):
        words = doc_content.split()

        ngrams = []
        for i in range(0, len(words) - self.n + 1):
            ngram = []
            for j in range(0, self.n):
                ngram.append(words[i + j])
            ngrams.append(ngram)
        return ngrams
