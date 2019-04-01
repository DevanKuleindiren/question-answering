from indexing.ngram import Ngram


class NgramGenerator:

    def __init__(self, n, include_unigrams=True):
        self.n = n
        self.include_unigrams = include_unigrams

    def generate_ngrams(self, doc_content):
        words = doc_content.split()

        ngrams = []

        if self.include_unigrams:
            for w in words:
                ngrams.append((Ngram(self.pad_with(w))))

        for i in range(0, len(words) - self.n + 1):
            ngram_tokens = []
            for j in range(0, self.n):
                ngram_tokens.append(words[i + j])
            ngrams.append(Ngram(ngram_tokens))
        return ngrams

    def pad_with(self, word):
        ls = [word]
        for i in range(self.n - 1):
            ls.append("<*>")
        return ls
