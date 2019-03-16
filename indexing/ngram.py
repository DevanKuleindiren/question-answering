class Ngram:

    def __init__(self, tokens):
        self.tokens = tokens

    def __hash__(self):
        h = 0
        c = 1
        for t in self.tokens:
            h += c * hash(t)
            c += 1
        return h

    def __eq__(self, other):
        return self.tokens == other.tokens

    def __str__(self):
        return str(self.tokens)

    def __repr__(self):
        return repr(self.tokens)