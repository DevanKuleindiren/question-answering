import numpy as np


class GloveEmbedder:

    def __init__(self, glove_vectors_file):
        # Deserialize GloVe vectors
        self.glove_word_vector_map = {}
        with open(glove_vectors_file, "r", encoding="utf8") as glove:
            for line in glove:
                name, vector = tuple(line.split(" ", 1))
                self.glove_word_vector_map[name] = np.fromstring(vector, sep=" ")

        word_vectors = []
        for item in self.glove_word_vector_map.items():
            word_vectors.append(item[1])
        word_vector_stack = np.vstack(word_vectors)

        # Gather statistics about the distribution of the word vectors, to be used for generating vectors for unknown
        # words.
        self.word_vector_variance = np.var(word_vector_stack, 0)
        self.word_vector_mean = np.mean(word_vector_stack, 0)
        self.random_state = np.random.RandomState()

    def embed(self, word):
        """Embeds the given word.

        :param word: The word to be embedded.
        :return: A vector representing the word.
        """
        if word not in self.glove_word_vector_map:
            # If the word is unknown, then we generate a new vectorization from the (Gaussian-approximated) distribution
            # of the original GloVe vectorizations.
            self.glove_word_vector_map[word] = self.random_state.multivariate_normal(self.word_vector_mean,
                                                                                     np.diag(self.word_vector_variance))
        return self.glove_word_vector_map[word]
