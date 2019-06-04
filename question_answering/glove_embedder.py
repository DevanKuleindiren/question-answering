import numpy as np
import tensorflow as tf


class GloveEmbedder:

    def __init__(self, glove_vectors_file):
        embedding_dim = 0
        with open(glove_vectors_file, "r", encoding="utf8") as glove_file:
            first_line = glove_file.readline()
            embedding_dim = first_line.count(" ")

        self.embedding_matrix = np.empty(shape=(0, embedding_dim))
        self.word_to_id = {"<unk>": 0}

        # Deserialize the GloVe vectors
        unknown_token_vector = np.zeros(shape=embedding_dim)
        current_id = 1
        with open(glove_vectors_file, "r", encoding="utf8") as glove_file:
            for line in glove_file:
                word, vector_string = tuple(line.split(" ", 1))
                vector = np.fromstring(vector_string, sep=" ")
                unknown_token_vector = np.add(unknown_token_vector, vector)
                self.embedding_matrix = np.append(self.embedding_matrix, np.expand_dims(vector, axis=0), axis=0)
                self.word_to_id[word] = current_id
                current_id += 1
        unknown_token_vector = unknown_token_vector / self.embedding_matrix.shape[0]

        # Append the unknown token vector to the start of the embedding matrix.
        self.embedding_matrix = np.insert(self.embedding_matrix, 0, unknown_token_vector, axis=0)
        self.vocab_size, self.embedding_dim = self.embedding_matrix.shape

    def get_embedding_layer(self):
        return tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim,
                                         embeddings_initializer=tf.constant_initializer(self.embedding_matrix),
                                         trainable=False)

    def get_ids(self, words):
        word_ids = np.array([])
        for word in words:
            if word in self.word_to_id:
                word_ids = np.append(word_ids, self.word_to_id[word])
            else:
                word_ids = np.append(word_ids, 0)
        return word_ids
