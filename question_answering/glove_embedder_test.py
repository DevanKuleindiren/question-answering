import numpy as np
import tempfile
import tensorflow as tf
import unittest
from os import path

from question_answering.glove_embedder import GloveEmbedder


class TestIndexServer(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def test_get_embedding_layer(self):
        temp_file_path = path.join(self.temp_dir, "test.txt")
        with open(temp_file_path, "w") as f:
            f.write("the 0.0 2.0 1.5\ncat -0.8 3.0 1.5")
        glove_embedder = GloveEmbedder(temp_file_path)
        embedding_layer = glove_embedder.get_embedding_layer()
        self.assertEqual(embedding_layer.output_dim, 3)
        input_word_ids = tf.constant(np.array([0, 1, 2]))
        output_embeddings = embedding_layer(input_word_ids)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output_embedding_vectors = sess.run(output_embeddings)
            np.testing.assert_array_almost_equal(output_embedding_vectors,
                                                 np.array([[-0.4, 2.5, 1.5], [0.0, 2.0, 1.5], [-0.8, 3.0, 1.5]]),
                                                 decimal=6)

    def test_get_ids(self):
        temp_file_path = path.join(self.temp_dir, "test.txt")
        with open(temp_file_path, "w") as f:
            f.write("the 0\ncat 8\nsat 9")
        glove_embedder = GloveEmbedder(temp_file_path)
        word_ids = glove_embedder.get_ids(["the", "dog", "cat", "sat"])
        np.testing.assert_array_almost_equal(word_ids, np.array([1, 0, 2, 3], dtype=np.float), decimal=6)
