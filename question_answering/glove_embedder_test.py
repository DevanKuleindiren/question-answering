import numpy as np
import tempfile
import unittest
from os import path

from question_answering.glove_embedder import GloveEmbedder


class TestIndexServer(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def test_embed_known_word(self):
        temp_file_path = path.join(self.temp_dir, "test.txt")
        with open(temp_file_path, "w") as f:
            f.write("the 0 0\ncat 1 1")
        glove_embedder = GloveEmbedder(temp_file_path)
        the_embedding = glove_embedder.embed("the")
        cat_embedding = glove_embedder.embed("cat")
        np.testing.assert_array_equal(the_embedding, np.array([0, 0]))
        np.testing.assert_array_equal(cat_embedding, np.array([1, 1]))

    def test_embed_unknown_word(self):
        temp_file_path = path.join(self.temp_dir, "test.txt")
        with open(temp_file_path, "w") as f:
            f.write("the 0 0\ncat 1 1")
        glove_embedder = GloveEmbedder(temp_file_path)
        dog_embedding = glove_embedder.embed("dog")
        print(dog_embedding)
        np.testing.assert_array_almost_equal(dog_embedding, np.array([0.5, 0.5]), decimal=0)
