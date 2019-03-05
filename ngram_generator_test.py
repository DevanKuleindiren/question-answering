import unittest

import ngram_generator


class TestNgramGenerator(unittest.TestCase):

    def setUp(self):
        self.test_doc = "hello world and hello world again hello hello world"

    def test_generate_bigram(self):
        bigram_generator = ngram_generator.NgramGenerator(2)
        self.assertEqual(bigram_generator.generate_ngrams(self.test_doc),
                         [["hello", "world"], ["world", "and"], ["and", "hello"], ["hello", "world"],
                          ["world", "again"], ["again", "hello"], ["hello", "hello"], ["hello", "world"]])
