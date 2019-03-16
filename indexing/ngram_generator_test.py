import unittest

from indexing import ngram_generator
from indexing.ngram import Ngram


class TestNgramGenerator(unittest.TestCase):

    def setUp(self):
        self.test_doc = "hello world and hello world again hello hello hello world"

    def test_generate_unigram(self):
        unigram_generator = ngram_generator.NgramGenerator(1)
        self.assertEqual(unigram_generator.generate_ngrams(self.test_doc),
                         [Ngram(["hello"]), Ngram(["world"]), Ngram(["and"]), Ngram(["hello"]), Ngram(["world"]),
                          Ngram(["again"]), Ngram(["hello"]), Ngram(["hello"]), Ngram(["hello"]), Ngram(["world"])])

    def test_generate_bigram(self):
        bigram_generator = ngram_generator.NgramGenerator(2)
        self.assertEqual(bigram_generator.generate_ngrams(self.test_doc),
                         [Ngram(["hello", "world"]), Ngram(["world", "and"]), Ngram(["and", "hello"]),
                          Ngram(["hello", "world"]), Ngram(["world", "again"]), Ngram(["again", "hello"]),
                          Ngram(["hello", "hello"]), Ngram(["hello", "hello"]), Ngram(["hello", "world"])])

    def test_generate_trigram(self):
        trigram_generator = ngram_generator.NgramGenerator(3)
        self.assertEqual(trigram_generator.generate_ngrams(self.test_doc),
                         [Ngram(["hello", "world", "and"]), Ngram(["world", "and", "hello"]),
                          Ngram(["and", "hello", "world"]), Ngram(["hello", "world", "again"]),
                          Ngram(["world", "again", "hello"]), Ngram(["again", "hello", "hello"]),
                          Ngram(["hello", "hello", "hello"]), Ngram(["hello", "hello", "world"])])
