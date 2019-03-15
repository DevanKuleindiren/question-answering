import unittest

from ngram import Ngram


class TestNgram(unittest.TestCase):

    def setUp(self):
        self.bigram1 = Ngram(["hello", "world"])
        self.bigram2 = Ngram(["world", "hello"])
        self.bigram3 = Ngram(["world", "again"])
        self.trigram1 = Ngram(["hello", "world", "again"])
        self.trigram2 = Ngram(["hello", "world", "again"])

    def test_equal(self):
        self.assertEqual(self.trigram1, self.trigram2)

    def test_not_equal_length(self):
        self.assertNotEqual(self.bigram1, self.trigram1)

    def test_not_equal_order(self):
        self.assertNotEqual(self.bigram1, self.bigram2)

    def test_not_equal(self):
        self.assertNotEqual(self.bigram1, self.bigram3)
