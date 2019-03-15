import unittest

from math import sqrt
from ngram import Ngram
from ngram_aggregator import NgramAggregator


class TestNgramAggregator(unittest.TestCase):

    def setUp(self):
        self.agg = NgramAggregator()
        self.ngram1 = Ngram(["hello", "world"])
        self.ngram2 = Ngram(["again", "another"])
        self.ngram3 = Ngram(["world", "hello"])
        self.doc1 = [self.ngram1, self.ngram2, self.ngram1]
        self.doc2 = [self.ngram1, self.ngram2, self.ngram2, self.ngram3]
        self.s1 = self.agg.aggregate(self.doc1)
        self.s2 = self.agg.aggregate(self.doc2)

    def test_aggregate(self):
        self.assertDictEqual(self.s1, {hash(self.ngram1): 2, hash(self.ngram2): 1})
        self.assertDictEqual(self.s2, {hash(self.ngram1): 1, hash(self.ngram2): 2, hash(self.ngram3): 1})

    def test_add_to_representation(self):
        doc1_id = "doc1"
        doc2_id = "doc2"
        self.agg.add_to_representation(doc1_id, self.s1)
        self.assertDictEqual(self.agg.get_representation(), {hash(self.ngram1): [(doc1_id, 2)],
                                                             hash(self.ngram2): [(doc1_id, 1)]})

        self.agg.add_to_representation(doc2_id, self.s2)
        self.assertDictEqual(self.agg.get_representation(), {hash(self.ngram1): [(doc1_id, 2), (doc2_id, 1)],
                                                             hash(self.ngram2): [(doc1_id, 1), (doc2_id, 2)],
                                                             hash(self.ngram3): [(doc2_id, 1)]})

    def test_get_magnitude(self):
        doc1_id = "doc1"
        self.agg.add_to_representation(doc1_id, self.s1)
        self.assertEqual(self.agg.get_magnitude("blah"), 0)
        self.assertEqual(self.agg.get_magnitude(doc1_id), sqrt(2 * 2 / (2 * 2) + 1 * 1 / (2 * 2)))
