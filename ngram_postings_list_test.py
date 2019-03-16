import unittest

from math import sqrt
from ngram import Ngram
from ngram_postings_list import NgramPostingsList


class TestNgramAggregator(unittest.TestCase):

    def setUp(self):
        self.agg = NgramPostingsList()
        self.ngram1 = Ngram(["hello", "world"])
        self.ngram2 = Ngram(["again", "another"])
        self.ngram3 = Ngram(["world", "hello"])
        self.doc1 = [self.ngram1, self.ngram2, self.ngram1]
        self.doc2 = [self.ngram1, self.ngram2, self.ngram2, self.ngram3, self.ngram2]
        self.query = [self.ngram1, self.ngram2]
        self.s1 = self.agg.aggregate(self.doc1)
        self.s2 = self.agg.aggregate(self.doc2)

    def test_aggregate(self):
        self.assertDictEqual(self.s1, {hash(self.ngram1): 2, hash(self.ngram2): 1})
        self.assertDictEqual(self.s2, {hash(self.ngram1): 1, hash(self.ngram2): 3, hash(self.ngram3): 1})

    def test_add_to_representation(self):
        doc1_id = "doc1"
        doc2_id = "doc2"
        self.agg.add_to_representation(doc1_id, self.s1)
        self.assertDictEqual(self.agg.get_representation(), {hash(self.ngram1): [(doc1_id, 2)],
                                                             hash(self.ngram2): [(doc1_id, 1)]})

        self.agg.add_to_representation(doc2_id, self.s2)
        self.assertDictEqual(self.agg.get_representation(), {hash(self.ngram1): [(doc1_id, 2), (doc2_id, 1)],
                                                             hash(self.ngram2): [(doc1_id, 1), (doc2_id, 3)],
                                                             hash(self.ngram3): [(doc2_id, 1)]})

    def test_get_magnitude(self):
        doc1_id = "doc1"
        self.agg.add_to_representation(doc1_id, self.s1)
        self.assertEqual(self.agg.get_magnitude("blah"), 0)
        # |d1| = √((2/3)^2 + (1/3)^2) = √5 / 3
        self.assertAlmostEqual(self.agg.get_magnitude(doc1_id), sqrt(5) / 3, places=5)

    def test_single_doc_query(self):
        doc1_id = "doc1"
        self.agg.add_to_representation(doc1_id, self.s1)
        # |q| = √2, |d1| = √5 / 3
        # Therefore, cosine_similarity(q, d1) = (2/3 + 1/3) / (|q| * |d1|)
        #                                     = 1 / (√2 * √5/3)
        #                                     = 3 / √10
        query_result = self.agg.query(self.query)
        self.assertEqual(len(query_result), 1)
        self.assertTrue(doc1_id in query_result)
        self.assertAlmostEqual(query_result[doc1_id], 3 / sqrt(10))

    def test_multiple_doc_query(self):
        doc1_id = "doc1"
        doc2_id = "doc2"
        self.agg.add_to_representation(doc1_id, self.s1)
        self.agg.add_to_representation(doc2_id, self.s2)
        # d1 score is same as above.
        # |q| = √2, |d2| = √((1/3)^2 + (3/3)^2 + (1/2)^2) = √(49/36) = 7/6
        # Therefore, cosine_similarity(q, d2) = (1/3 + 3/3) / (|q| * |d2|)
        #                                     = 4/3 / (√2 * 7/6)
        #                                     = 8 / (7 * √2)
        query_result = self.agg.query(self.query)
        self.assertEqual(len(query_result), 2)
        self.assertTrue(doc1_id in query_result)
        self.assertTrue(doc2_id in query_result)
        self.assertAlmostEqual(query_result[doc1_id], 3 / sqrt(10))
        self.assertAlmostEqual(query_result[doc2_id], 8 / (7 * sqrt(2)))
