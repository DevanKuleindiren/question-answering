import unittest

from indexing.index_builder import IndexBuilder
from indexing.ngram_generator import NgramGenerator
from indexing.ngram_postings_list import NgramPostingsList


class TestIndexBuilder(unittest.TestCase):

    def setUp(self):
        self.doc_a = "This is an example of a document"
        self.doc_a_url = "doc-a-url"
        self.doc_b = "This is an example of another document"
        self.doc_b_url = "doc-b-url"
        self.index_builder = IndexBuilder(NgramGenerator(1), NgramPostingsList())
        self.index_builder.add_document(doc_text=self.doc_a, doc_url=self.doc_a_url)
        self.index_builder.add_document(doc_text=self.doc_b, doc_url=self.doc_b_url)

    def test_size(self):
        self.assertEqual(self.index_builder.size(), 2)
