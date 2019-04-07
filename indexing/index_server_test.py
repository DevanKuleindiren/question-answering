import unittest

from indexing.index_builder import IndexBuilder
from indexing.ngram_generator import NgramGenerator
from indexing.ngram_postings_list import NgramPostingsList


class TestIndexServer(unittest.TestCase):

    def setUp(self):
        self.doc_a = "This is an example of a document"
        self.doc_a_url = "doc-a-url"
        self.doc_b = "This is an example of another document"
        self.doc_b_url = "doc-b-url"
        self.index_builder = IndexBuilder(NgramGenerator(1), NgramPostingsList())
        self.index_builder.add_document(doc_text=self.doc_a, doc_url=self.doc_a_url)
        self.index_builder.add_document(doc_text=self.doc_b, doc_url=self.doc_b_url)
        self.index_server = self.index_builder.generate_index_server()

    def test_query_another_returns_doc_b(self):
        (doc_id, _) = self.index_server.query("another")[0]

        self.assertEqual(doc_id, hash(self.doc_b_url))

    def test_query_with_url_another_returns_doc_b(self):
        (doc_id, _, doc_url) = self.index_server.query("another", include_urls=True)[0]

        self.assertEqual(doc_url, self.doc_b_url)
        self.assertEqual(doc_id, hash(doc_url))

    def test_query_with_text_another_returns_doc_b(self):
        (doc_id, _, doc_text) = self.index_server.query("another", include_text=True)[0]

        self.assertEqual(doc_text, self.doc_b)

    def test_query_example_returns_both_docs(self):
        results_list = self.index_server.query("example")
        (id_0, _) = results_list[0]
        (id_1, _) = results_list[1]
        results_ids = [id_0, id_1]

        # Find out why we cannot use assertItemsEqual.
        self.assertEqual(sorted(results_ids), sorted([hash(self.doc_a_url), hash(self.doc_b_url)]))

    def test_query_example_with_limit_examples_returns_one_doc(self):
        self.assertEqual(len(self.index_server.query("example", limit_to_top=1)), 1)
