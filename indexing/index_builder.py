from indexing.index_server import IndexServer


# Manages and indexes the given documents.
class IndexBuilder:

    def __init__(self, ngram_generator, ngram_postings_list):
        self.ngram_generator = ngram_generator
        self.ngram_postings_list = ngram_postings_list
        self.doc_id_to_url = {}
        self.doc_id_to_text = {}

    def add_document(self, doc_url, doc_text):
        doc_id = hash(doc_url)
        self.doc_id_to_url[doc_id] = doc_url
        self.doc_id_to_text[doc_id] = doc_text
        self.ngram_postings_list.add_to_representation(
            doc_id, self.ngram_postings_list.aggregate(self.ngram_generator.generate_ngrams(doc_text)))

    def generate_index_server(self):
        return IndexServer(self.ngram_generator,
                           self.ngram_postings_list,
                           self.doc_id_to_url,
                           self.doc_id_to_text)

    def size(self):
        return len(self.doc_id_to_url)
