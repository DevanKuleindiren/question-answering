

# Responsible for responding to queries.
class IndexServer:

    def __init__(self,
                 ngram_generator,
                 ngram_postings_list,
                 doc_id_to_url,
                 doc_id_to_text):
        self.ngram_generator = ngram_generator
        self.ngram_postings_list = ngram_postings_list
        self.doc_id_to_url = doc_id_to_url
        self.doc_id_to_text = doc_id_to_text

    # Returns the results matching the given query.
    # Note: the queries should be processed the same way that the documents were processed.
    def query(self, query_text, include_urls=False, limit_to_top=None, include_text=False):
        query_ngrams = self.ngram_generator.generate_ngrams(query_text)
        top_documents = self.ngram_postings_list.query(query_ngrams)
        sorted_top_documents = sorted(top_documents.items(), key=lambda kv: kv[1], reverse=True)
        results_list = []
        for (doc_id, score) in sorted_top_documents:
            ls = [doc_id, score]
            if include_urls:
                ls.append(self.doc_id_to_url[doc_id])
            if include_text:
                ls.append(self.doc_id_to_text[doc_id])
            results_list.append(tuple(ls))
        if limit_to_top is None:
            return results_list
        return results_list[:limit_to_top]

    def get_paragraphs(self, doc_id):
        non_empty_paragraphs = []
        if doc_id in self.doc_id_to_text:
            paragraphs = self.doc_id_to_text[doc_id].splitlines()
            # splitlines() may produce some empty lines where there are multiple newline characters in a row, so those
            # are filtered out.
            non_empty_paragraphs = list(filter(bool, paragraphs))
        return non_empty_paragraphs
