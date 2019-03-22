import argparse
import glob
import json
import string

from indexing.ngram_generator import NgramGenerator
from indexing.ngram_postings_list import NgramPostingsList

flag_parser = argparse.ArgumentParser(description="Send queries to a specified list of documents.")
flag_parser.add_argument(
    '--docs_file_path', help='File path to the JSON file(s) containing documents. Can be in regex format.',
    dest='docs_file_path')


def main():
    flags = flag_parser.parse_args()

    if flags.docs_file_path is None:
        print('--docs_file_path needs to be set.')
        return

    bigram_generator = NgramGenerator(2)
    bigram_postings_list = NgramPostingsList()
    doc_index = {}

    docs_file_paths = sorted(glob.glob(flags.docs_file_path))

    for docs_file_path in docs_file_paths:
        with open(docs_file_path) as docs_file:
            for doc_json in docs_file:
                json_data = json.loads(doc_json)
                doc_url = json_data['url']
                doc_id = hash(doc_url)
                doc_index[doc_id] = doc_url
                doc_text = json_data['text'].translate(str.maketrans('', '', string.punctuation))
                bigram_postings_list.add_to_representation(
                    doc_id, bigram_postings_list.aggregate(bigram_generator.generate_ngrams(doc_text)))
        print('Indexed: %s' % docs_file_path)
    print('Indexed %d docs' % len(doc_index))

    while True:
        query_text = input('Query: ')
        query_bigrams = bigram_generator.generate_ngrams(query_text)
        top_documents = bigram_postings_list.query(query_bigrams)
        sorted_top_documents = sorted(top_documents.items(), key=lambda kv: kv[1], reverse=True)
        sorted_top_documents_with_urls = [(doc_id, doc_index[doc_id], score)
                                          for (doc_id, score) in sorted_top_documents]
        print(sorted_top_documents_with_urls)


if __name__ == '__main__':
    main()
