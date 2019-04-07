import argparse
import glob
import json
import string

from indexing.ngram_generator import NgramGenerator
from indexing.ngram_postings_list import NgramPostingsList
from indexing.index_builder import IndexBuilder

flag_parser = argparse.ArgumentParser(description="Send queries to a specified list of documents.")
flag_parser.add_argument(
    '--docs_file_path', help='File path to the JSON file(s) containing documents. Can be in regex format.',
    dest='docs_file_path')


def main():
    flags = flag_parser.parse_args()

    if flags.docs_file_path is None:
        print('--docs_file_path needs to be set.')
        return

    index_builder = IndexBuilder(NgramGenerator(2), NgramPostingsList())

    docs_file_paths = sorted(glob.glob(flags.docs_file_path))

    for docs_file_path in docs_file_paths:
        with open(docs_file_path, 'r', encoding="utf-8") as docs_file:
            for doc_json in docs_file:
                json_data = json.loads(doc_json)
                index_builder.add_document(
                    json_data['url'],
                    json_data['text'].translate(str.maketrans('', '', string.punctuation)).lower())
        print('Indexed: %s' % docs_file_path)
    print('Indexed %d docs' % index_builder.size())

    index_server = index_builder.generate_index_server()

    while True:
        query_text = input('Query: ').lower()
        print(index_server.query(query_text, include_urls=True))


if __name__ == '__main__':
    main()
