from indexing.ngram_generator import NgramGenerator
from indexing.ngram_postings_list import NgramPostingsList

doc1 = "hello world hello world"
doc2 = "hello world hello again"
query = "hello world"

bigram_generator = NgramGenerator(2)
bigram_postings_list = NgramPostingsList()
bigram_postings_list.add_to_representation("doc1", bigram_postings_list.aggregate(bigram_generator.generate_ngrams(doc1)))
bigram_postings_list.add_to_representation("doc2", bigram_postings_list.aggregate(bigram_generator.generate_ngrams(doc2)))

print(bigram_postings_list.query(bigram_generator.generate_ngrams(query)))
