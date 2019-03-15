from ngram_generator import NgramGenerator
from ngram_aggregator import NgramAggregator

doc1 = "hello world hello world"
doc2 = "hello world hello again"

bigram_generator = NgramGenerator(2)
bigram_aggregator = NgramAggregator()
bigram_aggregator.add_to_representation("doc1", bigram_aggregator.aggregate(bigram_generator.generate_ngrams(doc1)))
bigram_aggregator.add_to_representation("doc2", bigram_aggregator.aggregate(bigram_generator.generate_ngrams(doc2)))

print(bigram_aggregator.get_representation())
