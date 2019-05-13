from nltk.stem import PorterStemmer
from nltk.tag import pos_tag
from nltk import ne_chunk
from chen_et_al.utils import FeatureVectorGenerator
from nltk.chunk import tree2conlltags


def create_nltk_feature_generator(word_embedding_vector_generator, tf_supplier):
    # nltk.download()
    return FeatureVectorGenerator(
        word_embedding_generator=word_embedding_vector_generator,
        lemmatizer=PorterStemmer(),# TODO: Find out why word net lemmatization does not work.
        pos_tagger=lambda x: [tag for (_, tag) in pos_tag(x)],
        ner_tagger=lambda x: [x.startswith("B-") for (_, _, x) in tree2conlltags(ne_chunk(pos_tag(x)))],
        tf_supplier=tf_supplier
    )
