import spacy


# Requires running python -m spacy download en
def create_spacy_feature_generator(word_embedding_vector_generator, tf_supplier):
    # nlp = spacy.load('C:\Users\Dimitris\AppData\Local\Continuum\anaconda3\lib\site-packages\spacy\data\en')
    return FeatureVectorGenerator(
        word_embedding_generator=word_embedding_vector_generator,
        lemmatizer=PorterStemmer(),# TODO: Find out why word net lemmatization does not work.
        pos_tagger=lambda x: [tag for (_, tag) in pos_tag(x)],
        ner_tagger=lambda x: [x.startswith("B-") for (_, _, x) in tree2conlltags(ne_chunk(pos_tag(x)))],
        tf_supplier=tf_supplier
    )
