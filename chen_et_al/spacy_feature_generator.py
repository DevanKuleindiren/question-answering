import spacy
from chen_et_al.utils import FeatureVectorGenerator


# Requires running python -m spacy download en
def create_spacy_feature_generator(word_embedding_vector_generator, tf_supplier):
    nlp = spacy.load('en_core_web_sm')
    return FeatureVectorGenerator(
        word_embedding_generator=word_embedding_vector_generator,
        lemmatizer=lambda x: nlp(x)[0].lemma_,
        pos_tagger=lambda x: [token.tag_ for token in nlp(" ".join(x))],
        ner_tagger=lambda x: [(token in list(map(lambda t: str(t), nlp(" ".join(x)).ents))) for token in x],
        tf_supplier=tf_supplier
    )
