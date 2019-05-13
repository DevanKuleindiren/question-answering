import numpy as np


class FeatureVectorGenerator():

    def __init__(self, word_embedding_generator, lemmatizer, pos_tagger, ner_tagger, tf_supplier):
        self.word_embedding_generator = word_embedding_generator
        self.lemmatizer = lemmatizer
        self.pos_tagger = pos_tagger
        self.ner_tagger = ner_tagger
        self.tf_supplier = tf_supplier

    def f_exact_match(self, token, question, lemmatized_question):
        found_original_form = False
        found_lowercase_form = False
        found_lemma_form = False
        for q in question:
            if q == token:
                found_original_form = True
            if q.lower() == token.lower():
                found_lowercase_form = True

        lemmatized_word = self.lemmatizer(token)
        for q in lemmatized_question:
            if lemmatized_word == q:
                found_lemma_form = True

        return [found_original_form, found_lowercase_form, found_lemma_form]

    def generate_lemmatized_question(self, question):
        lemmatized_question = []
        for q in question:
            lemmatized_question.append(self.lemmatizer(q))
        return lemmatized_question

    def generate_feature_vector_for_paragraph(self, paragraph, question):
        # Precompute the lemmatization for the question to avoid re-computing it multiple times.
        lemmatized_question = self.generate_lemmatized_question(question)
        pos_tags = self.pos_tagger(paragraph)
        ner_tags = self.ner_tagger(paragraph)
        feature_vector = []
        for (token, pos_tag, ner_tag) in zip(paragraph, pos_tags, ner_tags):
            feature_vector.append(self.generate_feature_vector(token, pos_tag, ner_tag, question, lemmatized_question))
        return feature_vector

    def f_token(self, token, pos_tag, ner_tag):
        return [self.tf_supplier(token), pos_tag, ner_tag]

    def generate_feature_vector(self, token, pos_tag, ner_tag, question, lemmatized_question):
        return [(self.word_embedding_generator(token),
                               self.f_exact_match(token, question, lemmatized_question),
                               self.f_token(token, pos_tag, ner_tag,))]
